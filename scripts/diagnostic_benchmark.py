"""Read-only ALPR baseline; all generated files stay in the diagnostic output.

Run with the project interpreter: python -B scripts/diagnostic_benchmark.py
Phases are restartable; production modules are instrumented only in memory.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]


def serial(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=serial), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, default=serial) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def images(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if folder.exists() else []


def decode(path):
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def pixel_sha(im):
    return hashlib.sha256(str(im.shape).encode() + im.tobytes()).hexdigest()


def normalized(text):
    # Symmetric evaluation normalization, with NO OCR-confusion repair.
    return re.sub('[^A-Z0-9]', '', text.upper())


def overlap(a, b):
    iw = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    ba = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return inter / (aa + ba - inter) if aa + ba - inter > 0 else 0., inter / ba if ba else 0.


def match(preds, gt, threshold=0.5):
    """Confidence-ordered, class-aware, greedy one-to-one matching.

    At each prediction choose the highest-IoU unmatched GT. Stable index breaks
    confidence ties. This convention is recorded (not claimed to be COCOeval).
    """
    used, matches = set(), []
    for i in sorted(range(len(preds)), key=lambda i: (-preds[i]['confidence'], i)):
        choices = [(overlap(preds[i]['xyxy'], g['xyxy'])[0], j) for j, g in enumerate(gt)
                   if j not in used and preds[i]['class_id'] == g['class_id']]
        best_iou, j = max(choices, default=(0., -1))
        if j >= 0 and best_iou >= threshold:
            used.add(j)
            matches.append({'prediction_index': i, 'gt_index': j, 'iou': best_iou})
    return matches


def evaluate(records, threshold, iou=0.5):
    tp = fp = fn = 0
    pairs = []
    for r in records:
        if not r['annotation_scorable'] or r.get('decode_error'):
            continue
        preds = [p for p in r['predictions'] if p['confidence'] > threshold]
        ms = match(preds, r['gt'], iou)
        tp += len(ms)
        fp += len(preds) - len(ms)
        fn += len(r['gt']) - len(ms)
        pairs.extend(m['iou'] for m in ms)
    return {'confidence_threshold': threshold, 'iou_threshold': iou, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0., 'recall': tp / (tp + fn) if tp + fn else 0.,
            'matched_mean_iou': sum(pairs) / len(pairs) if pairs else None}


def audit(args):
    import cv2
    out, folder = args.output, args.images
    label_dir = args.labels
    records, manifest = [], {}
    for path in images(folder):
        name = path.relative_to(folder).as_posix()
        label = label_dir / Path(name).with_suffix('.txt')
        r = {'filename': name, 'image_path': relative(path), 'label_path': relative(label),
             'image_sha256': sha(path), 'issues': [], 'gt': [], 'annotation_scorable': False}
        manifest[relative(path)] = r['image_sha256']
        try:
            im = decode(path)
            if im is None:
                raise ValueError('OpenCV decode returned None')
            h, w = im.shape[:2]
            r.update(width=w, height=h, dtype=str(im.dtype), pixel_sha256=pixel_sha(im))
        except Exception as exc:
            r['decode_error'] = str(exc)
            r['issues'].append('corrupt_or_unreadable_image')
        if not label.exists():
            r['annotation_status'] = 'missing_unknown_not_assumed_negative'
            r['issues'].append('missing_label_annotation_status_unknown')
        else:
            manifest[relative(label)] = sha(label)
            try:
                lines = label.read_text(encoding='utf-8-sig').splitlines()
                malformed = False
                for lineno, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        r['issues'].append(f'label_line_{lineno}:expected_5_fields_got_{len(parts)}')
                        malformed = True
                        continue
                    try:
                        cid, cx, cy, bw, bh = map(float, parts)
                        if not all(math.isfinite(v) for v in (cid, cx, cy, bw, bh)):
                            raise ValueError('nonfinite')
                        if cid != 0:
                            raise ValueError('class_id_must_be_0')
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            raise ValueError('invalid_normalized_center_or_size')
                        box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                        if min(box) < -1e-6 or max(box) > 1 + 1e-6:
                            raise ValueError('bbox_corners_outside_normalized_bounds')
                        if not r.get('decode_error'):
                            r['gt'].append({'class_id': 0, 'normalized_xywh': [cx, cy, bw, bh],
                                            'xyxy': [box[0] * w, box[1] * h, box[2] * w, box[3] * h]})
                    except ValueError as exc:
                        r['issues'].append(f'label_line_{lineno}:{exc}')
                        malformed = True
                r['annotation_scorable'] = not malformed and not r.get('decode_error')
                r['annotation_status'] = 'invalid' if malformed else ('present_nonempty' if r['gt'] else 'explicit_empty_assumed_background')
            except Exception as exc:
                r['annotation_status'] = 'unreadable_label'
                r['issues'].append(str(exc))
        records.append(r)
    labels = sorted(label_dir.rglob('*.txt'))
    expected = {r['label_path'] for r in records}
    orphan = [relative(p) for p in labels if relative(p) not in expected]
    for p in labels:
        manifest[relative(p)] = sha(p)
    # Exact file bytes and decoded pixels; not a perceptual near-duplicate test.
    test_byte, test_pixel = {}, {}
    for r in records:
        test_byte.setdefault(r['image_sha256'], []).append(r['filename'])
        if 'pixel_sha256' in r:
            test_pixel.setdefault(r['pixel_sha256'], []).append(r['filename'])
    duplicates, reference_counts, reference_errors = [], {}, []
    for source in ('custom', 'kaggle'):
        for split in ('train', 'val'):
            refs = images(ROOT / 'data' / source / 'images' / split)
            reference_counts[f'{source}/{split}'] = len(refs)
            print(f'AUDIT duplicate scan {source}/{split}: {len(refs)} images', flush=True)
            for p in refs:
                digest = sha(p)
                manifest[relative(p)] = digest
                byte_matches = test_byte.get(digest, [])
                try:
                    im = decode(p)
                    pixel_matches = test_pixel.get(pixel_sha(im), []) if im is not None else []
                    if im is None:
                        reference_errors.append(relative(p))
                except Exception:
                    pixel_matches = []
                    reference_errors.append(relative(p))
                for name in sorted(set(byte_matches + pixel_matches)):
                    duplicates.append({'filename': name, 'reference': relative(p), 'source_split': f'{source}/{split}',
                                       'same_file_bytes': name in byte_matches, 'same_decoded_pixels': name in pixel_matches})
    protected = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'be').rglob('*.py'))
    protected += [ROOT / '.env.example', ROOT / 'requirements.txt', ROOT / 'Dockerfile', ROOT / 'docker-compose.yml', args.weights]
    protected += list((ROOT / 'data').rglob('*.yaml'))
    if (ROOT / '.env').exists():
        protected.append(ROOT / '.env')
    for p in protected:
        manifest[relative(p)] = sha(p)
    # Search structured mapping candidates, excluding generated output and dependencies.
    mapping_candidates = []
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.csv', '.tsv', '.json', '.jsonl', '.xlsx', '.xml'}:
            continue
        rel = p.relative_to(ROOT)
        if any(part in {'.venv', '.git', 'node_modules', '__pycache__', '.kilo'} for part in rel.parts):
            continue
        if p.is_relative_to(out):
            continue
        mapping_candidates.append(relative(p))
    dataset = {'images_root': str(folder), 'labels_root': str(label_dir), 'image_count': len(records),
               'label_file_count': len(labels), 'gt_box_count_valid_rows': sum(len(r['gt']) for r in records),
               'scorable_images': sum(r['annotation_scorable'] for r in records),
               'corrupt_images': [r['filename'] for r in records if r.get('decode_error')],
               'missing_labels': [r['filename'] for r in records if r['annotation_status'].startswith('missing')],
               'explicit_empty_labels': [r['filename'] for r in records if r['annotation_status'].startswith('explicit_empty')],
               'invalid_labels': [r['filename'] for r in records if r['annotation_status'] in {'invalid', 'unreadable_label'}],
               'orphan_labels': orphan, 'records': records, 'reference_image_counts': reference_counts,
               'reference_decode_errors': reference_errors, 'cross_split_exact_duplicates': duplicates,
               'within_test_same_bytes': [v for v in test_byte.values() if len(v) > 1],
               'within_test_same_pixels': [v for v in test_pixel.values() if len(v) > 1],
               'duplicate_limit': 'Exact encoded-file and decoded-BGR-pixel hashes only; near-duplicates/re-encodings/crops not excluded.',
               'ocr_mapping_candidates': mapping_candidates,
               'missing_label_policy': 'Unknown annotation, excluded from detector scoring; never assumed background.',
               'empty_label_policy': 'Explicit empty YOLO file treated as background by convention; semantic correctness not manually certified.'}
    write_json(out / 'dataset_audit.json', dataset)
    write_json(out / 'protected_manifest_before.json', manifest)
    template = out / 'ground_truth_template.csv'
    if not template.exists():
        write_csv(template, [{'filename': r['filename'], 'plate_text': ''} for r in records], ['filename', 'plate_text'])
    box_template = out/'ground_truth_boxes_template.csv'
    if not box_template.exists():
        write_csv(box_template,[{'filename':r['filename'],'gt_index':i,'plate_text':''} for r in records for i,_ in enumerate(r['gt'])],['filename','gt_index','plate_text'])
    print(f'AUDIT DONE images={len(records)} labels={len(labels)} boxes={dataset["gt_box_count_valid_rows"]}', flush=True)


def environment(args):
    from src.utils import config
    import torch
    import cv2
    env = {'timestamp_Asia_Bangkok': datetime.now(timezone(timedelta(hours=7))).isoformat(),
           'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'cwd': str(Path.cwd()), 'weights': str(args.weights), 'checkpoint_sha256': sha(args.weights),
           'packages': {p: importlib.metadata.version(p) for p in ['ultralytics', 'torch', 'torchvision', 'easyocr', 'opencv-python', 'numpy', 'fastapi']},
           'cuda_available': torch.cuda.is_available(), 'torch_threads': torch.get_num_threads(),
           'opencv_threads': cv2.getNumThreads(), 'runtime_config': {k: getattr(config, k) for k in config.__all__},
           'runtime_model_path_resolved': str(Path(config.MODEL_PATH).resolve()),
           'diagnostic_prediction_parameters': {'conf': 0.001, 'iou': 0.7, 'imgsz': 640, 'device': 'cpu',
               'classes': None, 'agnostic_nms': False, 'max_det': 300, 'quantize': None, 'augment': False,
               'verbose': False, 'rect': True, 'batch': 1, 'save': False, 'stream': False},
           'ocr_download_enabled': False, 'production_source_modified': False}
    if Path(config.MODEL_PATH).resolve() != args.weights:
        raise RuntimeError('Runtime MODEL_PATH differs from requested weights; refusing silent config changes')
    write_json(args.output / 'environment.json', env)
    return env


def detect_all(args):
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.metrics import compute_ap
    env = environment(args)
    dataset = read_json(args.output / 'dataset_audit.json')
    model = YOLO(str(args.weights))
    env['model_names'] = model.names
    env['checkpoint_version'] = model.ckpt.get('version')
    env['checkpoint_train_args'] = model.ckpt.get('train_args')
    env['checkpoint_train_metrics'] = model.ckpt.get('train_metrics')
    if model.names != {0: 'license_plate'}:
        raise RuntimeError(f'Unexpected checkpoint classes: {model.names}')
    records, rows = [], []
    kwargs = env['diagnostic_prediction_parameters'].copy()
    for n, item in enumerate(dataset['records'], 1):
        r = dict(item, predictions=[])
        if not r.get('decode_error'):
            result = model.predict(decode(args.images / r['filename']), **kwargs)[0]
            for box in result.boxes:
                r['predictions'].append({'class_id': int(box.cls[0]), 'confidence': float(box.conf[0]), 'xyxy': box.xyxy[0].tolist()})
            if len(r['predictions']) == kwargs['max_det']:
                r['max_det_reached'] = True
            env['diagnostic_effective_predictor_args'] = vars(model.predictor.args)
            env['diagnostic_actual_device'] = str(model.predictor.device)
        records.append(r)
        matches = {m['prediction_index']: m for m in match(r['predictions'], r['gt'])} if r['annotation_scorable'] else {}
        used = {m['gt_index'] for m in matches.values()}
        for i, p in enumerate(r['predictions']):
            best = max(((overlap(p['xyxy'], g['xyxy'])[0], j) for j, g in enumerate(r['gt'])), default=(None, None))
            m = matches.get(i)
            rows.append({'filename': r['filename'], 'width': r.get('width'), 'height': r.get('height'),
                         'gt_count': len(r['gt']), 'prediction_count_low_threshold': len(r['predictions']),
                         'row_type': 'prediction', 'prediction_index': i, **p, 'best_iou': best[0],
                         'best_gt_index': best[1], 'matched_gt_index': m['gt_index'] if m else None,
                         'match_iou': m['iou'] if m else None, 'status_iou50_at_conf0001': ('TP' if m else 'FP') if r['annotation_scorable'] else 'UNSCORED'})
        for j, g in enumerate(r['gt']):
            if r['annotation_scorable'] and j not in used:
                rows.append({'filename': r['filename'], 'width': r['width'], 'height': r['height'], 'gt_count': len(r['gt']),
                             'prediction_count_low_threshold': len(r['predictions']), 'row_type': 'unmatched_ground_truth',
                             'class_id': g['class_id'], 'xyxy': g['xyxy'], 'matched_gt_index': j, 'status_iou50_at_conf0001': 'FN'})
        if not r['predictions'] and not r['gt']:
            rows.append({'filename': r['filename'], 'row_type': 'image_no_predictions_no_known_gt', 'gt_count': 0,
                         'prediction_count_low_threshold': 0, 'status_iou50_at_conf0001': 'BACKGROUND' if r['annotation_scorable'] else 'UNSCORED'})
        print(f'DETECT {n}/{len(dataset["records"])} {r["filename"]}: {len(r["predictions"])} boxes', flush=True)
    write_json(args.output / 'predictions_low_threshold.json', records)
    write_csv(args.output / 'detection_results.csv', rows)
    threshold_metrics = [evaluate(records, t) for t in THRESHOLDS]
    write_csv(args.output / 'threshold_metrics.csv', threshold_metrics)
    total_gt = sum(len(r['gt']) for r in records if r['annotation_scorable'])
    aps = []
    curves = []
    for iou in np.linspace(.5, .95, 10):
        flags = []
        for r in records:
            if not r['annotation_scorable']:
                continue
            matched = {m['prediction_index'] for m in match(r['predictions'], r['gt'], float(iou))}
            flags += [(p['confidence'], int(i in matched)) for i, p in enumerate(r['predictions'])]
        flags.sort(key=lambda x: -x[0])
        tp = np.cumsum([v for _, v in flags])
        fp = np.cumsum([1 - v for _, v in flags])
        recall = tp / total_gt if total_gt else np.zeros_like(tp, dtype=float)
        precision = tp / np.maximum(tp + fp, 1)
        ap = float(compute_ap(recall, precision)[0]) if len(flags) and total_gt else 0.
        aps.append({'iou': round(float(iou), 2), 'ap': ap})
        if iou == .5:
            curves = [{'confidence': c, 'precision': float(p), 'recall': float(r)} for (c, _), p, r in zip(flags, precision, recall)]
    confs = [p['confidence'] for r in records for p in r['predictions']]
    metrics = {'scorable_images': sum(r['annotation_scorable'] for r in records), 'gt_count': total_gt,
               'prediction_count_low_threshold': len(confs), 'ap50': aps[0]['ap'], 'map50_95': sum(x['ap'] for x in aps) / 10,
               'ap_by_iou': aps, 'thresholds': threshold_metrics,
               'confidence_quantiles_all_low_threshold_predictions': dict(zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(confs, [0, .25, .5, .75, 1]).tolist())) if confs else {},
               'matching': 'class-aware confidence-descending greedy one-to-one, highest IoU unmatched GT, IoU >= evaluation threshold',
               'ap_method': 'Ultralytics compute_ap 101-point interpolated trapezoidal PR integration; custom matching above, not official COCOeval/model.val',
               'nms_policy': 'Single prediction set at conf=0.001 after class-aware NMS iou=0.7 max_det=300; filter same set with confidence > threshold; never rerun NMS per threshold.',
               'max_det_reached_images': [r['filename'] for r in records if r.get('max_det_reached')]}
    # AP on exact non-overlapping test subset, without hiding contaminated full-set results.
    contaminated = {d['filename'] for d in dataset['cross_split_exact_duplicates']}
    clean = [r for r in records if r['filename'] not in contaminated]
    metrics['exact_overlap_excluded_subset'] = {'image_count': len(clean), 'metrics_conf025': evaluate(clean, .25)}
    write_json(args.output / 'detector_metrics.json', metrics)
    write_csv(args.output / 'precision_recall_curve.csv', curves, ['confidence', 'precision', 'recall'])
    write_json(args.output / 'environment.json', env)
    print('DETECT DONE', json.dumps({k: metrics[k] for k in ['ap50', 'map50_95', 'gt_count']}), flush=True)


class Recorder:
    """Observe real service/PlateOCR calls; never reimplement the baseline."""
    def __init__(self, args, detector, ocr):
        import be.services.detection_service as service_module
        import src.inference.preprocessing as pre
        import src.inference.postprocessing as post
        self.args, self.detector, self.ocr = args, detector, ocr
        self.sm, self.pre, self.post = service_module, pre, post
        self.current, self.trace = None, None
        self.real_crop, self.real_preprocess = pre.crop_plate, pre.preprocess_plate
        self.real_read = ocr.read_text
        self.real_format, self.real_split = post.format_vn_plate, post.split_two_lines
        self.real_deskew, self.real_clahe = pre.deskew_plate, pre.apply_clahe
        self.real_post_clahe = post.apply_clahe
        self.real_detect = detector.detect
        self.real_reader = ocr.reader

    def image(self, kind, im, suffix=''):
        import cv2
        if self.trace is None:
            return
        p = self.args.output / 'visualizations' / kind / f'{self.current["id"]}-{self.trace["index"]}{suffix}.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode('.png', im)
        if not ok:
            raise ValueError(f'Cannot encode artifact {p}')
        p.write_bytes(encoded.tobytes())
        self.trace.setdefault('artifacts', {})[kind + suffix] = p.relative_to(self.args.output).as_posix()
        self.checkpoint()

    def checkpoint(self):
        if self.current is not None:
            write_json(self.args.output/'partial_traces'/f'{self.current["id"]}.json',self.current)

    def detect(self, im):
        ds = self.real_detect(im)
        self.current['runtime_detections'] = [dict(xyxy=d.xyxy, confidence=d.confidence, class_id=d.class_id) for d in ds]
        return ds

    def crop(self, im, xyxy, padding=0.):
        self.trace = {'index': len(self.current['traces']), 'xyxy': list(xyxy), 'padding': padding, 'raw_ocr_calls': [], 'lines_before_cleaning': []}
        self.current['traces'].append(self.trace)
        h, w = im.shape[:2]
        x1, y1, x2, y2 = xyxy
        px, py = (x2-x1)*padding, (y2-y1)*padding
        self.trace['padded_xyxy'] = [max(0, round(x1-px)), max(0, round(y1-py)), min(w, round(x2+px)), min(h, round(y2+py))]
        crop = self.real_crop(im, xyxy, padding)
        self.trace['crop_shape'] = list(crop.shape)
        if crop.size:
            self.image('crops', crop)
        return crop

    def preprocess(self, im, **kwargs):
        out = self.real_preprocess(im, **kwargs)
        self.trace['processed_shape'] = list(out.shape)
        self.trace['processed_dtype'] = str(out.dtype)
        self.image('preprocessed', out)
        return out

    def deskew(self, im):
        out = self.real_deskew(im)
        if self.trace is not None:
            self.trace['deskew_applied'] = out is not im
            self.image('stages', out, '-deskew')
        return out

    def clahe(self, im):
        out = self.real_clahe(im)
        self.image('stages', out, '-clahe')
        return out

    def split(self, im):
        halves = self.real_split(im)
        self.trace['split_used'] = True
        self.trace['split_shapes'] = [list(x.shape) for x in halves]
        for label, half in zip(('top', 'bottom'), halves):
            if half.size:
                self.image('splits', half, '-' + label)
        return halves

    def format(self, lines):
        self.trace['lines_before_cleaning'] = list(lines)
        self.trace['cleaned_lines'] = [self.post._clean(s) for s in lines]
        result = self.real_format(lines)
        self.trace['after_heuristic_regex'] = result
        return result

    def readtext(self, im, **kwargs):
        out = self.real_reader.readtext(im, **kwargs)
        self.trace['raw_ocr_calls'].append({'input_shape': list(im.shape), 'tuples': out, 'kwargs': kwargs})
        # Save the actual OCR input: split branch applies CLAHE again.
        self.image('ocr_inputs', im, f'-call{len(self.trace["raw_ocr_calls"])}')
        return out

    def read(self, im):
        result = self.real_read(im)
        self.trace['final_plate_number'] = result
        self.trace['raw_text_normalized'] = normalized(''.join(self.trace.get('lines_before_cleaning', [])))
        self.trace['final_normalized'] = normalized(result)
        self.trace['postprocessing_changed_alphanumeric'] = self.trace['raw_text_normalized'] != normalized(result)
        return result

    @contextlib.contextmanager
    def instrument(self):
        with contextlib.ExitStack() as stack:
            for obj, name, replacement in [(self.detector, 'detect', self.detect),
                    (self.sm, 'crop_plate', self.crop), (self.sm, 'preprocess_plate', self.preprocess),
                    (self.pre, 'deskew_plate', self.deskew), (self.pre, 'apply_clahe', self.clahe),
                    (self.post, 'split_two_lines', self.split), (self.post, 'format_vn_plate', self.format),
                    (self.real_reader, 'readtext', self.readtext), (self.ocr, 'read_text', self.read)]:
                # readtext needs the unpatched bound method to avoid recursion.
                if obj is self.real_reader:
                    original_readtext = self.real_reader.readtext
                    def readtext(im, _original=original_readtext, **kwargs):
                        self.trace['active_ocr_input_shape'] = list(im.shape)
                        self.trace['active_ocr_call_status'] = 'started'
                        self.image('ocr_inputs', im, f'-call{len(self.trace["raw_ocr_calls"])+1}')
                        out = _original(im, **kwargs)
                        self.trace['raw_ocr_calls'].append({'input_shape': list(im.shape), 'tuples': out, 'kwargs': kwargs})
                        self.trace['active_ocr_call_status'] = 'completed'
                        self.image('ocr_inputs', im, f'-call{len(self.trace["raw_ocr_calls"])}')
                        return out
                    replacement = readtext
                stack.enter_context(patch.object(obj, name, replacement))
            yield


def save_overlay(args, record, preds, name):
    import cv2
    im = decode(args.images / record['filename'])
    if im is None:
        return None
    for g in record['gt']:
        x1,y1,x2,y2 = map(round, g['xyxy'])
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,220,0), 2)
        cv2.putText(im, 'GT', (x1,max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,220,0), 1)
    for i,p in enumerate(preds):
        x1,y1,x2,y2 = map(round, p['xyxy'])
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(im, f'P{i} {p["confidence"]:.3f}', (x1,max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1)
    path = args.output / 'visualizations' / 'detections' / (name + '.jpg')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode('.jpg', im)[1].tobytes())
    return path.relative_to(args.output).as_posix()


def pipeline(args):
    if args.isolate_images:
        return isolated_pipeline(args)
    import easyocr
    from src.inference.predictor import PlateDetector
    from src.inference.postprocessing import PlateOCR
    from be.services.detection_service import DetectionService
    ds = read_json(args.output / 'dataset_audit.json')
    detector, ocr = PlateDetector(), PlateOCR()
    # No download or engine switch. Cached Reader uses exactly runtime language/GPU defaults.
    ocr._reader = easyocr.Reader(ocr.languages, gpu=ocr.use_gpu, download_enabled=False)
    service = DetectionService(detector=detector, ocr=ocr)
    recorder = Recorder(args, detector, ocr)
    dest = args.output / 'pipeline_results.jsonl'
    done = {}
    if dest.exists():
        for line in dest.read_text(encoding='utf-8').splitlines():
            if line.strip():
                obj = json.loads(line)
                done[obj['filename']] = obj
    with recorder.instrument(), dest.open('a', encoding='utf-8') as f:
        for n, item in enumerate(ds['records'], 1):
            if args.only and item['filename'] != args.only:
                continue
            if item['filename'] in done:
                continue
            now = time.perf_counter()
            current = {'filename': item['filename'], 'id': f'{n:03d}', 'traces': []}
            recorder.current, recorder.trace = current, None
            current['response'] = service.detect_bytes((args.images / item['filename']).read_bytes())
            current['operational_status'] = 'completed'
            current['elapsed_seconds'] = time.perf_counter() - now
            current['overlay'] = save_overlay(args, item, current.get('runtime_detections', []), current['id'])
            f.write(json.dumps(current, ensure_ascii=False, default=serial) + '\n')
            f.flush()
            print(f'PIPELINE {n}/{len(ds["records"])} {item["filename"]} {current["elapsed_seconds"]:.1f}s {current["response"]}', flush=True)
    env = read_json(args.output / 'environment.json')
    env['runtime_effective_predictor_args'] = vars(detector.model.predictor.args) if detector.model.predictor else None
    env['runtime_actual_device'] = str(detector.model.predictor.device) if detector.model.predictor else None
    env['ocr_reader_device'] = str(ocr.reader.device)
    import torch
    env['runtime_torch_threads_after_inference'] = torch.get_num_threads()
    env['ocr_reader_details'] = {'languages': ocr.languages, 'allowlist': ocr.allowlist, 'download_enabled': False,
                               'detect_network': 'craft', 'detail': 1, 'paragraph': False, 'decoder': 'greedy'}
    write_json(args.output / 'environment.json', env)


def isolated_pipeline(args):
    """Bound native allocator lifetime without changing any inference setting."""
    import subprocess
    ds = read_json(args.output/'dataset_audit.json')
    dest = args.output/'pipeline_results.jsonl'
    done = {r['filename'] for r in load_pipeline(args)} if dest.exists() else set()
    attempts = []
    log = args.output/'isolated_workers.log'
    with log.open('a',encoding='utf-8') as f:
        for i,r in enumerate(ds['records'],1):
            name = r['filename']
            if name in done:
                continue
            cmd = [sys.executable,'-X','faulthandler','-B',str(Path(__file__).resolve()),'--phase','pipeline',
                   '--images',str(args.images),'--labels',str(args.labels),'--weights',str(args.weights),
                   '--output',str(args.output),'--only',name]
            f.write('\nWORKER '+name+'\n'); f.flush()
            started = time.perf_counter()
            child = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            timed_out = False
            try:
                child.wait(timeout=args.worker_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT)
                child.wait()
            attempts.append({'filename':name,'exit_code':child.returncode,'timed_out':timed_out,'timeout_seconds':args.worker_timeout,'seconds':time.perf_counter()-started})
            write_json(args.output/'isolated_worker_status.json',attempts)
            print(f'ISOLATED PIPELINE {i}/{len(ds["records"])} exit={child.returncode} {name}',flush=True)
            if child.returncode:
                partial = args.output/'partial_traces'/f'{i:03d}.json'
                current = read_json(partial) if partial.exists() else {'filename':name,'id':f'{i:03d}','traces':[]}
                current.update(response=None,operational_status='worker_failed',worker_exit_code=child.returncode,
                               worker_timed_out=timed_out,
                               elapsed_seconds=time.perf_counter()-started,
                               operational_error='Worker exited before completing service response. No fallback/resized prediction used. Root cause not established.')
                current['overlay'] = save_overlay(args,r,current.get('runtime_detections',[]),current['id'])
                with dest.open('a',encoding='utf-8') as failed_file:
                    failed_file.write(json.dumps(current,ensure_ascii=False,default=serial)+'\n')
                print(f'WORKER FAILURE recorded without substituting output: {name}',flush=True)
    print('ISOLATED PIPELINE DONE',flush=True)


def load_pipeline(args):
    return [json.loads(line) for line in (args.output / 'pipeline_results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def classify(args):
    ds = read_json(args.output / 'dataset_audit.json')
    direct = {r['filename']: r for r in read_json(args.output / 'predictions_low_threshold.json')}
    pipe = {r['filename']: r for r in load_pipeline(args)}
    rows = []
    for item in ds['records']:
        name, gt = item['filename'], item['gt']
        low, run = direct[name]['predictions'], pipe[name].get('runtime_detections', [])
        low_m, run_m = match(low, gt), match(run, gt)
        rescued = {m['gt_index'] for m in low_m} - {m['gt_index'] for m in run_m}
        supported_low = [j for j in rescued if any(p['confidence'] < .25 and overlap(p['xyxy'], gt[j]['xyxy'])[0] >= .5 for p in low)]
        bests = [max((overlap(p['xyxy'], g['xyxy'])[0] for g in gt), default=0.) for p in run]
        low_iou = [i for i,x in enumerate(bests) if .1 <= x < .5]
        nonoverlap_fp = [i for i,x in enumerate(bests) if x < .1]
        partial_boxes, partial_crops = [], []
        for i,p in enumerate(run):
            if gt:
                j = max(range(len(gt)), key=lambda j: overlap(p['xyxy'], gt[j]['xyxy'])[0])
                iou, coverage = overlap(p['xyxy'], gt[j]['xyxy'])
                if iou >= .1 and coverage < .95:
                    partial_boxes.append({'prediction_index': i, 'gt_index': j, 'gt_box_coverage': coverage})
        for t in pipe[name]['traces']:
            if gt:
                j = max(range(len(gt)), key=lambda j: overlap(t['xyxy'], gt[j]['xyxy'])[0])
                iou = overlap(t['xyxy'], gt[j]['xyxy'])[0]
                cover = overlap(t['padded_xyxy'], gt[j]['xyxy'])[1]
                if iou >= .1 and cover < .95:
                    partial_crops.append({'trace_index': t['index'], 'gt_index': j, 'gt_box_coverage': cover})
        traces = pipe[name]['traces']
        detector_ran = 'runtime_detections' in pipe[name]
        row = {'filename': name, 'runtime_prediction_count': len(run), 'low_threshold_prediction_count': len(low),
               'operational_status': pipe[name].get('operational_status','completed'),
               'runtime_tp': len(run_m), 'runtime_fp': len(run)-len(run_m), 'runtime_fn': len(gt)-len(run_m),
               'A_confirmed': not bool(run) if detector_ran else None, 'B_confirmed': bool(supported_low) and item['annotation_scorable'] if detector_ran else None,
               'C_confirmed': None,
               'C_suspected': bool(nonoverlap_fp) and item['annotation_scorable'],
               'D_confirmed': bool(low_iou or partial_boxes) and item['annotation_scorable'],
               'E_confirmed': (bool(partial_crops) and item['annotation_scorable']) or any(not all(t['crop_shape']) for t in traces),
               'F_confirmed': None, 'G_confirmed': None, 'H_confirmed': None, 'I_confirmed': None, 'J_confirmed': None,
               'F_suspected': any(t.get('deskew_applied') for t in traces),
               'G_suspected': any(t.get('split_used') and len([s for s in t.get('lines_before_cleaning', []) if s]) != 2 for t in traces),
               'H_suspected': any(t.get('final_plate_number') == '' for t in traces),
               'I_suspected': any(t.get('postprocessing_changed_alphanumeric') for t in traces),
               'low_confidence_rescued_gt_indices': supported_low, 'low_iou_prediction_indices': low_iou,
               'nonoverlapping_fp_indices': nonoverlap_fp, 'partial_predicted_boxes': partial_boxes,
               'partial_padded_crops': partial_crops, 'best_ious_runtime': bests,
               'annotation_scorable': item['annotation_scorable']}
        rows.append(row)
    rules = {'A': 'No predictions in the actual runtime detector at default conf=0.25; low-threshold count also retained.',
             'B': 'At least one annotated GT missed by runtime but one-to-one matched at IoU>=0.5 in the shared low-conf set with conf<0.25.',
             'C': 'Potential wrong-object review candidate: runtime max IoU<0.1 against all GT. This is a localization FP under annotation completeness, but object identity is not certified; C confirmed remains null without manual review.',
             'D': 'Runtime prediction max IoU in [0.1,0.5), or best-overlap GT bbox coverage<0.95. Geometric mismatch, not proof of cut characters.',
             'E': 'Padded crop covers <95% of a best-overlap GT bbox (prediction IoU>=0.1), or crop empty. GT bbox coverage does not itself prove lost glyphs.',
             'F': 'No confirmed preprocessing harm without text GT/manual verification. Deskew applied is only a review candidate.',
             'G': 'No confirmed wrong split without layout/text GT. Split branch yielding !=2 nonempty lines is only suspected.',
             'H': 'No confirmed raw OCR error without text GT; empty final text is a review candidate only.',
             'I': 'No confirmed harmful postprocessing without text GT. Changed alphanumeric content is a review candidate only.',
             'J': 'Cannot certify correct ALPR without text GT, even for detection TP and plausible formatted output.'}
    adjudication_path = args.output/'visual_adjudications.json'
    if adjudication_path.exists():
        for review in read_json(adjudication_path)['records']:
            if review['verdict']=='confirmed' and review['group'] in 'ABCDEFGHIJ':
                row = next(r for r in rows if r['filename']==review['filename'])
                row[review['group']+'_confirmed'] = True
                row[review['group']+'_suspected'] = False
                row.setdefault('manual_evidence',[]).append(review)
    write_json(args.output / 'error_classification.json', {'rules': rules, 'records': rows,
               'note': 'Multi-label categories; rates do not sum to 100%. Null means not evaluated, not zero errors.'})
    write_csv(args.output / 'error_classification.csv', rows)
    return rows


def cases(args):
    if args.isolate_cases:
        return isolated_cases(args)
    import cv2
    import easyocr
    from src.inference import preprocessing as pre
    from src.inference.postprocessing import PlateOCR
    from src.inference.predictor import PlateDetector
    ds = {r['filename']: r for r in read_json(args.output / 'dataset_audit.json')['records']}
    pp = {r['filename']: r for r in load_pipeline(args)}
    errors = classify(args)
    selected = []
    failed = [r['filename'] for r in errors if r.get('operational_status')=='worker_failed']
    if failed:
        selected.append(failed[0])
    # Include different phenomena first, then fill by severity (not by attractive OCR output).
    for key in ['A_confirmed', 'B_confirmed', 'C_suspected', 'D_confirmed', 'E_confirmed', 'G_suspected', 'H_suspected', 'I_suspected', 'F_suspected']:
        candidates = [r for r in errors if r.get(key) and r['filename'] not in selected]
        if candidates:
            selected.append(candidates[0]['filename'])
    for r in sorted(errors, key=lambda r: -sum(bool(r.get(k)) for k in ['A_confirmed','B_confirmed','C_confirmed','D_confirmed','E_confirmed'])):
        if len(selected) >= 10:
            break
        if r['filename'] not in selected:
            selected.append(r['filename'])
    ablation_names = [name for name in selected if pp[name].get('operational_status','completed')=='completed'
                      and pp[name]['traces'] and not next(r for r in errors if r['filename']==name)['E_confirmed']
                      and next(r for r in errors if r['filename']==name)['runtime_tp']>0][:3]
    ocr = PlateOCR()
    ocr._reader = easyocr.Reader(ocr.languages, gpu=ocr.use_gpu, download_enabled=False)
    import torch
    parity_env = read_json(args.output/'environment.json')
    torch.set_num_threads(parity_env.get('runtime_torch_threads_after_inference',8))
    cv2.setNumThreads(parity_env.get('opencv_threads',1))
    recorder = Recorder(args, PlateDetector(), ocr)
    results = []
    target = args.output / 'error_cases' / 'case_comparisons.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if target.exists():
        done = {r['filename']: r for r in [json.loads(s) for s in target.read_text(encoding='utf-8').splitlines() if s.strip()]}
    with recorder.instrument(), target.open('a', encoding='utf-8') as f:
        for n, name in enumerate(selected):
            if args.case_only and name != args.case_only:
                continue
            if name in done:
                results.append(done[name])
                continue
            item = ds[name]
            im = decode(args.images / name)
            result = {'filename': name, 'selection': next(r for r in errors if r['filename']==name),
                      'baseline': pp[name], 'gt_crop_runs': [], 'ablations': [], 'interpretation': 'Output differences alone do not establish correctness without independent OCR ground truth.'}
            if pp[name].get('operational_status')=='worker_failed':
                # Retain the failure as a representative; do not silently substitute another OCR pipeline.
                for j,g in enumerate(item['gt']):
                    cur = {'filename':name,'gt_index':j,'id':f'case{n+1:02d}-gt{j}','traces':[]}
                    recorder.current,recorder.trace = cur,None
                    recorder.crop(im,g['xyxy'],padding=recorder.sm.CROP_PADDING)
                    result['gt_crop_runs'].append(cur)
                result['operational_note'] = 'Baseline native worker failure: GT crop exported, additional OCR not substituted for failed baseline.'
                results.append(result)
                f.write(json.dumps(result,ensure_ascii=False,default=serial)+'\n'); f.flush()
                print(f'CASE {n+1}/{len(selected)} native failure retained {name}',flush=True)
                continue
            for j,g in enumerate(item['gt']):
                cur = {'filename': name, 'gt_index':j, 'id': f'case{n+1:02d}-gt{j}', 'traces': []}
                recorder.current, recorder.trace = cur, None
                crop = recorder.crop(im, g['xyxy'], padding=recorder.sm.CROP_PADDING)
                if crop.size:
                    processed = recorder.preprocess(crop, upscale=recorder.sm.OCR_UPSCALE)
                    recorder.read(processed)
                result['gt_crop_runs'].append(cur)
            # Controlled one-factor comparisons on up to 3 cases: same detector crop.
            if name in ablation_names:
                t = pp[name]['traces'][0]
                crop = recorder.real_crop(im, t['xyxy'], padding=t['padding'])
                if crop.size:
                    variants = [('raw_crop_gray', None), ('no_deskew', {'use_deskew': False}),
                                ('no_clahe', {'use_clahe': False}), ('no_split', {})]
                    for label, kwargs in variants:
                        cur = {'filename': name, 'id': f'case{n+1:02d}-{label}', 'traces': []}
                        recorder.current, recorder.trace = cur, {'index': 0, 'raw_ocr_calls': [], 'artifacts': {}}
                        cur['traces'].append(recorder.trace)
                        processed = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if kwargs is None else recorder.real_preprocess(crop, upscale=recorder.sm.OCR_UPSCALE, **kwargs)
                        recorder.image('preprocessed', processed)
                        old_split = ocr.split_two_line
                        old_post_clahe = recorder.post.apply_clahe
                        try:
                            if label == 'no_split':
                                ocr.split_two_line = False
                            # Disable BOTH CLAHE calls in the dedicated no-CLAHE ablation.
                            if label == 'no_clahe':
                                recorder.post.apply_clahe = lambda x: x
                            recorder.read(processed)
                        finally:
                            ocr.split_two_line = old_split
                            recorder.post.apply_clahe = old_post_clahe
                        result['ablations'].append({'variant': label, 'trace': cur})
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False, default=serial)+'\n'); f.flush()
            print(f'CASE {n+1}/{len(selected)} {name}', flush=True)
    write_json(args.output / 'error_cases' / 'case_comparisons.json', results)


def isolated_cases(args):
    import subprocess
    # Let the cases phase choose the same deterministic list; a lightweight
    # --list-cases run avoids initializing OCR in the parent.
    errors = classify(args)
    selected = [r['filename'] for r in errors if r.get('operational_status')=='worker_failed'][:1]
    for key in ['A_confirmed','B_confirmed','C_suspected','D_confirmed','E_confirmed','G_suspected','H_suspected','I_suspected','F_suspected']:
        candidates = [r for r in errors if r.get(key) and r['filename'] not in selected]
        if candidates:
            selected.append(candidates[0]['filename'])
    for r in sorted(errors,key=lambda r:-sum(bool(r.get(k)) for k in ['A_confirmed','B_confirmed','C_confirmed','D_confirmed','E_confirmed'])):
        if len(selected)>=10:
            break
        if r['filename'] not in selected:
            selected.append(r['filename'])
    path = args.output/'error_cases'/'case_comparisons.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    done = {json.loads(s)['filename'] for s in path.read_text(encoding='utf-8').splitlines() if s.strip()} if path.exists() else set()
    base = {r['filename']:r for r in load_pipeline(args)}
    with (args.output/'case_workers.log').open('a',encoding='utf-8') as log:
        for name in selected:
            if name in done:
                continue
            cmd = [sys.executable,'-X','faulthandler','-B',str(Path(__file__).resolve()),'--phase','cases',
                   '--output',str(args.output),'--images',str(args.images),'--labels',str(args.labels),
                   '--weights',str(args.weights),'--case-only',name]
            log.write('\nCASE WORKER '+name+'\n'); log.flush()
            child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            try:
                child.wait(timeout=args.worker_timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
                child.wait()
            if child.returncode:
                item = {'filename':name,'selection':next(r for r in errors if r['filename']==name),'baseline':base[name],
                        'gt_crop_runs':[],'ablations':[],'case_worker_exit_code':child.returncode,
                        'interpretation':'Additional case experiment failed; partial stage images retained, no fallback output substituted.'}
                with path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(item,ensure_ascii=False,default=serial)+'\n')
            print(f'ISOLATED CASE exit={child.returncode} {name}',flush=True)
    write_json(args.output/'error_cases'/'case_comparisons.json',[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()])


def retry_failures(args):
    """Repeat original failures once, without replacing first-pass records."""
    import shutil
    import subprocess
    outcomes = []
    failed = [p for p in load_pipeline(args) if p.get('operational_status')=='worker_failed']
    logpath = args.output/'reliability_repeat_workers.log'
    with logpath.open('a',encoding='utf-8') as log:
        for p in failed:
            out = args.output/'reliability_repeats'/p['id']
            out.mkdir(parents=True,exist_ok=True)
            for filename in ('dataset_audit.json','environment.json'):
                shutil.copyfile(args.output/filename,out/filename)
            dest = out/'pipeline_results.jsonl'
            cached_lines = [s for s in dest.read_text(encoding='utf-8').splitlines() if s.strip()] if dest.exists() else []
            if not cached_lines:
                cmd = [sys.executable,'-X','faulthandler','-B',str(Path(__file__).resolve()),'--phase','pipeline',
                       '--images',str(args.images),'--labels',str(args.labels),'--weights',str(args.weights),
                       '--output',str(out),'--only',p['filename']]
                log.write('\nREPEAT '+p['filename']+'\n'); log.flush()
                child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                timed_out = False
                try:
                    child.wait(timeout=args.worker_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
                    child.wait()
                exit_code = child.returncode
            else:
                timed_out,exit_code = False,0
            result_lines = [s for s in dest.read_text(encoding='utf-8').splitlines() if s.strip()] if dest.exists() else []
            repeat = json.loads(result_lines[-1]) if result_lines else None
            outcomes.append({'filename':p['filename'],'first_pass_exit_code':p.get('worker_exit_code'),
                             'repeat_exit_code':exit_code,'repeat_timed_out':timed_out,
                             'repeat_result':repeat,'artifact_directory':out.relative_to(args.output).as_posix(),
                             'policy':'Same full service/config/weights; repeat kept separate, never replaces first-pass baseline.'})
            write_json(args.output/'reliability_repeats.json',outcomes)
            print(f'REPEAT exit={exit_code} {p["filename"]}',flush=True)


def edit_distance(a, b):
    prev = list(range(len(b)+1))
    for i,x in enumerate(a, 1):
        cur = [i]
        for j,y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x!=y)))
        prev = cur
    return prev[-1]


def ocr_tables(args):
    gt_map, gt_errors = {}, []
    if args.ocr_gt:
        with args.ocr_gt.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'filename', 'plate_text'} <= set(reader.fieldnames or []):
                raise ValueError('OCR ground truth requires filename,plate_text headers')
            for r in reader:
                name, text = r['filename'].strip(), r['plate_text'].strip()
                if name in gt_map:
                    raise ValueError(f'Duplicate OCR ground truth filename: {name}')
                if text:
                    gt_map[name] = text
    ds = {r['filename']: r for r in read_json(args.output / 'dataset_audit.json')['records']}
    rows, image_rows = [], []
    raw_correct = final_correct = raw_edits = final_edits = char_count = scored = e2e_correct = 0
    for p in load_pipeline(args):
        name = p['filename']
        ground = gt_map.get(name)
        is_scorable = ground is not None and len(ds[name]['gt']) == 1 and ds[name]['annotation_scorable']
        runtime_matches = match(p.get('runtime_detections', []), ds[name]['gt'])
        gt_text = normalized(ground) if ground else ''
        if ground and not is_scorable:
            gt_errors.append({'filename': name, 'reason': 'filename-only text mapping supported only for one-box, scorable images'})
        traces = p['traces']
        for t in traces:
            raw = ''.join(t.get('lines_before_cleaning', []))
            row = {'filename': name, 'crop_index': t['index'], 'xyxy': t['xyxy'], 'padded_xyxy': t['padded_xyxy'],
                   'raw_ocr_tuples': t['raw_ocr_calls'], 'raw_text': raw, 'cleaned_lines': t.get('cleaned_lines', []),
                   'ocr_completed': 'final_plate_number' in t,
                   'text_after_heuristic_regex': t.get('after_heuristic_regex'), 'final_plate_number': t.get('final_plate_number'),
                   'raw_normalized': normalized(raw), 'final_normalized': normalized(t.get('final_plate_number', '')),
                   'ground_truth_raw': ground, 'ground_truth_normalized': gt_text if ground else None,
                   'ocr_metrics_eligible': is_scorable, 'artifacts': t.get('artifacts', {})}
            rows.append(row)
        image_rows.append({'filename': name, 'crop_count': len(traces), 'response': p['response'],
                           'operational_status':p.get('operational_status','completed'), 'elapsed_seconds': p['elapsed_seconds']})
        if is_scorable:
            # Match text to the detection matched to GT, rather than best OCR string.
            eligible = []
            for m in runtime_matches:
                pred = p['runtime_detections'][m['prediction_index']]
                eligible += [t for t in traces if t['xyxy'] == pred['xyxy']]
            t = eligible[0] if eligible else {}
            raw = normalized(''.join(t.get('lines_before_cleaning', [])))
            final = normalized(t.get('final_plate_number', ''))
            scored += 1; char_count += len(gt_text)
            raw_correct += raw == gt_text; final_correct += final == gt_text
            raw_edits += edit_distance(raw, gt_text); final_edits += edit_distance(final, gt_text)
            e2e_correct += final == gt_text and p['response'] is not None and len(p['response']['results']) == 1 and len(runtime_matches) == 1
    unknown_gt_names = sorted(set(gt_map)-set(ds))
    if unknown_gt_names:
        raise ValueError(f'OCR ground truth contains unknown filenames: {unknown_gt_names}')
    metrics = {'ground_truth_source': str(args.ocr_gt) if args.ocr_gt else None, 'status': 'evaluated' if scored else 'not_evaluated_no_independent_text_ground_truth',
               'normalization': 'Uppercase then remove characters outside ASCII A-Z0-9, identically for GT/raw/final; no O/0 or B/8 replacement in evaluation normalization.',
               'scored_images': scored, 'ground_truth_format_errors': gt_errors,
               'raw_exact_match': raw_correct/scored if scored else None, 'final_exact_match': final_correct/scored if scored else None,
               'raw_cer': raw_edits/char_count if char_count else None, 'final_cer': final_edits/char_count if char_count else None,
               'end_to_end_exact_image_accuracy': e2e_correct/scored if scored else None,
               'denominator_policy': 'For one-box images with independent text GT: choose IoU>=0.5 matched runtime detection; unmatched GT -> empty prediction. E2E also requires exactly one response and one detection match. No best-text selection.',
               'operationally_failed_images': [p['filename'] for p in load_pipeline(args) if p.get('operational_status')=='worker_failed'],
               'incomplete_ocr_traces': sum(not r['ocr_completed'] for r in rows),
               'empty_final_crops': sum(r['final_plate_number'] == '' for r in rows),
               'alphanumeric_changed_crops': sum(r['ocr_completed'] and r['raw_normalized'] != r['final_normalized'] for r in rows),
               'ocr_crop_count': len(rows)}
    write_csv(args.output / 'ocr_results.csv', rows)
    write_csv(args.output / 'pipeline_image_results.csv', image_rows)
    write_json(args.output / 'ocr_metrics.json', metrics)
    return metrics


def report(args):
    print('REPORT: reading completed benchmark artifacts',flush=True)
    ds = read_json(args.output / 'dataset_audit.json')
    dm = read_json(args.output / 'detector_metrics.json')
    env = read_json(args.output / 'environment.json')
    errors = classify(args)
    om = ocr_tables(args)
    confidence_details = confidence_diagnostics(args)
    case_path = args.output / 'error_cases' / 'case_comparisons.json'
    cs = read_json(case_path) if case_path.exists() else []
    for case in cs:
        case['selection'] = next(r for r in errors if r['filename']==case['filename'])
    render_case_sheets(args,cs)
    print(f'REPORT: rendered {len(cs)} case sheets',flush=True)
    n = ds['image_count']
    lines = ['# Diagnostic benchmark ALPR — baseline tập custom/test', '',
             f'Ngày: {env["timestamp_Asia_Bangkok"]}. Tổng số ảnh: **{n}**.', '',
             f'Pipeline đã thử {len(load_pipeline(args))}/{n} ảnh; response hoàn tất: {len(load_pipeline(args))-len(om["operationally_failed_images"])}; operational failure: {len(om["operationally_failed_images"])}. Các ảnh worker lỗi không được coi là OCR rỗng hoặc ALPR đúng.', '',
             'Không sửa production source, dataset, annotation, checkpoint hoặc config. Không retrain. OCR prediction không được dùng làm ground truth.', '',
             '## A. Dataset audit', '',
             f'- Ảnh: {n}; file label: {ds["label_file_count"]}; bbox hợp lệ: {ds["gt_box_count_valid_rows"]}; ảnh đủ annotation để scoring: {ds["scorable_images"]}.',
             f'- Ảnh nhiều biển: {sum(len(r["gt"])>1 for r in ds["records"])}. Template `filename,plate_text` dùng cho mapping cấp ảnh; template bổ sung `ground_truth_boxes_template.csv` có gt_index cho từng biển (metric nhiều biển cần bổ sung khi có nhãn).',
             f'- Ảnh hỏng: {len(ds["corrupt_images"])}; thiếu label: {len(ds["missing_labels"])}; label sai định dạng: {len(ds["invalid_labels"])}; label rỗng: {len(ds["explicit_empty_labels"])}; orphan label: {len(ds["orphan_labels"])}.',
             '- Kiểm tra mỗi dòng đúng 5 trường, class ID 0, số hữu hạn, center/size normalized và corner trong [0,1]; chuyển pixel theo width/height ảnh decode thực tế.',
             '- Label thiếu là annotation chưa rõ, không tự coi ảnh âm. Label rỗng được coi background theo quy ước YOLO, chưa chứng nhận ngữ nghĩa thủ công.',
             f'- Exact overlap với train/val: {len(ds["cross_split_exact_duplicates"])} cặp; {len({d["filename"] for d in ds["cross_split_exact_duplicates"]})} ảnh test liên quan.',
             f'- Số ảnh tham chiếu: `{ds["reference_image_counts"]}`. So cả SHA256 file và decoded BGR pixels; không loại trừ near-duplicate/re-encode/crop.',
             f'- Checkpoint SHA256: `{env["checkpoint_sha256"]}`.',
             f'- Packages: `{env["packages"]}`.',
             '- Chi tiết theo ảnh và integrity manifest nằm trong `dataset_audit.json`, `protected_manifest_before.json` và `integrity_check.json`.', '',
             '## B. Detector evaluation', '',
             f'- AP@0.5: **{dm["ap50"]:.6f}**; mAP@0.5:0.95: **{dm["map50_95"]:.6f}**.',
             f'- GT: {dm["gt_count"]}; predictions tại conf=0.001 sau NMS: {dm["prediction_count_low_threshold"]}.',
             '- Matching class-aware một-một theo confidence giảm dần; mỗi prediction chọn GT chưa ghép có IoU cao nhất; TP khi IoU ≥ ngưỡng. Mọi prediction không ghép là FP; GT không ghép là FN.',
             '- AP dùng hàm `ultralytics.utils.metrics.compute_ap` (101-point interpolation/trapezoidal integration) với matching đã công bố; không ghi nhận như output chính thức của COCOeval hay `model.val()`.',
             '- Tất cả threshold được đánh giá lại từ CÙNG tập prediction tại conf=0.001, class-aware NMS IoU=0.7, max_det=300. Không dùng max confidence thay matching; không chạy lại NMS theo từng threshold.',
             f'- Ảnh chạm max_det: {len(dm["max_det_reached_images"])}. Prediction dưới 0.001 không quan sát được.',
             f'- Tham số explicit: `{env["diagnostic_prediction_parameters"]}`.',
             f'- Device thực tế: `{env.get("diagnostic_actual_device")}`. Runtime args/device lưu riêng trong `environment.json`.', '',
             '| Confidence | TP | FP | FN | Precision | Recall | Mean IoU TP |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for m in dm['thresholds']:
        miou = f'{m["matched_mean_iou"]:.4f}' if m['matched_mean_iou'] is not None else 'N/A'
        lines.append(f'| {m["confidence_threshold"]} | {m["tp"]} | {m["fp"]} | {m["fn"]} | {m["precision"]:.4f} | {m["recall"]:.4f} | {miou} |')
    lines += ['', f'Confidence quantiles (tất cả low-threshold candidates, gồm FP): `{dm["confidence_quantiles_all_low_threshold_predictions"]}`.',
              f'Phân bố riêng TP/FP và runtime: `{confidence_details}`. Xem `confidence_diagnostics.json`.',
              f'Đối chiếu subset không exact-overlap: `{dm["exact_overlap_excluded_subset"]}`.',
              'Đây là baseline trên test hiện có; không dùng kết quả sweep để tự đổi config hoặc tuyên bố threshold tối ưu.', '',
              '## C. OCR evaluation', '',
              f'- Ground truth source: `{om["ground_truth_source"]}`; trạng thái: `{om["status"]}`.',
              f'- Số crop/trace được ghi nhận: {om["ocr_crop_count"]}; trace OCR chưa hoàn tất: {om["incomplete_ocr_traces"]}; final rỗng đã hoàn tất: {om["empty_final_crops"]}; crop có nội dung alphanumeric bị post-processing thay đổi: {om["alphanumeric_changed_crops"]}.',
              f'- Worker pipeline lỗi native/chưa trả response: {len(om["operationally_failed_images"])} ảnh. Lưu riêng trong `isolated_worker_status.json`, `partial_traces/` và logs; response `null` không được thay thành kết quả OCR rỗng.',
              f'- Raw Exact Match: {om["raw_exact_match"]}; Final Exact Match: {om["final_exact_match"]}; raw CER: {om["raw_cer"]}; final CER: {om["final_cer"]}; E2E accuracy: {om["end_to_end_exact_image_accuracy"]}.',
              '- `null` nghĩa là chưa đo được, không phải accuracy bằng 0. Chưa có text GT độc lập thì không kết luận OCR đúng/sai chỉ từ chuỗi có format hợp lý.',
              '- Chuẩn hóa đánh giá: uppercase và chỉ giữ ASCII A-Z0-9 giống nhau cho GT/prediction; không sửa O/0, B/8 trong chuẩn hóa metric. CSV giữ raw và normalized.',
              '- `ground_truth_template.csv` có `filename,plate_text` để nhóm điền độc lập; script hỗ trợ `--ocr-gt PATH --phase report`.',
              '- Baseline chạy `DetectionService.detect_bytes` thật → `PlateDetector` thật → crop/preprocess/PlateOCR/formatter thật. Chỉ bọc quan sát trong RAM; cached EasyOCR weights, download disabled.',
              '- Raw EasyOCR tuples/text confidence, cleaned lines, heuristic output, final response và mapping ảnh trung gian có trong CSV/JSONL.', '',
              '## D. Error classification', '',
              'Một ảnh có thể thuộc nhiều nhóm. Các flag hình học so với annotation không tự chứng minh mất ký tự. Không ép phân loại F–J khi thiếu bằng chứng. Visual adjudication (nếu có) ở `visual_adjudications.json` chỉ xác nhận những gì ảnh trung gian chứng minh; tỷ lệ manual-confirmed là lower bound trên tập test, không phải prevalence đã review hết.', '',
              '| Nhóm | Confirmed ảnh | Tỷ lệ toàn bộ test | Review candidates (suspected) |',
              '|---|---:|---:|---:|']
    counts = {}
    for group in 'ABCDEFGHIJ':
        vals = [r[group+'_confirmed'] for r in errors]
        count = sum(v is True for v in vals) if any(v is not None for v in vals) else None
        suspected = sum(bool(r.get(group+'_suspected')) for r in errors)
        counts[group] = {'confirmed_images': count, 'confirmed_rate': count/n if count is not None and n else None, 'suspected_images': suspected}
        lines.append(f'| {group} | {count if count is not None else "N/A"} | {f"{count/n:.2%}" if count is not None else "N/A"} | {suspected if group in "CFGHI" else "—"} |')
    rules = read_json(args.output / 'error_classification.json')['rules']
    lines += ['', *[f'- **{g}:** {rule}' for g,rule in rules.items()], '',
              '## E. Representative failure cases', '',
              'Các trường hợp được chọn theo nhóm lỗi/flag để review, không dựa vào OCR prediction làm nhãn. So detector crop với GT crop có cùng padding và cùng pipeline OCR; thay đổi output không đủ chứng minh bên nào đúng.', '']
    for c in cs:
        name, base = c['filename'], c['baseline']
        flags = [k for k,v in c['selection'].items() if k.endswith(('_confirmed','_suspected')) and v is True]
        lines += [f'### {base["id"]}. {name}', '', f'- Flags: `{flags}`.',
                  '- Overlay: GT xanh, prediction đỏ.',
                  f'- [Overlay]({base["overlay"]}); baseline response: `{base["response"]}`.',
                  f'- IoU runtime: `{c["selection"]["best_ious_runtime"]}`; padded-crop coverage issues: `{c["selection"]["partial_padded_crops"]}`.']
        if c.get('comparison_sheet'):
            lines.append(f'![Crop comparison]({c["comparison_sheet"]})')
        for review in c['selection'].get('manual_evidence',[]):
            lines.append(f'- Visual-confirmed group {review["group"]}: {review["observation"]} Limit: {review["limit"]}')
        if c.get('operational_note'):
            lines.append('- Operational note: '+c['operational_note'])
        if c.get('case_worker_exit_code'):
            lines.append(f'- Additional case experiment failed with exit code `{c["case_worker_exit_code"]}`; intermediate files are retained in `partial_traces/`. No replacement result was substituted.')
        for t in base['traces']:
            lines.append(f'- Detector crop {t["index"]}: raw lines `{t.get("lines_before_cleaning")}` → clean `{t.get("cleaned_lines")}` → final `{t.get("final_plate_number")}`; artifacts `{t.get("artifacts")}`.')
        for g in c['gt_crop_runs']:
            for t in g['traces']:
                lines.append(f'- GT crop: raw lines `{t.get("lines_before_cleaning")}` → final `{t.get("final_plate_number")}`; artifacts `{t.get("artifacts")}`.')
        for a in c['ablations']:
            t = a['trace']['traces'][0]
            lines.append(f'- Ablation `{a["variant"]}`: raw `{t.get("lines_before_cleaning")}` → final `{t.get("final_plate_number")}`.')
        lines += ['', 'Không xác nhận raw/final OCR đúng nếu chưa có text GT độc lập.', '']
    lines += ['## F. Technical findings', '', '**Confirmed:**', '',
              '- Detection errors được lượng hóa bằng matching/IoU/coverage trong các bảng; đây là sai lệch đo được, chưa tự giải thích nguyên nhân model tạo box đó.',
              '- Runtime có confidence filter bên trong YOLO và tại service (`OCR_MIN_CONFIDENCE`, thực tế là YOLO confidence). Response confidence là YOLO confidence.',
              '- OCR confidence không được application dùng để lọc; formatter có thể sửa/cắt ký tự. Số crop thay đổi alphanumeric được thống kê, chưa đồng nghĩa thay đổi có hại.',
              '- Nhánh split luôn áp dụng CLAHE lần nữa; baseline giữ nguyên hành vi này.', '', '**Suspected:**', '',
              '- Deskew/CLAHE/split có thể liên quan lỗi OCR: các flag chỉ là hàng đợi review, không phải nguyên nhân đã xác minh.',
              '- Padded crop không bao đủ annotation bbox có thể ảnh hưởng OCR; cần kiểm tra chữ thực sự bị cắt trên ảnh trung gian.', '', '**Requires additional evidence:**', '',
              '- Text GT độc lập để tính Exact Match/CER/E2E, xác nhận nhóm H/I/J và tác hại của preprocessing/split.',
              '- Ground-truth annotation cần review thủ công nếu detector predictions không khớp; bbox annotation chưa được coi là hoàn hảo tuyệt đối.',
              '- Hash không phát hiện near-duplicate; chưa thể chứng minh không có leakage hoặc test chưa từng được dùng fine-tune nếu thiếu manifest Kaggle thực tế.', '',
              '## G. Recommended next experiments', '',
              '1. Review annotation và exact-overlap trước khi diễn giải chất lượng generalization.',
              '2. Điền text GT độc lập trong template, đặc biệt các ca đại diện; không copy OCR output làm nhãn.',
              '3. So detector crop/GT crop với text GT; xác định localization hay OCR chi phối lỗi.',
              '4. Đánh giá riêng raw crop, deskew, CLAHE và split, thay từng yếu tố. Các ablation diagnostic không thay baseline/config.',
              '5. Chọn threshold bằng validation có nhãn và mục tiêu FP/FN; giữ test để báo cáo độc lập. Chưa đổi production/retrain/OCR engine.', '',
              '## Artefacts và tái lập', '',
              '- `environment.json`: package/checkpoint/config/effective predictor args.',
              '- `dataset_audit.json`: audit từng ảnh/label và exact duplicate pairs.',
              '- `detection_results.csv`, `predictions_low_threshold.json`, `threshold_metrics.csv`, `detector_metrics.json`.',
              '- `pipeline_results.jsonl`, `pipeline_image_results.csv`, `ocr_results.csv`, `ocr_metrics.json`.',
              '- `error_classification.csv/json`, `error_cases/case_comparisons.json/jsonl`.',
              '- `visualizations/`: detections, crops, preprocessed, splits, OCR inputs, deskew/CLAHE stages.',
              '- `protected_manifest_before.json`, `integrity_check.json`: kiểm chứng file được bảo vệ không đổi.', '',
              '```powershell', '.\\.venv\\Scripts\\python.exe -B scripts/diagnostic_benchmark.py --phase all --isolate-images --isolate-cases', '```', '',
              'Mỗi phase có thể chạy riêng (`audit`, `detect`, `pipeline`, `cases`, `report`); pipeline/cases có resume theo filename. Để chạy baseline mới với config khác, dùng output directory mới, không trộn records cũ.']
    lines += ['', '![Detector curves](charts/detector_curves.png)', '',
              '![Confidence distributions](charts/confidence_distribution.png)', '']
    (args.output / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(args.output / 'error_counts.json', counts)
    outcomes = [{'filename':p['filename'],'status':p.get('operational_status','completed'),
                 'exit_code':p.get('worker_exit_code',0),'timed_out':p.get('worker_timed_out',False),
                 'elapsed_seconds':p['elapsed_seconds'],'response':p['response']} for p in load_pipeline(args)]
    write_json(args.output/'operational_results.json',outcomes)
    write_csv(args.output/'operational_results.csv',outcomes)
    native_log = args.output/'reliability_repeat_workers.log'
    native_text = native_log.read_text(encoding='utf-8',errors='replace') if native_log.exists() else ''
    native_evidence = {'fatal_access_violation_trace_count':native_text.count('Windows fatal exception: access violation'),
                       'max_pool2d_in_trace':'_max_pool2d' in native_text,
                       'craft_in_trace':'easyocr\\craft.py' in native_text,
                       'source':'reliability_repeat_workers.log',
                       'conclusion':'Failure observed during EasyOCR CRAFT text detection / PyTorch max_pool2d; deeper native root cause remains unproven.'}
    write_json(args.output/'native_failure_evidence.json',native_evidence)
    # Append observed runtime faults separately from unproven OCR-quality causes.
    with (args.output/'summary.md').open('a',encoding='utf-8') as report_file:
        report_file.write('\n## Runtime reliability evidence\n\n')
        report_file.write('Có worker thoát native với mã `0xC0000005`. Windows Application events trong `windows_crash_events.json` ghi faulting modules `torch/lib/c10.dll` và `VCRUNTIME140.dll` cho các crash trong phiên audit. Đây là lỗi native đã quan sát, chưa chứng minh root cause cụ thể ở thư viện/backend nào. Không gộp native crash vào nhóm OCR trả text rỗng.\n\n')
        if native_evidence['max_pool2d_in_trace'] and native_evidence['craft_in_trace']:
            report_file.write('Faulthandler đã ghi stage crash: **EasyOCR CRAFT text detection → PyTorch `torch.nn.functional._max_pool2d`**, trước text recognition. Trace nằm trong `reliability_repeat_workers.log`; đây là vị trí quan sát được, không phải bằng chứng đủ để quy lỗi một wheel/DLL cụ thể. Native access violation có thể kết thúc process và không được Python `except Exception` trong service bắt.\n\n')
        probe_path = args.output/'native_probe_exit.json'
        if probe_path.exists():
            report_file.write('Probe OCR độc lập dùng chính PNG input đã lưu, cùng EasyOCR config: `'+json.dumps(read_json(probe_path),ensure_ascii=False)+'`. Xem `native_probe.jsonl` và `native_probe_stderr.log` để biết stage/traceback. Không dùng probe để thay response baseline.\n')
        repeats_path = args.output/'reliability_repeats.json'
        if repeats_path.exists():
            repeats = read_json(repeats_path)
            report_file.write('\nLặp lại full service trên các ảnh lỗi, không thay kết quả lượt đầu:\n\n')
            for r in repeats:
                response = r['repeat_result']['response'] if r['repeat_result'] else None
                report_file.write(f'- `{r["filename"]}`: first pass exit `{r["first_pass_exit_code"]}`, repeat exit `{r["repeat_exit_code"]}`, response `{response}`; artefacts `{r["artifact_directory"]}`.\n')
            report_file.write('\nNếu repeat thành công, crash là hiện tượng chưa tái lập ổn định; không kết luận lỗi chắc chắn do riêng ảnh, model, RAM hoặc một DLL chỉ từ faulting module.\n')
        report_file.write('\nƯu tiên thí nghiệm bổ sung: tái lập riêng CRAFT/max_pool2d từ OCR input đã lưu, ghi memory/thread/backend và thử môi trường runtime biệt lập trước khi thay production dependencies. Probe có thể thành công ở một lần nhưng full service vẫn crash; cần giữ toàn bộ lịch sử lượt thử.\n')
    if not (args.output/'ground_truth_boxes_template.csv').exists():
        write_csv(args.output/'ground_truth_boxes_template.csv',[{'filename':r['filename'],'gt_index':i,'plate_text':''} for r in ds['records'] for i,_ in enumerate(r['gt'])],['filename','gt_index','plate_text'])
    env['diagnostic_script_sha256'] = sha(Path(__file__))
    env['worker_safety_timeout_seconds'] = args.worker_timeout
    env['worker_policy'] = 'Native crash/timeout is recorded as operational failure with response=null, never as an empty successful OCR result. No fallback resizing or engine used.'
    write_json(args.output/'environment.json',env)
    reconcile(args)
    charts(args)
    print('REPORT: tables and charts complete; verifying protected file hashes',flush=True)
    integrity(args)
    print('REPORT DONE', flush=True)


def integrity(args):
    before = read_json(args.output / 'protected_manifest_before.json')
    changes = []
    for i,(path, digest) in enumerate(before.items(),1):
        p = ROOT / path
        if not p.is_file() or sha(p) != digest:
            changes.append(path)
        if i%1000==0:
            print(f'INTEGRITY {i}/{len(before)}',flush=True)
    write_json(args.output / 'integrity_check.json', {'checked_files': len(before), 'changed_or_missing': changes,
               'unchanged': not changes, 'method': 'Recompute SHA256 of production sources/configs/checkpoint and audited dataset files; diagnostic output excluded.'})
    if changes:
        raise RuntimeError(f'Protected files changed: {changes}')


def reconcile(args):
    """Verify the shared low-threshold set reproduces runtime predictions."""
    ds = {r['filename']: r for r in read_json(args.output/'dataset_audit.json')['records']}
    low = {r['filename']: r for r in read_json(args.output/'predictions_low_threshold.json')}
    rows, per_image, differences = [], [], []
    for run in load_pipeline(args):
        name = run['filename']
        preds, gt = run.get('runtime_detections', []), ds[name]['gt']
        expected = [p for p in low[name]['predictions'] if p['confidence'] > .25]
        equal = len(expected) == len(preds) and all(
            a['class_id'] == b['class_id'] and abs(a['confidence']-b['confidence']) < 1e-6 and
            max(abs(x-y) for x,y in zip(a['xyxy'],b['xyxy'])) < 1e-4 for a,b in zip(expected,preds))
        if not equal:
            differences.append(name)
        ms = {m['prediction_index']: m for m in match(preds,gt)}
        used = {m['gt_index'] for m in ms.values()}
        for i,p in enumerate(preds):
            m = ms.get(i)
            rows.append({'filename':name, 'row_type':'prediction', 'prediction_index':i, **p,
                         'matched_gt_index':m['gt_index'] if m else None, 'iou':m['iou'] if m else None,
                         'best_iou':max((overlap(p['xyxy'],g['xyxy'])[0] for g in gt), default=0.),
                         'status_iou50':'TP' if m else 'FP'})
        for j,g in enumerate(gt):
            if j not in used:
                rows.append({'filename':name, 'row_type':'unmatched_ground_truth', 'xyxy':g['xyxy'], 'matched_gt_index':j, 'status_iou50':'FN'})
        per_image.append({'filename':name,'width':ds[name].get('width'),'height':ds[name].get('height'),
                          'gt_count':len(gt),'runtime_prediction_count':len(preds),'tp':len(ms),
                          'fp':len(preds)-len(ms),'fn':len(gt)-len(ms),
                          'low_threshold_prediction_count':len(low[name]['predictions']),
                          'low_threshold_filtered_matches_runtime':equal})
    write_csv(args.output/'detection_runtime_results.csv', rows)
    write_csv(args.output/'detection_image_results.csv', per_image)
    write_json(args.output/'prediction_reconciliation.json', {'confidence_threshold':.25,
               'images_checked':len(per_image),'different_images':differences,'all_equal':not differences,
               'comparison_tolerances':{'confidence':1e-6,'xyxy_pixels':1e-4}})


def charts(args):
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'runtime_cache'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    dm = read_json(args.output/'detector_metrics.json')
    records = read_json(args.output/'predictions_low_threshold.json')
    folder = args.output/'charts'
    folder.mkdir(exist_ok=True)
    ts = dm['thresholds']
    fig,ax = plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.plot([r['confidence_threshold'] for r in ts],[r['precision'] for r in ts],marker='o',label='Precision @ IoU 0.5')
    ax.plot([r['confidence_threshold'] for r in ts],[r['recall'] for r in ts],marker='o',label='Recall @ IoU 0.5')
    ax.axvline(.25,linestyle='--',color='gray',label='Runtime conf 0.25')
    ax.set(xlabel='Confidence threshold (same post-NMS predictions)',ylabel='Score',ylim=(0,1.04),title='Custom test detector baseline (73 images / 103 GT boxes)')
    ax.grid(alpha=.2); ax.legend(); fig.savefig(folder/'detector_curves.png',dpi=160); plt.close(fig)
    tp,fp = [],[]
    for r in records:
        if not r['annotation_scorable']:
            continue
        matched = {m['prediction_index'] for m in match(r['predictions'],r['gt'])}
        for i,p in enumerate(r['predictions']):
            (tp if i in matched else fp).append(p['confidence'])
    fig,ax = plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.hist([fp,tp],bins=np.linspace(0,1,21),label=['FP','TP'],stacked=True,color=['#d27c69','#489b87'])
    ax.set_yscale('log'); ax.axvline(.25,linestyle='--',color='gray')
    ax.set(xlabel='YOLO confidence',ylabel='Prediction count (log scale)',title='Confidence by one-to-one IoU >= 0.5 match, conf >= 0.001')
    ax.legend(); fig.savefig(folder/'confidence_distribution.png',dpi=160); plt.close(fig)


def confidence_diagnostics(args):
    import numpy as np
    groups = {'matched_tp_iou50':[],'unmatched_fp_iou50':[],'all_predictions_at_runtime_conf025':[]}
    for r in read_json(args.output/'predictions_low_threshold.json'):
        if not r['annotation_scorable']:
            continue
        matched = {m['prediction_index'] for m in match(r['predictions'],r['gt'])}
        for i,p in enumerate(r['predictions']):
            groups['matched_tp_iou50' if i in matched else 'unmatched_fp_iou50'].append(p['confidence'])
            if p['confidence']>.25:
                groups['all_predictions_at_runtime_conf025'].append(p['confidence'])
    result = {k:{'count':len(v),'min_q25_median_q75_max':np.quantile(v,[0,.25,.5,.75,1]).tolist() if v else None} for k,v in groups.items()}
    write_json(args.output/'confidence_diagnostics.json',result)
    return result


def render_case_sheets(args,cases_data):
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'runtime_cache'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    for i,c in enumerate(cases_data,1):
        panels = []
        for title,traces in [('Detector',c['baseline']['traces']),('Ground truth', [t for g in c['gt_crop_runs'] for t in g['traces']])]:
            if traces:
                t = traces[0]
                for key,label in [('crops','crop'),('preprocessed','preprocessed'),('splits-top','top split'),('splits-bottom','bottom split')]:
                    path = t.get('artifacts',{}).get(key)
                    if path and (args.output/path).is_file():
                        panels.append((f'{title}: {label}',args.output/path))
        if not panels:
            continue
        cols = 2
        rows = (len(panels)+cols-1)//cols
        fig,axes = plt.subplots(rows,cols,figsize=(11,3*rows),squeeze=False,layout='constrained')
        for ax in axes.ravel():
            ax.axis('off')
        for ax,(title,path) in zip(axes.ravel(),panels):
            im = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB) if im.ndim==3 else im,cmap='gray',vmin=0,vmax=255)
            ax.set_title(title)
        fig.suptitle(f'Case {i:02d} / baseline {c["baseline"]["id"]}: {c["filename"]}',fontsize=9)
        dest = args.output/'error_cases'/f'comparison-{i:02d}.png'
        fig.savefig(dest,dpi=120); plt.close(fig)
        c['comparison_sheet'] = dest.relative_to(args.output).as_posix()
    if cases_data:
        write_json(args.output/'error_cases'/'case_comparisons.json',cases_data)


def main():
    if os.name=='nt':
        import ctypes
        # Diagnostic process only: avoid Windows crash dialogs delaying workers.
        # Does not change model, OCR algorithms, thresholds, or global settings.
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, default=ROOT/'data/custom/images/test')
    ap.add_argument('--labels', type=Path, default=ROOT/'data/custom/labels/test')
    ap.add_argument('--weights', type=Path, default=ROOT/'weights/best.pt')
    ap.add_argument('--output', type=Path, default=ROOT/'experiments/diagnostic-test')
    ap.add_argument('--ocr-gt', type=Path)
    ap.add_argument('--phase', choices=['all','audit','detect','pipeline','cases','report','retry-failures','finish'], default='all')
    ap.add_argument('--only', help='Run one named image in the pipeline worker; preserve original dataset ordinal.')
    ap.add_argument('--isolate-images', action='store_true', help='Use fresh process per pipeline image to bound allocator memory.')
    ap.add_argument('--case-only', help='Run just one selected representative case.')
    ap.add_argument('--isolate-cases', action='store_true', help='Isolate representative OCR experiments in fresh processes.')
    ap.add_argument('--worker-timeout',type=int,default=120,help='Diagnostic per-image safety timeout; timeout is an operational failure, never an empty OCR result.')
    args = ap.parse_args()
    for k in ('images','labels','weights','output'):
        setattr(args, k, getattr(args,k).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(args.output/'runtime_cache'))
    logging.basicConfig(filename=str(args.output/'diagnostic.log'), level=logging.INFO, encoding='utf-8')
    started = time.perf_counter()
    if args.phase=='finish':
        retry_failures(args)
        isolated_cases(args)
        report(args)
        print(f'FINISH DONE: {time.perf_counter()-started:.1f}s',flush=True)
        return
    if args.phase=='retry-failures':
        retry_failures(args)
        return
    phases = [('audit', audit), ('detect', detect_all), ('pipeline', pipeline), ('cases', cases), ('report', report)]
    for name, fn in phases:
        if args.phase in ('all',name):
            print(f'PHASE {name}', flush=True)
            fn(args)
    print(f'DONE {args.phase}: {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
