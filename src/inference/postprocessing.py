"""OCR post-processing for detected licence plates.

Pipeline:

    preprocessed plate
      -> aspect-ratio candidate + reliable gap: OCR two lines separately
         otherwise OCR the whole crop and group fragments
      -> order lines top-to-bottom, fragments left-to-right
      -> separate normalization from bounded structural correction/formatting
      -> abstain from API plate output on incomplete/unrecognized text
"""

import logging
import re
import traceback
from typing import Dict, List, Optional

import numpy as np

from ..utils.config import (
    OCR_ALLOWLIST,
    OCR_CRAFT_MAX_PIXELS,
    OCR_CRAFT_WORKSPACE_FACTOR,
    OCR_MEMORY_RESERVE_BYTES,
    OCR_LANGUAGES,
    OCR_SPLIT_TWO_LINE,
    OCR_TWO_LINE_ASPECT,
    OCR_USE_GPU,
)
# Keep apply_clahe as an alias for legacy diagnostic hooks; OCR does not call it.
from .preprocessing import apply_clahe, is_two_line, split_two_lines
from .runtime_memory import ocr_memory_admission

logger = logging.getLogger(__name__)

_CLEAN_RE = re.compile(r"[^A-Z0-9]")

# Single-line plate patterns (tried together, best match chosen):
#   car-like : province + letters-only series + 4/5 digits   (30A53103)
#   bike-like: province + letter+digit series + 4/5 digits   (68G166886)
_CAR_SINGLE_RE = re.compile(r"^([0-9]{2})([A-Z]{1,2})([0-9]{4,5})$")
_BIKE_SINGLE_RE = re.compile(r"^([0-9]{2})([A-Z][0-9])([0-9]{4,5})$")
# Top line of a 2-line plate: province + series (series may end with a digit).
_TOP_RE = re.compile(r"^([0-9]{2})([A-Z]{1,2}|[A-Z][0-9])$")

# --- OCR confusion tables -------------------------------------------------
# A plate series slot must be a LETTER; map a digit read there to the closest
# letter (G1 read as 61 -> 68G1).
_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}
# The province is the first two digits; map letters read there to the closest
# digit (38 read as 3B -> 38, 60 read as 6O -> 60, 52 read as S2 -> 52).
_PROVINCE_LETTER_TO_DIGIT = {
    "B": "8",
    "O": "0",
    "D": "0",
    "Q": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "A": "4",
    "E": "3",
    "J": "3",
}
# The serial number must be digits; map letters read there to digits
# (O/D -> 0, Z -> 2, S -> 5).
_LETTER_TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "A": "4",
    "T": "7",
}

_MAX_PLATE_CHARS = 9  # Vietnamese plates are 7-9 alphanumeric characters


def _clean(text: str) -> str:
    """Normalization only; never repair ambiguous characters here."""
    return _CLEAN_RE.sub("", text.upper())


def _cap_length(text: str, limit: int = _MAX_PLATE_CHARS) -> str:
    """Compatibility helper: preserve evidence; length is checked by patterns."""
    return text


def _format_number(digits: str) -> str:
    """Insert punctuation only; never repair, discard or truncate characters."""
    if re.fullmatch(r"[0-9]{5}", digits):
        return f"{digits[:3]}.{digits[3:]}"
    return digits


def format_vn_plate(lines: List[str]) -> str:
    """Format supported complete structures; preserve unrecognized normalized text.

    This helper is NOT a correctness validator. PlateOCR abstains from returning
    unrecognized structures as plate numbers; details retain the raw evidence.
    """
    return format_vn_plate_details(lines)['formatted_text']


def _format_single_line(text: str) -> str:
    candidates = []
    for kind, regex in (("car", _CAR_SINGLE_RE), ("bike", _BIKE_SINGLE_RE)):
        match = regex.match(text)
        if match:
            province, series, number = match.groups()
            candidates.append((kind, province, series, number))
    if not candidates:
        return text

    # Prefer an exact 5-digit number (VN standard), then the longest numeric group.
    candidates.sort(key=lambda c: (0 if len(c[3]) == 5 else 1, -len(c[3])))
    kind, province, series, number = candidates[0]
    number = _format_number(number)
    if kind == "bike":
        return f"{province}-{series} {number}"
    return f"{province}{series}-{number}"


def _structured_format(parts):
    """Only bounded patterns, no mutation. None means unsupported/incomplete."""
    if len(parts) == 1:
        if _CAR_SINGLE_RE.fullmatch(parts[0]) or _BIKE_SINGLE_RE.fullmatch(parts[0]):
            return _format_single_line(parts[0])
    elif len(parts) == 2:
        match = _TOP_RE.fullmatch(parts[0])
        if match and re.fullmatch(r'[0-9]{4,5}', parts[1]):
            province, series = match.groups()
            number = _format_number(parts[1])
            return f'{province}-{series} {number}' if series[-1].isdigit() else f'{province}{series}-{number}'
    return None


def format_vn_plate_details(lines: List[str]) -> dict:
    """Trace normalization separately from a maximum of ONE contextual repair.

    Two-line boundaries provide slots: top province/series, bottom 4/5 digits.
    A correction is considered only when every other slot already fits. For a
    single merged line, only province repair is enabled with a complete suffix.
    A structural match never certifies the plate text or province semantics.
    """
    raw = list(lines)
    normalized = [_clean(line) for line in raw]
    parts = [part for part in normalized if part]
    result = {'raw_lines':raw, 'normalized_lines':normalized,
              'corrected_lines':list(parts), 'corrections':[],
              'formatted_text':' '.join(parts), 'plate_number':'',
              'format_status':'unrecognized', 'text_verified':False}
    if not parts:
        result['format_status'] = 'empty'
        return result
    # Do not silently turn '?' or an unsupported glyph into a valid plate.
    if any(re.search(r'[^A-Za-z0-9\s.\-]', line) for line in raw):
        result['reason'] = 'unsupported_or_ambiguous_character'
        return result
    formatted = _structured_format(parts)
    if formatted is not None:
        result.update(formatted_text=formatted, plate_number=formatted, format_status='structured')
        return result
    candidates = []
    def consider(line_index, char_index, mapping, slot):
        old = parts[line_index][char_index]
        new = mapping.get(old)
        if not new or old == new:
            return
        trial = list(parts)
        trial[line_index] = trial[line_index][:char_index]+new+trial[line_index][char_index+1:]
        output = _structured_format(trial)
        if output is not None:
            candidates.append((trial, output, {'line_index':line_index, 'char_index':char_index,
                'before':old, 'after':new, 'slot':slot, 'reason':'all_other_slots_complete'}))
    if len(parts) in {1,2}:
        # Province repair is accepted only if series + full serial already fit.
        for position in range(min(2,len(parts[0]))):
            consider(0,position,_PROVINCE_LETTER_TO_DIGIT,'province')
    if len(parts) == 2:
        # Top can only be province + 1/2-character series; no long numeric tail.
        if len(parts[0]) in {3,4} and parts[0][:2].isdigit() and re.fullmatch(r'[0-9]{4,5}', parts[1]):
            consider(0,2,_DIGIT_TO_LETTER,'series')
        if _TOP_RE.fullmatch(parts[0]) and len(parts[1]) in {4,5}:
            for position in range(len(parts[1])):
                consider(1,position,_LETTER_TO_DIGIT,'serial')
    if len(candidates) == 1:
        corrected,formatted,change = candidates[0]
        result.update(corrected_lines=corrected, corrections=[change], formatted_text=formatted,
                      plate_number=formatted, format_status='heuristic_corrected')
    else:
        result['reason'] = 'ambiguous_corrections' if candidates else 'incomplete_or_unsupported_structure'
    return result


def _craft_resize_plan(shape, max_pixels):
    """Bound CRAFT's padded tensor, matching EasyOCR imgproc.resize_aspect_ratio.

    Only reduce its default 2560 canvas when necessary. Recognition still sees
    the original preprocessed pixels; EasyOCR maps detection boxes back itself.
    This bounds one full-resolution activation, not total model/workspace RAM.
    """
    height, width = map(int, shape[:2])
    if height <= 0 or width <= 0:
        raise ValueError('OCR input must have positive height and width')
    for canvas in range(2560, 31, -32):
        ratio = min(1.0, canvas / max(height, width))  # mag_ratio remains 1.
        resized_h, resized_w = int(height * ratio), int(width * ratio)
        if min(resized_h, resized_w) < 1:
            continue
        padded_h = ((resized_h + 31) // 32) * 32
        padded_w = ((resized_w + 31) // 32) * 32
        if padded_h * padded_w <= max_pixels:
            return {'canvas_size':canvas, 'mag_ratio':1.0,
                    'max_pixels':max_pixels, 'limited':canvas < 2560,
                    'resized_shape':[resized_h, resized_w],
                    'craft_tensor_shape':[1, 3, padded_h, padded_w],
                    'full_resolution_64ch_float32_bytes':64 * padded_h * padded_w * 4}
    raise ValueError('OCR aspect ratio cannot fit the CRAFT pixel budget')


class PlateOCR:
    """Wrapper around an EasyOCR reader with two-line handling and VN formatting."""

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_gpu: Optional[bool] = None,
        allowlist: Optional[str] = None,
        split_two_line: Optional[bool] = None,
        max_craft_pixels: Optional[int] = None,
    ):
        self.languages = list(languages) if languages else list(OCR_LANGUAGES)
        self.use_gpu = OCR_USE_GPU if use_gpu is None else use_gpu
        self.allowlist = allowlist or OCR_ALLOWLIST
        self.split_two_line = OCR_SPLIT_TWO_LINE if split_two_line is None else split_two_line
        self.max_craft_pixels = OCR_CRAFT_MAX_PIXELS if max_craft_pixels is None else max_craft_pixels
        if not isinstance(self.max_craft_pixels, int) or self.max_craft_pixels < 32 * 32:
            raise ValueError('max_craft_pixels must be an integer >= 1024')
        if OCR_CRAFT_WORKSPACE_FACTOR < 1 or OCR_MEMORY_RESERVE_BYTES < 0:
            raise ValueError('OCR memory workspace factor must be >= 1 and reserve >= 0')
        self._reader = None

    @property
    def reader(self):
        """Return the (lazily created) EasyOCR reader instance."""
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    @staticmethod
    def _join_fragments(ocr_res) -> str:
        """Join fragments of a single line left-to-right."""
        ordered = sorted(ocr_res, key=lambda item: item[0][0][0])
        return "".join(item[1] for item in ordered)

    @classmethod
    def _group_lines(cls, ocr_res) -> List[str]:
        """Group fragments into line texts using an adaptive Y threshold."""
        items = []
        for bbox, text, _conf in ocr_res:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            items.append(
                {
                    "cx": sum(xs) / len(xs),
                    "cy": sum(ys) / len(ys),
                    "h": (max(ys) - min(ys)) or 1.0,
                    "text": text,
                }
            )
        if not items:
            return []

        items.sort(key=lambda d: d["cy"])
        avg_h = sum(d["h"] for d in items) / len(items)
        threshold = max(avg_h * 0.6, 10.0)

        lines: List[List[dict]] = []
        current = [items[0]]
        for item in items[1:]:
            if abs(item["cy"] - current[-1]["cy"]) <= threshold:
                current.append(item)
            else:
                lines.append(current)
                current = [item]
        lines.append(current)

        return ["".join(d["text"] for d in sorted(line, key=lambda d: d["cx"])) for line in lines]

    def read_text(self, image: np.ndarray) -> str:
        """API-compatible string; incomplete text is retained in details, not a plate."""
        details = self.read_text_details(image)
        logger.info('OCR status=%s raw=%r normalized=%r corrections=%r formatted=%r format_status=%s plate=%r',
                    details['ocr_status'],
                    details['raw_lines'], details['normalized_lines'], details['corrections'],
                    details['formatted_text'], details['format_status'], details['plate_number'])
        return details['plate_number']

    def read_text_details(self, image: np.ndarray) -> dict:
        """Read an already-preprocessed image without further contrast transforms.

        Whole and split routes consume the same prepared pixels. Contrast
        enhancement belongs to preprocess_plate and its use_clahe setting.
        """
        trace = {'ocr_calls':[], 'split_row':None, 'split_status':'not_requested'}
        stage = 'split_decision'
        def read(part, label):
            nonlocal stage
            stage = 'craft_resize_plan'
            call = {'label':label, 'input_shape':list(part.shape),
                    'status':'started', 'results':[]}
            trace['ocr_calls'].append(call)
            plan = _craft_resize_plan(part.shape, self.max_craft_pixels)
            call['resize_plan'] = plan
            kwargs = {'allowlist':self.allowlist}
            if plan['limited']:
                kwargs['canvas_size'] = plan['canvas_size']
                logger.info('OCR CRAFT budget: input=%s plan=%s', part.shape, plan)
            stage = 'reader_initialization'
            reader = self.reader
            if not self.use_gpu:
                stage = 'memory_admission'
                admission = ocr_memory_admission(plan, OCR_CRAFT_WORKSPACE_FACTOR, OCR_MEMORY_RESERVE_BYTES)
                call['memory_admission'] = admission
                if not admission['allowed']:
                    raise MemoryError(
                        'OCR commit budget refused before CRAFT: '
                        f"available={admission['available_commit_bytes']} bytes, "
                        f"required_headroom={admission['required_headroom_bytes']} bytes")
            stage = 'easyocr_readtext'
            values = reader.readtext(part, **kwargs)
            call.update(status='completed', results=[
                {'bbox':np.asarray(box).tolist(), 'text':text, 'confidence':float(conf)}
                for box,text,conf in values])
            return values
        try:
            line_texts = None
            if self.split_two_line and is_two_line(image, OCR_TWO_LINE_ASPECT):
                halves = split_two_lines(image)
                trace['split_status'] = 'rejected_no_reliable_gap'
                if halves is not None:
                    trace.update(split_row=halves[0].shape[0], split_status='accepted_gap')
                    # Splitting must not apply CLAHE again or bypass use_clahe=False.
                    texts = [self._join_fragments(read(half, label))
                             for label,half in zip(('top','bottom'),halves)]
                    # Do not collapse a missing half into a fake single-line plate.
                    if all(_clean(text) for text in texts):
                        line_texts = texts
                    else:
                        trace['split_status'] = 'fallback_missing_line'
            if line_texts is None:
                values = read(image, 'whole')
                stage = 'line_grouping'
                line_texts = self._group_lines(values)
            stage = 'formatting'
            details = format_vn_plate_details(line_texts)
            has_text = any(item['text'].strip() for call in trace['ocr_calls'] for item in call['results'])
            ocr_status = 'ok' if details['plate_number'] else 'unrecognized_format' if has_text else 'empty_ocr'
            return dict(details, **trace, ocr_status=ocr_status)
        except Exception as exc:  # never let OCR bring the API down
            logger.exception('OCR status=ocr_runtime_error stage=%s calls=%r: %s', stage, trace['ocr_calls'], exc)
            if trace['ocr_calls'] and trace['ocr_calls'][-1]['status'] == 'started':
                trace['ocr_calls'][-1].update(status='ocr_runtime_error', error=str(exc), error_type=type(exc).__name__)
            # Preserve completed fragments if a later half/whole call failed.
            # Do not run the formatter on an invented empty OCR response.
            raw = [item['text'] for call in trace['ocr_calls'] for item in call['results']]
            return dict(raw_lines=raw, normalized_lines=[], corrected_lines=[], corrections=[],
                        formatted_text='', plate_number='', text_verified=False,
                        format_status='not_completed', ocr_status='ocr_runtime_error',
                        error=str(exc), error_type=type(exc).__name__, error_stage=stage,
                        traceback=traceback.format_exc(), **trace)


def build_detection(confidence: float, plate_number: str) -> Dict[str, object]:
    """Build a single API result entry (same schema as the original backend)."""
    return {"confidence": float(confidence), "plate_number": plate_number}


__all__ = ["PlateOCR", "format_vn_plate", "format_vn_plate_details", "build_detection"]
