"""The single candidate: geometric reading lines, independent of plate text."""
import numpy as np


def layout_fragments(ocr_res):
    """Project boxes onto their dominant baseline; retain small inline glyphs.

    An isolated, much smaller box is not part of the dominant text block.
    Keep its original raw OCR in the caller's trace and report its exclusion.
    No confidence/text/regex/ground-truth score participates in this layout.
    """
    if not ocr_res:
        return {'line_texts':[], 'line_indices':[], 'excluded_fragments':[], 'angle_degrees':0.0}
    boxes=np.asarray([item[0] for item in ocr_res],dtype=float)
    edges=((boxes[:,1]-boxes[:,0])+(boxes[:,2]-boxes[:,3]))/2
    edges[edges[:,0]<0]*=-1
    angles=np.arctan2(edges[:,1],edges[:,0])
    widths=np.linalg.norm(edges,axis=1)
    order=np.argsort(angles)
    angle=float(angles[order[np.searchsorted(np.cumsum(widths[order]),widths.sum()/2)]])
    along=np.array([np.cos(angle),np.sin(angle)])
    normal=np.array([-np.sin(angle),np.cos(angle)])
    centers=boxes.mean(axis=1)
    us=centers@along;vs=centers@normal
    heights=(np.abs((boxes[:,3]-boxes[:,0])@normal)+np.abs((boxes[:,2]-boxes[:,1])@normal))/2
    scale=max(float(np.median(heights)),1.)
    tolerance=max(scale*.6,10.)
    main=[i for i in range(len(boxes)) if heights[i]>=scale*.25]
    # A lone large box cannot establish a dominant text block against others.
    if len(main)<2:main=list(range(len(boxes)))
    groups=[]
    for i in sorted(main,key=lambda j:vs[j]):
        nearest=min(range(len(groups)),key=lambda k:abs(vs[i]-np.mean(vs[groups[k]]))) if groups else None
        if nearest is not None and abs(vs[i]-np.mean(vs[groups[nearest]]))<=tolerance:
            groups[nearest].append(i)
        else:groups.append([i])
    excluded=[]
    for i in range(len(boxes)):
        if i in main:continue
        nearest=min(range(len(groups)),key=lambda k:abs(vs[i]-np.mean(vs[groups[k]])))
        distance=float(abs(vs[i]-np.mean(vs[groups[nearest]])))
        if distance<=tolerance:
            groups[nearest].append(i)
        else:
            excluded.append({'index':i,'reason':'isolated_small_box',
                'normal_height':float(heights[i]),'distance_to_nearest_line':distance})
    groups.sort(key=lambda g:float(np.mean(vs[g])))
    for group in groups:group.sort(key=lambda i:us[i])
    return {'line_texts':[''.join(ocr_res[i][1] for i in group) for group in groups],
        'line_indices':groups, 'excluded_fragments':excluded,
        'angle_degrees':float(np.degrees(angle)), 'normal_heights':heights.tolist(),
        'projected_centers':np.column_stack((us,vs)).tolist(),
        'reference_height':scale,'line_tolerance':tolerance}


if __name__=='__main__':
    from pathlib import Path
    from compare_alpr_cases import load,dump
    from src.inference.postprocessing import format_vn_plate_details
    root=Path(__file__).resolve().parents[1]/'experiments/plate-A-recognition-fix'
    results={}
    for case in ['A','B']:
        source=load(root/('before-'+case)/'trace.json')
        details=source['details']
        assert details['ocr_status']!='ocr_runtime_error'
        assert all(c['status']=='completed' and c['memory_admission']['allowed'] for c in details['ocr_calls'])
        raw=details['ocr_calls'][0]['results']
        result=layout_fragments([(item['bbox'],item['text'],item['confidence']) for item in raw])
        results[case]={'layout':result,'formatted':format_vn_plate_details(result['line_texts']),
                       'engine_rerun':False,'candidate':'geometric_line_layout_only'}
    dump(root/'single-candidate.json',results)
    print({case:result['formatted']['plate_number'] for case,result in results.items()})
