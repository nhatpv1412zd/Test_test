import numpy as np
import pytest


@pytest.fixture(autouse=True)
def healthy_memory_for_fake_readers(monkeypatch):
    # These are unit/fake-reader tests; host load must not decide their outcome.
    from src.inference import runtime_memory
    monkeypatch.setattr(runtime_memory, '_available_commit_bytes', lambda: 16 * 1024**3)

from src.inference.postprocessing import (
    PlateOCR, build_detection, format_vn_plate, format_vn_plate_details,
    _clean, _cap_length, _format_number,
)


def _box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class _FakeReader:
    """Returns a different response for each successive readtext call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.allowlist = None

    def readtext(self, image, allowlist=None):
        self.allowlist = allowlist
        response = self._responses[self.calls] if self.calls < len(self._responses) else []
        self.calls += 1
        return response


def test_build_detection_schema():
    result = build_detection(0.987654, "51A-123.45")
    assert result == {"confidence": 0.987654, "plate_number": "51A-123.45"}
    assert isinstance(result["confidence"], float)


def test_build_detection_casts_confidence_to_float():
    assert isinstance(build_detection(1, "X")["confidence"], float)


# --- format_vn_plate -------------------------------------------------------

def test_format_two_line_motorcycle():
    assert format_vn_plate(["68-G1", "668.86"]) == "68-G1 668.86"


def test_format_two_line_four_digit_bike():
    assert format_vn_plate(["78-H8", "5968"]) == "78-H8 5968"


def test_format_two_line_car():
    assert format_vn_plate(["30A", "531.03"]) == "30A-531.03"


def test_format_single_line_car():
    assert format_vn_plate(["30A53103"]) == "30A-531.03"


def test_format_single_line_motorcycle_merged():
    assert format_vn_plate(["68G166886"]) == "68-G1 668.86"


def test_format_two_line_repairs_series_digit_to_letter():
    assert format_vn_plate(["68-61", "668.86"]) == "68-G1 668.86"


def test_overlong_serial_is_preserved_without_province_repair_or_truncation():
    assert format_vn_plate(["3BA", "531032"]) == "3BA 531032"
    assert format_vn_plate_details(["3BA", "531032"])['plate_number'] == ''


def test_vn_cleaner_maps_o_to_zero_in_province():
    assert format_vn_plate(["6OA", "12345"]) == "60A-123.45"


def test_vn_cleaner_maps_j_to_three_in_province():
    assert format_vn_plate(["J2A", "12345"]) == "32A-123.45"


def test_vn_cleaner_maps_letters_to_digits_in_serial():
    # serial "66S86": S -> 5  =>  665.86
    assert format_vn_plate(["68-G1", "66S86"]) == "68-G1 665.86"


def test_vn_cleaner_maps_s_to_five_and_z_to_two_in_province():
    assert format_vn_plate(["S2A", "12345"]) == "52A-123.45"
    assert format_vn_plate(["1ZA", "12345"]) == "12A-123.45"


def test_format_empty_returns_empty():
    assert format_vn_plate([]) == ""


# --- read_text: single line vs two-line split ------------------------------

def test_read_text_single_line_car():
    ocr = PlateOCR(languages=["en"], use_gpu=False)
    # wide image => single line
    ocr._reader = _FakeReader([[(_box(0, 10, 90, 30), "30A-531.03", 0.9)]])
    assert ocr.read_text(np.zeros((50, 200), dtype=np.uint8)) == "30A-531.03"


def test_read_text_two_line_split():
    ocr = PlateOCR(languages=["en"], use_gpu=False)
    # Two actual ink bands + gap, not merely a square blank image.
    ocr._reader = _FakeReader(
        [
            [(_box(0, 5, 80, 25), "68-G1", 0.9)],    # top half
            [(_box(0, 5, 90, 25), "668.86", 0.9)],   # bottom half
        ]
    )
    image = np.full((200,150),255,dtype=np.uint8)
    image[30:70,20:130] = 0
    image[130:170,20:130] = 0
    assert ocr.read_text(image) == "68-G1 668.86"
    assert ocr._reader.calls == 2


def test_read_text_returns_empty_string_without_detections():
    ocr = PlateOCR(languages=["en"], use_gpu=False)
    ocr._reader = _FakeReader([])
    assert ocr.read_text(np.zeros((50, 200), dtype=np.uint8)) == ""


@pytest.mark.parametrize('lines',[
    ['L8953'], ['68-61','G6Q'], ['54K697','02L52'], ['29-PI','239.641'],
    ['159-CL','05478'], ['30A','12?45'], ['30A','12Y45'], ['30A','123'],
    ['30A','123456'], ['30A','123','45'], ['SZA','12345'],
])
def test_unrecognized_text_never_becomes_valid_plate(lines):
    result = format_vn_plate_details(lines)
    assert result['format_status'] == 'unrecognized'
    assert result['plate_number'] == ''
    assert result['raw_lines'] == lines
    assert result['corrections'] == []
    assert _clean(result['formatted_text']) == ''.join(_clean(line) for line in lines)


def test_normalization_is_not_character_correction():
    assert _clean(' o0 - i1 b8\n s5. ') == 'O0I1B8S5'
    assert _format_number('G6Q') == 'G6Q'
    assert _format_number('123456') == '123456'
    assert _format_number('12Y45') == '12Y45'
    assert _cap_length('30A123456789') == '30A123456789'


def test_contextual_correction_has_trace_and_never_claims_verified():
    result = format_vn_plate_details(['68-G1','66S86'])
    assert result['plate_number'] == '68-G1 665.86'
    assert result['normalized_lines'] == ['68G1','66S86']
    assert result['corrected_lines'] == ['68G1','66586']
    assert result['corrections'] == [{'line_index':1,'char_index':2,'before':'S','after':'5',
                                     'slot':'serial','reason':'all_other_slots_complete'}]
    assert result['format_status'] == 'heuristic_corrected'
    assert result['text_verified'] is False


def test_single_line_numeric_string_does_not_invent_series():
    assert format_vn_plate_details(['686166886'])['plate_number'] == ''


def test_square_image_without_gap_uses_one_whole_crop_call():
    ocr = PlateOCR(use_gpu=False)
    reader = _FakeReader([[(_box(0,10,80,30),'30A12345',.9)]])
    ocr._reader = reader
    result = ocr.read_text_details(np.zeros((200,150),np.uint8))
    assert result['split_status'] == 'rejected_no_reliable_gap'
    assert result['plate_number'] == '30A-123.45'
    assert result['ocr_calls'][0]['input_shape'] == [200,150]
    assert reader.calls == 1


def test_missing_split_half_falls_back_to_whole_crop():
    ocr = PlateOCR(use_gpu=False)
    image = np.full((200,150),255,np.uint8)
    image[30:70,20:130] = 0; image[130:170,20:130] = 0
    reader = _FakeReader([[(_box(0,0,80,30),'30A',.8)], [],
        [(_box(0,10,90,30),'30A12345',.9)]])
    ocr._reader = reader
    result = ocr.read_text_details(image)
    assert reader.calls == 3
    assert result['split_status'] == 'fallback_missing_line'
    assert result['plate_number'] == '30A-123.45'
    assert [call['label'] for call in result['ocr_calls']] == ['top','bottom','whole']


def test_partial_ocr_is_traceable_but_api_does_not_emit_it_as_plate():
    ocr = PlateOCR(use_gpu=False)
    ocr._reader = _FakeReader([[(_box(0,10,90,30),'L8953',.3)]])
    result = ocr.read_text_details(np.zeros((50,200),np.uint8))
    assert result['raw_lines'] == ['L8953']
    assert result['formatted_text'] == 'L8953'
    assert result['plate_number'] == ''
    ocr._reader = _FakeReader([[(_box(0,10,90,30),'L8953',.3)]])
    assert ocr.read_text(np.zeros((50,200),np.uint8)) == ''


def test_whole_crop_groups_lines_and_fragments_in_reading_order():
    ocr = PlateOCR(use_gpu=False,split_two_line=False)
    ocr._reader = _FakeReader([[
        (_box(40,60,80,80),'345',.9),(_box(40,10,80,30),'A',.9),
        (_box(0,60,30,80),'12',.9),(_box(0,10,30,30),'30',.9)]])
    result = ocr.read_text_details(np.zeros((100,200),np.uint8))
    assert result['raw_lines'] == ['30A','12345']
    assert result['plate_number'] == '30A-123.45'


@pytest.mark.parametrize('use_clahe', [False, True])
def test_split_uses_exact_preprocessed_pixels_and_respects_clahe_toggle(monkeypatch, use_clahe):
    from src.inference import preprocessing as pre
    from src.inference import postprocessing as post

    # Nonbinary values ensure an accidental second CLAHE changes real pixels.
    image = np.full((200, 150), 220, np.uint8)
    for start in [30, 130]:
        image[start:start+40, 20:130] = np.arange(110, dtype=np.uint8)[None, :] // 2 + 20
    calls = []
    original = pre.apply_clahe
    def count_clahe(part):
        calls.append(part.shape)
        return original(part)
    monkeypatch.setattr(pre, 'apply_clahe', count_clahe)
    def forbidden_second_clahe(part):
        pytest.fail('OCR must not apply CLAHE after preprocess_plate')
    # Keep this legacy instrumentation alias while asserting no calls to it.
    monkeypatch.setattr(post, 'apply_clahe', forbidden_second_clahe)
    prepared = pre.preprocess_plate(image, upscale=1, use_deskew=False, use_clahe=use_clahe)
    halves = pre.split_two_lines(prepared)
    assert halves is not None
    class CaptureReader(_FakeReader):
        def __init__(self):
            super().__init__([[(_box(0,0,70,20),'30A',.8)], [(_box(0,0,70,20),'12345',.8)]])
            self.inputs = []
        def readtext(self, part, **kwargs):
            self.inputs.append(part.copy())
            return super().readtext(part, **kwargs)
    reader = CaptureReader()
    ocr = PlateOCR(use_gpu=False); ocr._reader = reader
    details = ocr.read_text_details(prepared)
    assert details['split_status'] == 'accepted_gap'
    assert details['plate_number'] == '30A-123.45'
    assert len(calls) == int(use_clahe)
    assert len(reader.inputs) == 2
    for sent, expected in zip(reader.inputs, halves):
        assert np.array_equal(sent, expected)
    assert np.array_equal(np.concatenate(reader.inputs), prepared)


def test_whole_route_uses_exact_input_without_implicit_clahe(monkeypatch):
    from src.inference import postprocessing as post
    def forbidden_clahe(part):
        pytest.fail('Whole route must not apply contrast transforms')
    monkeypatch.setattr(post, 'apply_clahe', forbidden_clahe)
    image = np.arange(200, dtype=np.uint8)[None, :].repeat(50, axis=0)
    inputs = []
    class Reader:
        def readtext(self, part, **kwargs):
            inputs.append(part.copy())
            return [(_box(0,0,90,20),'30A12345',.9)]
    ocr = PlateOCR(use_gpu=False); ocr._reader = Reader()
    assert ocr.read_text(image) == '30A-123.45'
    assert len(inputs) == 1
    assert np.array_equal(inputs[0], image)


@pytest.mark.parametrize('responses,status,format_status',[
    ([], 'empty_ocr', 'empty'),
    ([(_box(0,0,90,20),'L8953',.5)], 'unrecognized_format', 'unrecognized'),
    ([(_box(0,0,90,20),'...',.5)], 'unrecognized_format', 'empty'),
    ([(_box(0,0,90,20),'17AK04470',.9)], 'ok', 'structured'),
])
def test_ocr_completion_status_is_separate_from_formatter(responses,status,format_status):
    ocr=PlateOCR();ocr._reader=_FakeReader([responses])
    result=ocr.read_text_details(np.zeros((50,200),np.uint8))
    assert result['ocr_status']==status
    assert result['format_status']==format_status
    assert result['ocr_calls'][0]['status']=='completed'
    assert 'error' not in result


def test_allocator_failure_has_failed_call_traceback_and_runtime_status(caplog):
    class Reader:
        def readtext(self,part,**kwargs):
            raise RuntimeError('DefaultCPUAllocator: not enough memory: 1656750080 bytes')
    ocr=PlateOCR();ocr._reader=Reader()
    result=ocr.read_text_details(np.zeros((50,200),np.uint8))
    assert result['ocr_status']=='ocr_runtime_error'
    assert result['format_status']=='not_completed'
    assert result['plate_number']==''  # string API stays compatible
    assert result['error_type']=='RuntimeError'
    assert result['error_stage']=='easyocr_readtext'
    assert '1656750080' in result['error']
    assert 'RuntimeError' in result['traceback']
    assert result['ocr_calls'][0]['status']=='ocr_runtime_error'
    assert any(r.exc_info and 'ocr_runtime_error' in r.getMessage() for r in caplog.records)
    assert 'status=empty_ocr' not in caplog.text


def test_failed_bottom_preserves_completed_top_without_formatting_partial_plate():
    class Reader:
        def __init__(self):self.calls=0
        def readtext(self,part,**kwargs):
            self.calls+=1
            if self.calls==2:raise MemoryError('bottom allocation failed')
            return [(_box(0,0,90,20),'30A',.9)]
    image=np.full((200,150),255,np.uint8)
    image[30:70,20:130]=0;image[130:170,20:130]=0
    ocr=PlateOCR();ocr._reader=Reader()
    result=ocr.read_text_details(image)
    assert result['raw_lines']==['30A']
    assert result['ocr_status']=='ocr_runtime_error'
    assert result['plate_number']==''
    assert [c['status'] for c in result['ocr_calls']]==['completed','ocr_runtime_error']
    assert result['error_type']=='MemoryError'


def test_reader_initialization_failure_is_runtime_error(monkeypatch):
    def fail(self):raise OSError('model unavailable')
    monkeypatch.setattr(PlateOCR,'reader',property(fail))
    result=PlateOCR().read_text_details(np.zeros((50,200),np.uint8))
    assert result['ocr_status']=='ocr_runtime_error'
    assert result['error_stage']=='reader_initialization'
    assert result['format_status']=='not_completed'


def test_grouping_failure_is_not_mislabeled_as_easyocr_failure(monkeypatch):
    ocr=PlateOCR();ocr._reader=_FakeReader([[(_box(0,0,90,20),'30A12345',.9)]])
    def fail(values):raise ValueError('invalid grouping geometry')
    monkeypatch.setattr(ocr,'_group_lines',fail)
    result=ocr.read_text_details(np.zeros((50,200),np.uint8))
    assert result['ocr_status']=='ocr_runtime_error'
    assert result['error_stage']=='line_grouping'
    assert result['raw_lines']==['30A12345']
    assert result['ocr_calls'][0]['status']=='completed'


@pytest.mark.parametrize('shape',[(3670,3750),(2692,5260),(1160,1444),(1441,1441),(50,200),(4096,70)])
def test_craft_plan_matches_easyocr_padding_and_budget(shape):
    from src.inference.postprocessing import _craft_resize_plan
    budget=2*1024*1024
    plan=_craft_resize_plan(shape,budget)
    h,w=shape;ratio=min(plan['canvas_size']/max(shape),1.)
    expected=[((int(h*ratio)+31)//32)*32,((int(w*ratio)+31)//32)*32]
    assert plan['craft_tensor_shape']==[1,3,*expected]
    assert expected[0]*expected[1]<=budget
    assert plan['full_resolution_64ch_float32_bytes']<=512*1024*1024
    if plan['limited']:
        # Next larger canvas cannot fit, so we keep the most detail allowed.
        r=min((plan['canvas_size']+32)/max(shape),1.)
        assert ((int(h*r)+31)//32)*32*((int(w*r)+31)//32)*32>budget


def test_padding_can_exceed_budget_even_when_unpadded_image_fits():
    from src.inference.postprocessing import _craft_resize_plan
    shape=(1441,1441)
    assert shape[0]*shape[1]<2*1024*1024
    plan=_craft_resize_plan(shape,2*1024*1024)
    assert plan['limited'] is True
    assert plan['canvas_size']==1440


@pytest.mark.parametrize('shape,canvas',[( (3670,3750),1440),((2692,5260),1984),((1160,1444),None)])
def test_only_oversized_craft_canvas_changes_while_recognition_pixels_are_preserved(shape,canvas):
    calls=[]
    image=np.zeros(shape,np.uint8)
    class Reader:
        def readtext(self,part,**kwargs):
            calls.append(kwargs)
            assert part is image
            return []
    ocr=PlateOCR(split_two_line=False);ocr._reader=Reader()
    result=ocr.read_text_details(image)
    assert calls[0].get('canvas_size')==canvas
    assert calls[0].get('mag_ratio',1.)==1.
    assert result['ocr_calls'][0]['resize_plan']['limited']==(canvas is not None)


def test_every_split_and_fallback_call_uses_same_craft_budget():
    class Reader(_FakeReader):
        def readtext(self,part,**kwargs):
            assert kwargs['canvas_size']<2560
            return super().readtext(part,allowlist=kwargs['allowlist'])
    image=np.full((200,150),255,np.uint8)
    image[30:70,20:130]=0;image[130:170,20:130]=0
    ocr=PlateOCR(max_craft_pixels=4096)
    ocr._reader=Reader([[(_box(0,0,90,20),'30A',.9)],[],[(_box(0,0,90,20),'30A12345',.9)]])
    result=ocr.read_text_details(image)
    assert result['split_status']=='fallback_missing_line'
    assert result['plate_number']=='30A-123.45'
    assert len(result['ocr_calls'])==3
    for call in result['ocr_calls']:
        n,c,h,w=call['resize_plan']['craft_tensor_shape']
        assert h*w<=4096


@pytest.mark.parametrize('budget',[0,-1,1023,1.5])
def test_invalid_craft_budget_is_not_silently_disabled(budget):
    with pytest.raises(ValueError,match='max_craft_pixels'):
        PlateOCR(max_craft_pixels=budget)


def test_low_commit_refuses_native_entry_with_runtime_status(monkeypatch):
    from src.inference import runtime_memory
    monkeypatch.setattr(runtime_memory,'_available_commit_bytes',lambda:64*1024**2)
    class Reader:
        def readtext(self,*args,**kwargs):pytest.fail('unsafe native entry must be refused')
    ocr=PlateOCR();ocr._reader=Reader()
    result=ocr.read_text_details(np.zeros((50,200),np.uint8))
    assert result['ocr_status']=='ocr_runtime_error'
    assert result['format_status']=='not_completed'
    assert result['error_type']=='MemoryError'
    assert result['error_stage']=='memory_admission'
    call=result['ocr_calls'][0]
    assert call['status']=='ocr_runtime_error'
    assert call['memory_admission']['allowed'] is False
    assert 'OCR commit budget refused before CRAFT' in result['error']


@pytest.mark.parametrize('available,allowed',[(1000,False),(2000,True),(None,True)])
def test_memory_admission_boundary_and_unsupported_platform(monkeypatch,available,allowed):
    from src.inference import runtime_memory
    monkeypatch.setattr(runtime_memory,'_available_commit_bytes',lambda:available)
    result=runtime_memory.ocr_memory_admission({'full_resolution_64ch_float32_bytes':450},4,200)
    assert result['required_headroom_bytes']==2000
    assert result['allowed'] is allowed
    assert result['applied']==(available is not None)


def test_memory_query_failure_is_traceable_and_does_not_enter_native(monkeypatch):
    from src.inference import runtime_memory
    def fail():raise OSError('commit query failed')
    monkeypatch.setattr(runtime_memory,'_available_commit_bytes',fail)
    ocr=PlateOCR();ocr._reader=_FakeReader([])
    result=ocr.read_text_details(np.zeros((50,200),np.uint8))
    assert result['error_type']=='OSError'
    assert result['error_stage']=='memory_admission'
    assert result['ocr_status']=='ocr_runtime_error'
    assert ocr._reader.calls==0
