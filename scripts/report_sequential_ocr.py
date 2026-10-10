"""Package existing two-crop experiments; no model inference."""
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from debug_sequential_ocr import OUT,package
from compare_alpr_cases import load,dump
from src.inference.postprocessing import _craft_resize_plan


def walk_memory(value):
    values=[]
    if isinstance(value,dict):
        if 'PeakWorkingSetSize' in value:values.append(value['PeakWorkingSetSize'])
        for child in value.values():values+=walk_memory(child)
    elif isinstance(value,list):
        for child in value:values+=walk_memory(child)
    return values


def main():
    package()
    runs=load(OUT/'runs.json')
    rows=[];replays=[]
    for run in runs:
        tag=run['tag']
        if tag not in ['plate-alone','sign-alone','plate-sign-shared','sign-plate-shared','plate-sign-fresh','sign-plate-fresh','default12-repeat'] and not tag.startswith('ocr1-'):continue
        trace=load(OUT/tag/'trace.json');calls=trace['calls']
        if 'measured_pid' in run:
            peak=run['memory_peaks']['PeakWorkingSetSize'];prefix=''
        else:
            peak=max(walk_memory(trace),default=0);prefix='≥'
        completed=sum(c['stage']=='completed' for c in calls)
        last=calls[-1];headroom=last.get('last_conv',{}).get('before',{}).get('system',{}).get('avail_pagefile',0)
        rows.append(f'| [{tag}]({tag}/trace.json) | {run["order"]} | {run["reader_mode"]} | '+
            f'{trace.get("ocr_num_threads","auto (initial12)")} | {completed}/{len(run["order"])} | '+
            f'{run["exit_code"]} | {last.get("last_conv",{}).get("name","—")} | {headroom/1024**2:.0f} | '+
            f'{prefix}{peak/1024**2:.0f} | {run["elapsed_seconds"]:.2f} |')
        if run['native_crash']:
            call=calls[-1];plan=_craft_resize_plan(call['input_shape'],2*1024*1024)
            available=call['craft_before']['system']['avail_pagefile'];required=4*plan['full_resolution_64ch_float32_bytes']+128*1024**2
            replays.append({'tag':tag,'available_commit_at_craft_entry':available,
                'current_policy_required':required,'would_refuse':available<required,
                'counterfactual_only':True})
    dump(OUT/'historical-admission-replay.json',replays)
    final=load(OUT/'api-persistent-final/trace.json')
    assert final['status']=='completed'
    assert [r['http_status'] for r in final['requests']]==[200,200,200]
    response_A={'results':[{'confidence':0.7889413237571716,'plate_number':''},{'confidence':0.5282004475593567,'plate_number':''}]}
    assert final['requests'][0]['response']==response_A
    assert final['requests'][2]['response']==response_A
    assert final['requests'][1]['response']['results'][0]['plate_number']=='17AK-044.70'
    prior_B=load(ROOT/'experiments/ocr-runtime-fix/B-after/trace.json')['traces'][0]['details']['ocr_calls'][0]['results']
    assert final['requests'][1]['ocr'][0]['details']['ocr_calls'][0]['results']==prior_B
    assert all(c['same_pixels_as_frozen_input'] for r in final['requests'] for c in r['ocr'])
    before=ast.parse((OUT/'before/src/inference/postprocessing.py').read_text(encoding='utf-8'))
    after=ast.parse((ROOT/'src/inference/postprocessing.py').read_text(encoding='utf-8'))
    functions=lambda tree:{n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert functions(before)==functions(after)
    dump(OUT/'regression-checks.json',{'api_http':[200,200,200],'B_raw_boxes_text_confidence_exact_match':True,
        'all_preprocessed_pixels_unchanged':True,'formatter_and_craft_resize_plan_ast_unchanged':True,
        'historical_native_entries_refused_by_policy':all(x['would_refuse'] for x in replays)})
    api_rows=[]
    for i,request in enumerate(final['requests']):
        for call in request['ocr']:
            details=call['details'];memory=details['ocr_calls'][0]['memory_admission']
            api_rows.append(f'| {i+1}/{request["case"]} | {call["index"]} | {details["ocr_status"]} | '+
                f'{memory["available_commit_bytes"]/1024**2:.0f} | {memory["required_headroom_bytes"]/1024**2:.0f} | '+
                f'{details.get("error_stage","—")} | {request["elapsed_seconds"]:.3f} |')
    test_result=(OUT/'pytest.log').read_text(encoding='utf-8').strip().splitlines()[-1]
    report='''# Sequential ALPR native crash — resource admission patch

## 1. Kết luận đúng phạm vi

**Full API A → B → A đã hoàn tất trên cùng FastAPI service/Reader, không chết process; B vẫn đúng `17AK-044.70`.** Patch nhỏ kiểm tra commit headroom sau khi Reader đã nạp và trước mỗi call CRAFT. Nếu không đủ ngân sách workspace, trả **ocr_runtime_error / MemoryError / memory_admission** cùng traceback, không gọi OCR native và không gọi đó là empty OCR.

**Chưa xác định được dòng C++ lỗi bên trong c10.dll, và chưa khẳng định đã loại bỏ mọi native fault.** Các dữ liệu đã tái lập cho thấy native crash không đòi hỏi một ảnh nhất định, thứ tự nhất định, hoặc Reader dùng lại: dưới commit headroom thấp, cả hai crop đều crash ngay lần đầu. Khi headroom cao hơn, cùng cấu hình mặc định chạy được cả hai. Vì vậy patch xử lý đường vào thiếu tài nguyên đã quan sát; nó là admission policy dựa trên profile, **không phải sửa binary PyTorch hoặc bảo đảm cô lập process tuyệt đối**.

Không đổi ảnh, crop, preprocessing, OCR candidates, character correction, formatter, YOLO threshold/checkpoint hoặc dependency. Không đổi thread count production. Không mở benchmark dataset. Bài test chỉ dùng đúng hai crop A và ảnh B làm control API.

## 2. Thiết kế đối chứng

`scripts/debug_sequential_ocr.py`: mỗi thử nghiệm nặng trong một subprocess, chạy **nối tiếp**, timeout180s; faulthandler ghi runtime.log, exit.json ghi code/latency/peak RSS. Tất cả matrix giữ YOLO resident và giữ ảnh gốc, gọi OCR trong một request thread. Fixed inputs lấy nguyên PNG sau preprocessing từ task trước, kèm SHA256; không rerun recognition preprocessing hoặc tạo candidate mới.

- 0 = biển thật A, input readtext3750×3670; CRAFT canvas1440, tensor1×3×1440×1440.
- 1 = bảng quảng cáo A, input5260×2692; canvas1984, tensor1×3×1024×1984.
- Cùng EasyOCR1.7.2 / PyTorch2.14.1+cpu, allowlist/decoder/threshold/pixel budget2Mi như production.
- Shared: một Reader lazy cho cả cặp. Fresh: bỏ Reader và gc.collect trước tạo Reader tiếp theo; không giữ hai model nặng song song. Đây chỉ là đối chứng, không phải patch production.
- Phase CPU1 áp dụng sau YOLO, giữ các điều kiện còn lại. Trước đó có một pilot CPU1 trước YOLO (`plate-sign-one-thread`); pilot không nằm trong bảng so sánh chính.
- `num_threads` khởi tạo mặc định12; sau YOLO lượt repeat đo8. Vì vậy không gán nhãn mọi lượt ban đầu là “OCR12 threads” khi chưa đo tại call. CPU1 là diagnostic, không bật trong production.

**Headroom hệ thống không đứng yên.** Loạt mặc định đầu có available commit tại entry CRAFT khoảng1.3GB; loạt CPU1 có khoảng2.3GB. Không dùng việc CPU1 thành công làm bằng chứng root cause đa luồng. Đã chạy lại mặc định sau đó, nó cũng thành công.

## 3. Từng thử nghiệm (0/1 là detection index)

RSS và headroom ở bảng tính bằng MiB; headroom là snapshot trước convolution cuối ghi được, không phải memory budget. Cột completed cho biết OCR calls hoàn tất / số calls dự định; nếu call đầu crash thì call2 chưa được thực hiện.

| Experiment / trace | Order | Reader | OCR threads | Completed | Exit | Last conv | Commit headroom | Peak RSS | Wall s |
|---|---|---|---|---|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Mỗi thư mục có `runtime.log` faulthandler, `trace.json` gồm memory trước/sau Reader/OCR/CRAFT và last_conv, `exit.json`. [Tất cả run records](runs.json).

**Lưu ý đo lường:** six runs đầu watcher ngoài process theo PID launcher của Windows venv, không phải interpreter thực. Không dùng con số~4MiB đó làm RSS model. Cột RSS cho sáu lượt này lấy PeakWorkingSetSize do chính worker đo, là **lower bound trước fatal**, đánh dấu≥. Sau đó đã sửa watcher đọc worker_pid.json và đo đúng OS process, gồm peak working set/peak private commit. Không tính launcher là model worker.

Các native runs đều `0xC0000005`, last Python F.conv2d trong `basenet.slice1.3`. Windows Application Error Event1000 khớp6 lượt, cùng faulting module `torch/lib/c10.dll`, offset0x91444. [Windows events](windows-crash-events.json). Không có PDB/native dump để kết luận hàm/dòng C++ cụ thể; nearest exports quá xa nên không dùng chúng để quy lỗi cho một hàm.

### Điều có thể / chưa thể kết luận

| Giả thuyết | Bằng chứng / kết luận |
|---|---|
| Chỉ bảng quảng cáo gây crash | Không đúng trong phạm vi trace: biển thật chạy một mình cũng crash dưới low commit; bảng quảng cáo chạy được khi đủ headroom |
| Chỉ call thứ2 / thứ tự0→1 gây crash | Không phải điều kiện cần: crash cả call đầu; cả0→1 và1→0 đều hoàn tất trong loạt đủ tài nguyên |
| Reader dùng chung bị hỏng trạng thái | Chưa có bằng chứng; shared/fresh đều thành công trong loạt đủ tài nguyên; đổi Reader không cứu được call đầu dưới low commit |
| Pixel/tensor bất thường | Không thấy: shape/dtype/plan nhất quán; cùng pixels có thể chạy xong; pixel cap trước vẫn hoạt động |
| Lỗi cấp phát/commit liên quan | Có tương quan lặp lại mạnh: trước conv crash còn~0.24–0.35GB commit nhưng cần buffers cỡ0.52–0.53GB; có free physical RAM vẫn không đủ khả năng commit |
| Native allocator/backend bug cụ thể | Chưa xác định cơ chế C++ gây AV; không đồng nhất mọi AV với MemoryError, không kết luận race condition từ stack Python |

## 4. Ngân sách còn thiếu trong patch trước

Pixel cap2Mi chỉ chặn **một** activation64chfloat32≤512MiB. Nó không chặn normalization buffers, overlapping features, native workspaces, model allocations và cache. Các lượt thành công có peak private commit khoảng3.5–3.6GB dù mỗi activation chỉ~0.53GB; trace CRAFT entry sau models khoảng1.75GB private. Phần tăng thêm vào khoảng1.8–1.9GB. RSS vật lý và private commit là hai metric khác nhau.

Root cause code có thể sửa: `PlateOCR.read_text_details()` trước đây đi thẳng từ pixel plan vào `reader.readtext()` mà không kiểm tra **working-set headroom sau Reader initialization**, rồi exception handler không thể bắt AV trong C++.

## 5. Patch tối thiểu đã áp dụng

- **`src/inference/runtime_memory.py::_available_commit_bytes()` dòng9:** đọc `GlobalMemoryStatusEx.avail_pagefile` trên Windows bằng stdlib ctypes, phân biệt commit với physical RAM. Query lỗi → exception rõ; không tự giả định RAM vô hạn. Platform khác trả None và trace `applied=False`; patch admission này chỉ được kiểm chứng trên Windows CPU.
- **`ocr_memory_admission()` dòng26:** budget dựa profile `4 × full_resolution_64ch_float32_bytes + 128MiB`. Factor4 bao phủ khoảng1.9GB working increment đã đo cho activation~0.53GB;128MiB là reserve thêm. Đây là **policy có cấu hình, có thể bảo thủ**, không phải công thức peak RAM chính xác hoặc reservation của OS.
- **`src/utils/config.py:52–53`:** thêm `OCR_CRAFT_WORKSPACE_FACTOR=4`, `OCR_MEMORY_RESERVE_BYTES=134217728`; không đổi .env, thread policy hoặc pixel cap.
- **`postprocessing.py:345–353`:** sau `reader=self.reader`, đo headroom **cho mỗi OCR call**, kể cả top/bottom/fallback nếu có. Ghi `call.memory_admission` trước quyết định; thiếu headroom thì raise MemoryError trước native, được handler hiện có ghi `ocr_runtime_error`, `format_status=not_completed`, error_stage, error_type và traceback. Không formatter([]), không fake raw text.
- Model/Reader tiếp tục lazy và reuse như cũ; không tạo Reader từng crop. Service/API contract không đổi. Hàm formatter và `_craft_resize_plan` so AST giống nguyên trước task.

Required headroom: A/0 **2153MiB**, A/1 **2112MiB**, B **1830MiB**. [Replay snapshots sáu native entries](historical-admission-replay.json): cả6 đều dưới policy và sẽ bị từ chối. Đây là **counterfactual replay của snapshot**, không giả là6 reruns đã chạy guard dưới đúng cùng trạng thái RAM lịch sử.

## 6. Full production API sau patch — service/Reader dùng lại

Lượt quyết định là [api-persistent-final/trace.json](api-persistent-final/trace.json): khởi tạo Reader bằng property production thật, không LazyReader proxy; A→B→A trong cùng TestClient/service. Chỉ wrapper quan sát detector/details và torch hooks; giữ nguyên returned values và pixels. [Runtime/tracebacks](api-persistent-final/runtime.log), [exit/peak RSS](api-persistent-final/exit.json).

| Request/case | Detection | OCR status | Available commit MiB | Required MiB | Error stage | Request latency s |
|---|---:|---|---:|---:|---|---:|
'''+ '\n'.join(api_rows)+'''

**Full API response A (cả hai lần), HTTP200:**

```json
'''+json.dumps(response_A,ensure_ascii=False,indent=2)+'''
```

Không diễn giải hai chuỗi rỗng public là hai empty OCR: lần A đầu là **unrecognized_format + ocr_runtime_error**, lần A sau là **hai memory_admission runtime errors**. Call bị từ chối không vào CRAFT/recognizer, raw_results không tồn tại và failed call được ghi rõ. Đây là giảm công việc khi thiếu tài nguyên, không phải tăng accuracy.

**B, cùng Reader đã xử lý A, HTTP200:**

```json
'''+json.dumps(final['requests'][1]['response'],ensure_ascii=False,indent=2)+'''
```

B raw `17-AK` confidence0.9058670595759083 và `044.70`0.9998050357717924, boxes/text/confidence khớp task trước; plate đúng không nhờ formatter mới. [Regression checks](regression-checks.json). Full processed pixels mọi detection ở sequence cũng khớp artefact cố định.

**Before/After:** trước có full request A chết process tại CRAFT; sau sequence này exit0 và cả3 HTTP200, vẫn còn error được từ chối an toàn. Không coi request A rỗng là recognition đã sửa. Guard có thể từ chối cả crop vốn đọc được nếu headroom xuống thấp; đây là tradeoff được ghi log.

### Latency và RAM

Đo sequence thật: A đầu12.438s (OCR biển thật, bỏ call bảng vì budget), B6.875s (warm Reader), A sau0.468s (cả hai bị từ chối). Không gọi đây là speedup vì số OCR calls khác nhau. Peak RSS sequence~2276MiB, peak private commit~2916MiB; [OS metrics](api-persistent-final/exit.json). Cùng sequence0→1 mặc định khi đủ tài nguyên trước patch mất21.984s gồm process/model startup, peak RSS~2250MiB và private~3464MiB; không so trực tiếp như cùng workload/transport.

Các thử nghiệm CPU1 có shared pair khoảng46.6s, fresh pair50–51.8s so với mặc định healthy~22s/process. Không áp dụng CPU1 hay đổi Reader production từ kết quả bị confound này.

### Những lượt harness phụ

`api-A-after-1`, `api-B-after`, `api-A-after-2` dùng legacy observer LazyReader, đều exit0. Memory snapshot trước lazy model initialization không hoàn toàn giống property production, nên không lấy chúng làm chứng cứ admission quyết định. `api-persistent-A-B-A` do dispatch harness ban đầu vẫn chạy matrix crop dù tag trông như API; metadata api_case=null và trace xác định rõ, **không tính là API response**. Đã sửa dispatch và chạy `api-persistent-final` đúng production property/service như trên. Không xóa các lượt phụ hoặc gán nhầm kết quả của chúng.

## 7. Đánh giá worker isolation và giới hạn còn lại

**Guard không bảo đảm API sống trước mọi native fault.** Headroom không được reserve; process khác có thể cấp phát sau snapshot, workspace thực tế có thể vượt estimate, native bug độc lập bộ nhớ vẫn có thể xảy ra. Python except không bắt AV. Cần worker-process isolation nếu yêu cầu là bảo đảm boundary của FastAPI trước mọi worker death.

Chưa triển khai worker architecture trong patch này vì policy nhỏ đã chặn đường thiếu tài nguyên quan sát được và full A/B regression đã hoàn tất. Đồng thời, thêm một Torch/OCR process ở môi trường hiện tại làm tăng commit: riêng import/model runtime ở worker có khoảng1.36GB private baseline, parent YOLO giữ~1.5GB. Cộng worker peak3.6GB có thể đẩy tổng lên~5.1GB, làm áp lực commit nặng hơn; đây là **ước lượng từ profile**, chưa benchmark kiến trúc worker thật.

Phương án nếu cần hard isolation: **một** persistent worker, queue hữu hạn, một Reader lazy tái sử dụng; process memory limit khoảng4GiB qua Windows Job Object và timeout90s/crop là budget khởi điểm cần kiểm chứng. Worker exit/timeout → `ocr_runtime_error` với exit code/faulthandler, không empty OCR; bỏ crop lỗi, respawn cho crop tiếp theo, không tự retry vô hạn crop gây crash; cleanup chỉ tiến trình do service tạo. Nên cân nhắc đặt cả YOLO và OCR trong cùng inference worker để tránh hai bản Torch trong RAM, nhưng đó là thay đổi lớn hơn transport hiện tại. Public schema có thể giữ nguyên, debug cần worker status riêng.

Không có số đo latency worker architecture chưa triển khai. Profile hiện tại cho thấy process/model startup tốn vài giây và model residency thêm trên1GB commit; cần đánh giá thực tế trước khi chọn hard limits. Không giả lập rằng guard hiện tại đã cung cấp sự cô lập này.

## 8. Tests, files và rollback

**'''+test_result+'''** bằng `python -B -m pytest tests -q -p no:cacheprovider`, có inference smoke. [Log](pytest.log). Test mới kiểm tra headroom boundary, query failure, từ chối trước native, status/traceback, còn tiếp tục detection sau MemoryError, HTTP200/schema cũ và result detection sau. Fake Reader tests có snapshot memory giả để test không phụ thuộc tải máy; unit refusal không được gọi là native-crash simulation.

Production thay đổi: thêm runtime_memory.py, sửa postprocessing.py và config.py. Tests bổ sung trong3 test files. Không sửa service/detector/preprocessing/formatter/weights/dataset/labels/dependencies. [Integrity](integrity.json), [unified patch](patch.diff), snapshots trước task tại `before/`. Tool mới `scripts/debug_sequential_ocr.py`, `scripts/report_sequential_ocr.py`, và `scripts/inspect_c10_exports.py` (chỉ đọc PE exports, không cung cấp native line symbol đáng tin cậy).

```powershell
Set-Location D:\\CCNLTHD
.\\.venv\\Scripts\\python.exe -B -m pytest tests -q -p no:cacheprovider
# Dùng tag mới; A/B/A cùng service thực, current guard bật:
.\\.venv\\Scripts\\python.exe -B scripts/debug_sequential_ocr.py launch --tag api-review-repeat --case sequence
```

Các matrix before là artefact trước patch; chạy lại script hiện tại sẽ có guard nên không còn cùng code baseline. Không tắt guard trong API đang phục vụ chỉ để tái tạo crash. Không đổi/tắt tiến trình người dùng hoặc điều chỉnh pagefile/system environment trong task này.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.md'),'control_rows':len(rows),'native_entries_refused':len(replays),
                      'B_exact_match':True,'test_result':test_result}))


if __name__=='__main__':main()
