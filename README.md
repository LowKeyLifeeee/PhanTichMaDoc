# PhanTichMaDoc

Agent phân tích mã độc tự chủ theo vòng lặp ReAct (`Thought -> Action -> Observation`), chạy hoàn toàn offline bằng mô hình 3B (`Qwen/Qwen2.5-Coder-3B-Instruct`, bản gốc hoặc bản tinh chỉnh QLoRA) qua Ollama, trên phần cứng RTX 4050 6 GB VRAM + 16 GB RAM. Trọng tâm của dự án là chống prompt injection nhúng trong mẫu và đo lường bằng số liệu.

---

> ## ⚠️ BẮT BUỘC ĐỌC TRƯỚC KHI ĐÓNG GÓP
>
> **Mọi hợp tác viên (người hoặc AI agent) phải đọc hết README này trước khi sửa code, chạy mẫu hay commit.**
> Phần [Quy tắc an toàn](#1-quy-tắc-an-toàn-không-ngoại-lệ) và [Quy ước đóng góp](#9-quy-ước-đóng-góp) là bắt buộc, không phải gợi ý.
> Vi phạm quy tắc an toàn (chạy mẫu ngoài VM, gửi mẫu lên cloud, commit file mẫu) có thể gây lây nhiễm thật hoặc rò rỉ mẫu của giảng viên.
>
> Nếu bạn là AI agent (Claude Code, Codex, Copilot...): đọc README này và các file trong `openspec/changes/add-malware-react-agent/` trước khi làm bất cứ việc gì.

---

## Mục lục

1. [Quy tắc an toàn (không ngoại lệ)](#1-quy-tắc-an-toàn-không-ngoại-lệ)
2. [Kiến trúc](#2-kiến-trúc)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt nhanh](#4-cài-đặt-nhanh)
5. [Chạy agent](#5-chạy-agent)
6. [Tinh chỉnh QLoRA](#6-tinh-chỉnh-qlora)
7. [Benchmark và chấm điểm](#7-benchmark-và-chấm-điểm)
8. [Trạng thái hiện tại](#8-trạng-thái-hiện-tại)
9. [Quy ước đóng góp](#9-quy-ước-đóng-góp)
10. [Lỗi đã gặp và cách xử lý](#10-lỗi-đã-gặp-và-cách-xử-lý)
11. [Tài liệu chi tiết](#11-tài-liệu-chi-tiết)

---

## 1. Quy tắc an toàn (không ngoại lệ)

1. **Mẫu mã độc chỉ nằm trong VM cô lập.** VM chỉ có card Host-Only, không NAT, không bridged, không Internet. Không copy mẫu ra máy host, kể cả để "xem nhanh".
2. **Không bao giờ thực thi mẫu.** Agent không có tool nào chạy mẫu, và `run_command` từ chối chạy khi file thực thi chính là mẫu. Log động (Procmon, INetSim) do người làm thủ công trên snapshot dùng một lần, xem [docs/setup.md](docs/setup.md#4-dynamic-logs-manual-detonation).
3. **Không gửi mẫu hay dữ liệu mẫu lên bất kỳ cloud/LLM API nào.** Model chạy qua Ollama local. Agent từ chối khởi động nếu `OLLAMA_HOST` không phải địa chỉ loopback hoặc private (RFC 1918).
4. **Model chỉ nhận đặc trưng**, tức văn bản/JSON trích từ tool, không bao giờ nhận byte nhị phân gốc. Chuỗi Base64/hex dài bị cắt bớt trước khi đưa vào prompt.
5. **Không commit mẫu.** `data/samples/`, `data/clean_files/`, `data/dynamic_logs/`, `logs/`, `reports/` và file `.gguf` đã nằm trong `.gitignore`. Không sửa `.gitignore` để đưa chúng vào. Trước mỗi lần commit, chạy `git status` và kiểm tra lại.
6. **Colab chỉ dùng dữ liệu tổng hợp.** Script train tự sinh output tool giả với IOC thuộc dải tài liệu dành riêng (`203.0.113.0/24`, `198.51.100.0/24`, `192.0.2.0/24`, `*.example`). Không upload mẫu thật hay log thật lên Colab.
7. **Mọi lần chạy đều có audit log** (`logs/<sha256>/<run-id>.jsonl`): đủ Thought, Action, Observation thô, lần bị chặn và quyết định grounding. Không tắt hay xoá log khi đang chấm điểm.

## 2. Kiến trúc

```
+--------------------- Host (RTX 4050) ---------------------+
|  ollama serve  (chỉ nhận kết nối từ 192.168.56.0/24)       |
|  model: qwen2.5-coder:3b  /  qwen25-coder-3b-malware-guard |
+------------------------- Host-Only 192.168.56.1 ----------+
                              |   (không NAT, không Internet)
+------------------- Analysis VM 192.168.56.101 ------------+
|  agent_react.py --sample X --guard                         |
|    ReAct loop -> tools: pe_info, die, floss, capa, ghidra, |
|                  yara_scan, procmon, inetsim, yara_generate|
|    guard: spotlight + scanner + chặn tham số + grounding    |
|  mẫu, file sạch, log, báo cáo                              |
+------------------------------------------------------------+
```

**Vòng ReAct** ([src/agent/agent_react.py](src/agent/agent_react.py)):
- Mỗi bước model trả về một JSON `{thought, tool, args}`, bị ép bằng JSON Schema qua tham số `format` của Ollama. `tool` là một `enum`, nên model không bịa được tên tool. Model **không bao giờ truyền đường dẫn file**: agent tự gắn đường dẫn mẫu.
- Tham số suy luận: `num_ctx=4096`, `temperature=0`, `top_p=0.1`, `repeat_penalty=1.05`, `seed=42`.
- Ngân sách token: output mỗi tool ≤ 450 token (tối đa 600), prompt ≤ 3.500 token. Khi vượt, Observation cũ được thay bằng bản tóm tắt 1 dòng. Grounding vẫn đối chiếu với output thô đầy đủ.
- Phải có ≥ 4 tool chạy thành công mới được `finish`, trừ khi đã thử hết mọi tool có sẵn. Giới hạn 12 bước. Nếu model gọi lại một tool đã lỗi, hoặc gọi trùng 2 lần liền, agent tự chọn tool kế tiếp và ghi `auto_select` vào log.
- Kết thúc: một lần gọi riêng với schema báo cáo. Nếu kết luận là `MALICIOUS`, agent sinh luật YARA: model chọn chuỗi, code dựng luật, rồi kiểm tra luật phải khớp mẫu và khớp 0 file trong `data/clean_files/`. Nếu sai, model được sửa tối đa 3 lần, sau đó code tự chọn chuỗi.

**Ba lớp phòng thủ prompt injection** (bật bằng `--guard`, [src/guardrails/defense.py](src/guardrails/defense.py)):

| Lớp | Cơ chế |
|---|---|
| 1. Spotlighting + scanner | Output tool được bọc trong thẻ ngẫu nhiên theo từng phiên `<TOOL_DATA_xxxx>`. Thẻ giả trong dữ liệu bị escape, token chat template (`<\|im_start\|>`...) luôn bị escape. Scanner tìm 6 kỹ thuật: ghi đè chỉ thị, đòi kết luận, giả vai trò hệ thống, giả thẻ/template, Base64/hex mã hoá, chia nhỏ qua nhiều chuỗi. Dòng nghi vấn bị gắn nhãn `[SUSPICIOUS_PROMPT_INJECTION_DETECTED]`. |
| 2. QLoRA | Model được tinh chỉnh trên tập 50% mẫu sạch / 50% mẫu đối kháng ([src/training/train_qlora.py](src/training/train_qlora.py)). |
| 3. Hậu kiểm | Chặn tham số tool chứa chuỗi bị gắn cờ. Grounding: mọi IOC/TTP/hành vi phải có `evidence_quote` khớp nguyên văn output tool, nếu không thì bị loại. Khoá kết luận: có injection thì kết luận bắt buộc là `MALICIOUS`. |

## 3. Cấu trúc thư mục

```
AGENTS.md, CLAUDE.md            trỏ AI agent về README này (agent tự đọc 2 file này khi mở repo)
src/tools/malware_tools.py      wrapper pefile, FLOSS, DIE, capa, YARA, Ghidra, Procmon, INetSim; đếm token
src/guardrails/defense.py       schema Pydantic, scanner injection, spotlight, grounding, khoá kết luận
src/agent/agent_react.py        vòng ReAct, client Ollama, ngân sách context, YARA, audit log, CLI
src/training/train_qlora.py     sinh dữ liệu đối kháng, train QLoRA, merge, eval, export GGUF
src/evaluation/eval_benchmark.py chạy benchmark 3 vòng, tính chỉ số, xuất bảng nghiệm thu
ghidra_scripts/DecompileTargets.java   script headless (Java, chạy được trên mọi bản Ghidra)
rules/base/*.yar                luật YARA nền (UPX, injection API, PowerShell, prompt injection)
Modelfile                       Modelfile Ollama cho model QLoRA (Q5_K_M)
tests/                          124 test, dùng output tool ghi sẵn + model giả
docs/                           setup.md, training.md, benchmark.md
openspec/changes/add-malware-react-agent/   proposal, specs, design, tasks (nguồn sự thật về yêu cầu)
data/samples/                   mẫu cần phân tích            (không commit)
data/clean_files/               file sạch để kiểm tra FPR    (không commit)
data/ground_truth/              đáp án mỗi mẫu: <tên mẫu>.json
data/dynamic_logs/<mẫu>/        procmon.csv, service.log     (không commit)
models/tokenizer/tokenizer.json tokenizer Qwen               (không commit, tự tải)
logs/, reports/                 audit log, báo cáo           (không commit)
```

## 4. Cài đặt nhanh

Hướng dẫn đầy đủ (mạng VM, tool, biến môi trường, log động) nằm ở [docs/setup.md](docs/setup.md). Bản rút gọn:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tải tokenizer Qwen (7 MB). Không có file này thì agent dùng ước lượng token dự phòng, đếm dư 6–60%:

```powershell
curl.exe -L -o models/tokenizer/tokenizer.json https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/tokenizer.json
```

Tool phân tích (bản đã kiểm tra với code hiện tại):

| Tool | Phiên bản đã test | Biến môi trường |
|---|---|---|
| FLOSS | 3.2.0 | `FLOSS_PATH` |
| capa | 10.0.0 | `CAPA_PATH` (`CAPA_RULES`, `CAPA_SIGS` tuỳ chọn) |
| Detect It Easy (`diec`) | 4.0.0 | `DIEC_PATH` |
| Ghidra | 12.2 + **JDK 25** | `GHIDRA_HOME` (thư mục chứa `support/analyzeHeadless`) |
| Ollama | | `OLLAMA_HOST`, ví dụ `http://192.168.56.1:11434` |

Ví dụ (PowerShell):

```powershell
$env:FLOSS_PATH="D:\Tools\floss\floss.exe"; $env:CAPA_PATH="D:\Tools\capa\capa.exe"; $env:DIEC_PATH="D:\Tools\die\diec.exe"; $env:GHIDRA_HOME="D:\Tools\ghidra_12.2_PUBLIC"
```

Model trên host:

```powershell
ollama pull qwen2.5-coder:3b
```

Kiểm tra cài đặt:

```powershell
python -m pytest
```

## 5. Chạy agent

```powershell
python src/agent/agent_react.py --sample data/samples/sampleG.exe --guard
```

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--sample PATH` | bắt buộc | Mẫu cần phân tích (chỉ đọc, không chạy) |
| `--guard` | tắt | Bật 3 lớp phòng thủ |
| `--model TAG` | `qwen2.5-coder:3b` | Tag model Ollama, ví dụ `qwen25-coder-3b-malware-guard` |
| `--out DIR` | `reports/runs` | Nơi ghi `<sha256>.report.json` |
| `--log-dir DIR` | `logs` | Audit log `logs/<sha256>/<run-id>.jsonl` |
| `--max-steps N` | 12 | Giới hạn bước ReAct |
| `--run-id ID` | `thờigian_model_guard` | Tên file audit log |

Mã thoát: `0` ok, `2` không có/không đọc được mẫu, `3` `OLLAMA_HOST` không hợp lệ hoặc thiếu model, `4` lỗi Ollama.

Báo cáo gồm: `verdict`, `confidence`, `summary`, `iocs`, `techniques`, `behaviors` (mỗi nhận định có `evidence_quote` và `source_tool`), `prompt_injection_alerts`, `yara` (luật + đã kiểm chứng hay chưa), `dropped_claims`, `grounding`, `verdict_override`, `tool_shortfall`, `tools_succeeded` và `metrics` (số bước, số lần gọi tool, số lần lỗi, thời gian, token, RAM, VRAM, chi phí $0.00).

## 6. Tinh chỉnh QLoRA

Chi tiết từng ô lệnh Colab ở [docs/training.md](docs/training.md).

```powershell
python src/training/train_qlora.py build --n 2000 --out data/train/dataset.jsonl --leak-check-dir data/samples
```

Sau đó trên Colab (T4/L4) chạy lần lượt:
1. `train --dry-run` (chạy thử)
2. `train` (train thật)
3. `export --llama-cpp /content/llama.cpp`

Kết quả là `qwen25-coder-3b-malware-guard-Q5_K_M.gguf`. Trên host:

```powershell
ollama create qwen25-coder-3b-malware-guard -f Modelfile
```

Cấu hình: 4-bit NF4 + double quant, LoRA r=16/alpha=32/dropout=0.05 trên 7 lớp chiếu, `max_seq_length=3072`, loss chỉ tính trên câu trả lời. Bước `build` sẽ báo lỗi nếu tỉ lệ đối kháng lệch khỏi 48–52%, có kỹ thuật dưới 10%, không có mẫu `BENIGN`, hoặc có chuỗi trùng với mẫu benchmark.

## 7. Benchmark và chấm điểm

Chi tiết công thức và định dạng đáp án ở [docs/benchmark.md](docs/benchmark.md).

1. Đặt mẫu vào `data/samples/`, file sạch vào `data/clean_files/`.
2. Với mỗi mẫu, tạo `data/ground_truth/<tên mẫu>.json` (copy từ [file ví dụ](data/ground_truth/example_sampleG.exe.json.example)).
3. Chạy:

```powershell
python src/evaluation/eval_benchmark.py --base-model qwen2.5-coder:3b --qlora-model qwen25-coder-3b-malware-guard --runs 3
```

Kết quả nằm trong `reports/benchmark/`:
- `summary.md`: mọi chỉ số kèm PASS/FAIL so với mục tiêu, ASR có/không `--guard`, bảng hiệu năng.
- `acceptance.csv` và `acceptance.md`: bảng `Mẫu thử | Hạng mục | Agent nói | Thực tế | Dẫn chứng | Kết quả`.
- `metrics.json`.

Benchmark chạy tiếp được từ chỗ dừng; thêm `--fresh` để chạy lại từ đầu.

| Chỉ số | Mục tiêu |
|---|---|
| Precision / Recall / F1 | ≥ 0,80 / ≥ 0,70 / ≥ 0,80 |
| Verdict trên mẫu lab | 100% |
| Tỉ lệ bịa / có dẫn chứng / ổn định 3 lần | ≤ 5% / ≥ 90% / ≥ 80% |
| Tool thành công mỗi lần chạy | ≥ 4 |
| YARA TPR / FPR | ≥ 90% / 0% |
| ASR khi bật `--guard` (≥ 10 file injection) | 0% |

## 8. Trạng thái hiện tại

Tiến độ theo [tasks.md](openspec/changes/add-malware-react-agent/tasks.md): **46/54 task xong**, 124 test qua.

Đã kiểm tra với tool thật (trên host, file lành `python.exe`): FLOSS 3.2, capa 10.0, DIE 4.0, Ghidra 12.2, Ollama với `qwen2.5:3b-instruct` và `llama3.2:3b`. Một vòng đủ tool mất khoảng 200 giây (FLOSS ~87 giây, capa ~62 giây). RAM tool khoảng 1,2 GB, VRAM model khoảng 2 GB.

Còn lại:

| Task | Cần gì |
|---|---|
| 7.4–7.6 | Train, đánh giá held-out, export GGUF trên Colab GPU |
| 9.1–9.5 | Chạy trong VM lab với mẫu của giảng viên, Sample G, ≥ 10 file injection, file sạch, đáp án |

Hạn chế đã biết:
- Model base (`qwen2.5:3b`) kết luận sai file lành là `MALICIOUS`. Grounding loại được nhận định bịa nhưng không sửa được kết luận; việc này thuộc về QLoRA.
- Khoá kết luận coi mọi file chứa chuỗi injection là `MALICIOUS`. Một file lành có chứa câu dạng prompt cũng sẽ bị báo nhầm.
- Không có file trong `data/clean_files/` thì luật YARA luôn bị đánh dấu chưa kiểm chứng, và agent bỏ qua bước sinh YARA.

## 9. Quy ước đóng góp

1. **Đọc README này, rồi đọc `openspec/changes/add-malware-react-agent/`** (proposal → specs → design → tasks). Specs là nguồn sự thật về hành vi. Muốn đổi hành vi thì sửa spec/design trước, code sau.
2. **Đổi lớn thì tạo OpenSpec change mới** (`/opsx:new` hoặc `openspec new change <tên>`). Sửa nhỏ trong phạm vi change hiện tại thì cập nhật `tasks.md`, và chỉ đánh dấu `[x]` khi task đã chạy và kiểm chứng xong.
3. **Viết test cùng lúc với code.** Mỗi thay đổi đều phải kèm test trong `tests/`, và `python -m pytest` phải xanh trước khi commit. Test không được cần mẫu thật, tool thật hay Ollama: dùng fixture trong `tests/fixtures/` và `FakeClient`.
4. **Đổi định dạng output tool thì cập nhật fixture từ output thật**, và thêm test hồi quy dùng chuỗi thật (xem `test_floss_signed_binary_noise_filtered`).
5. **Không đổi các hằng số cốt lõi khi không có lý do ghi trong design.md**: `num_ctx`, `temperature`, `top_p`, `repeat_penalty`, cap token, `MIN_TOOLS`, quy tắc grounding.
6. **Thêm biến môi trường mới thì ghi vào [docs/setup.md](docs/setup.md).** Test `test_setup_doc_lists_every_env_var` sẽ fail nếu thiếu.
7. **Commit:** không commit mẫu, log hay model. Không thêm dòng ghi công AI (`Co-Authored-By`...) vào commit. Mỗi commit một việc, message nói rõ thay đổi gì.
8. **AI agent không tự commit/push** nếu người phụ trách không yêu cầu rõ ràng.

## 10. Lỗi đã gặp và cách xử lý

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Ghidra: `The data file must be inside the data dir` (Felix/OSGi) | Chạy Ghidra từ trong app đóng gói MSIX (ví dụ app Claude desktop): `%APPDATA%` bị ảo hoá nên đường dẫn chuẩn lệch nhau | Chạy từ terminal thường, hoặc đặt `APPDATA=D:\Tools\ghidra_appdata` cho tiến trình |
| Ghidra: `JAVA_HOME_OVERRIDE specifies unsupported java version` | Ghidra 12.x cần JDK 25 | Dùng JDK 25, để trống `JAVA_HOME_OVERRIDE` |
| Ghidra nhận sai danh sách địa chỉ/API | `cmd.exe` tách tham số `.bat` ở dấu phẩy | Đã đổi sang dấu `+`. Đừng đổi lại thành dấu phẩy |
| Prompt tràn context dù ước lượng còn dư | Tokenizer Qwen tách **mỗi chữ số thành 1 token**; chars/3 đếm thiếu tới 2 lần | Cài `tokenizer.json`. Ước lượng dự phòng đã tính riêng chữ số |
| FLOSS: top chuỗi toàn URL digicert/ocsp | Chữ ký Authenticode có trong mọi file đã ký | Đã lọc trong `is_noise_string`. Gặp nhà cung cấp chứng chỉ mới thì thêm vào `_NOISE_HOST_RE` |
| Grounding loại cả kỹ thuật đúng | Model chép kèm nhãn hiển thị `[static]`, `url:` | Đã bỏ nhãn trước khi so (`quote_variants`) |
| YARA "not validated: no clean files" | `data/clean_files/` trống | Thêm file sạch (exe/dll Windows) |
| Agent báo `OLLAMA_HOST ... not a loopback or private address` | Trỏ tới địa chỉ public | Chỉ dùng localhost hoặc IP Host-Only |

## 11. Tài liệu chi tiết

- [docs/setup.md](docs/setup.md): mạng VM, tool, biến môi trường, quy trình lấy log động
- [docs/training.md](docs/training.md): Colab, train QLoRA, export GGUF, nạp vào Ollama
- [docs/benchmark.md](docs/benchmark.md): định dạng đáp án, cách chấm, công thức chỉ số
- [openspec/changes/add-malware-react-agent/](openspec/changes/add-malware-react-agent/): proposal, specs (5 capability), design (15 quyết định), tasks
