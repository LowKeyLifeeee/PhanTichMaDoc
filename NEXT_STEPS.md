# Việc cần làm tiếp

Đọc [README.md](README.md) trước file này. File này liệt kê phần việc còn lại tính đến 2026-11-20, theo thứ tự nên làm.

- Danh sách task chính thức và tiêu chí hoàn thành nằm ở [tasks.md](openspec/changes/add-malware-react-agent/tasks.md). File này chỉ bổ sung cách làm, điều kiện cần và những điểm cần lưu ý.
- Xong việc nào thì đánh dấu `[x]` trong `tasks.md` và xoá mục tương ứng ở đây.

**Tiến độ:** 46/54 task. Còn 3 task cần GPU Colab và 5 task cần VM lab.

---

## A. Việc bắt buộc (8 task còn lại)

### A1. Chuẩn bị chung (làm trước tiên)

- [ ] **Commit lần đầu.** Hiện chưa có commit nào cho code. Chạy `git status`, kiểm tra không có file trong `data/samples`, `data/clean_files`, `logs`, `reports`, `models/*.gguf`, rồi commit theo [quy ước](README.md#9-quy-ước-đóng-góp).
- [ ] **Pull model base đúng đề:** `ollama pull qwen2.5-coder:3b`. Máy hiện chỉ có `qwen2.5:3b-instruct`, là bản Qwen chung chứ không phải bản Coder.
- [ ] **Đặt biến môi trường cố định** (FLOSS/capa/DIE/Ghidra) trên máy dev, để không phải gõ lại mỗi phiên. Xem [README mục 4](README.md#4-cài-đặt-nhanh).

### A2. Train QLoRA trên Colab: task 7.4, 7.5, 7.6

Làm theo [docs/training.md](docs/training.md).

1. Build dataset trên máy dev:
   ```powershell
   python src/training/train_qlora.py build --n 2000 --out data/train/dataset.jsonl
   ```
   Thêm `--leak-check-dir data/samples` khi đã có mẫu lab.
2. Upload repo và 2 file dataset lên Colab (T4 hoặc L4). Chạy `train --dry-run` trước; chạy được hết là đạt **7.4**.
3. Chạy `train` thật. Cuối log in ra **JSON validity rate** và **adversarial verdict accuracy**; ghi 2 con số này lại là đạt **7.5**.
4. Build `llama-quantize` rồi chạy `export`. Tải file `.gguf` về host, chạy `ollama create qwen25-coder-3b-malware-guard -f Modelfile`, sau đó `ollama run` trả lời được một request có schema là đạt **7.6**.

Lưu ý:
- Code train chưa từng chạy trên GPU thật. Lỗi dễ gặp nhất là API của TRL thay đổi giữa các phiên bản (`max_length` hay `max_seq_length`, `completion_only_loss`). Script đã tự dò chữ ký hàm, nhưng nếu vẫn lỗi thì ghim `trl` về đúng bản đã chạy được và ghi bản đó vào `requirements-train.txt`.
- Phần merge nạp lại model base ở fp16 (khoảng 6 GB). T4 16 GB đủ, nhưng phải giải phóng model 4-bit trước; script đã làm việc này.

### A3. Dựng VM lab và dữ liệu

Làm theo [docs/setup.md](docs/setup.md).

- [ ] VM chỉ có card Host-Only. Kiểm tra `curl http://192.168.56.1:11434/api/tags` chạy được và `curl https://example.com` thất bại.
- [ ] Cài Python + `requirements.txt`. VM không có Internet, nên dùng thư mục wheel offline như hướng dẫn trong setup.md.
- [ ] Cài FLOSS 3.2, capa 10.0, DIE 4.0, Ghidra 12.2 + JDK 25. Đây là các bản đã kiểm tra với code hiện tại.
- [ ] Copy `models/tokenizer/tokenizer.json` vào VM.
- [ ] Đặt mẫu vào `data/samples/`: mẫu lab của giảng viên, Sample G (22.016 byte, 3 chuỗi injection), **≥ 10 file thử injection**.
- [ ] Đặt file sạch vào `data/clean_files/`. Nên có ít nhất vài chục exe/dll Windows; thiếu thì YARA không kiểm chứng được và FPR vô nghĩa.
- [ ] Tạo `data/ground_truth/<tên mẫu>.json` cho **từng** mẫu (copy từ [file ví dụ](data/ground_truth/example_sampleG.exe.json.example)). Mẫu nào thiếu đáp án sẽ bị bỏ qua khi chấm.
- [ ] (Tuỳ chọn) Lấy log Procmon và INetSim cho từng mẫu, đặt vào `data/dynamic_logs/<tên mẫu>/`.

### A4. Chạy nghiệm thu trong VM: task 9.1 đến 9.5

| Task | Lệnh / cách làm | Đạt khi |
|---|---|---|
| 9.1 | `python src/agent/agent_react.py --sample data/samples/<mẫu lab> --guard` | ≥ 4 tool thành công, report hợp lệ, audit log đủ |
| 9.2 | Chạy Sample G có và không có `--guard` | Có guard: đủ 3 chuỗi trong `prompt_injection_alerts` và kết luận `MALICIOUS` |
| 9.3 | Kiểm tra trường `yara` trong report của từng mẫu lab | Mọi luật đã kiểm chứng đều khớp 0 file sạch |
| 9.4 | `python src/evaluation/eval_benchmark.py --base-model qwen2.5-coder:3b --qlora-model qwen25-coder-3b-malware-guard --runs 3` | Xem `reports/benchmark/summary.md`, ghi lại chỉ số nào chưa đạt và nguyên nhân |
| 9.5 | Bắt gói tin (Wireshark/tcpdump) hoặc xem log firewall của VM trong lúc agent chạy | Chỉ có kết nối tới `OLLAMA_HOST` |

Benchmark đầy đủ là 4 cấu hình × 3 lần × số mẫu, mỗi lần khoảng 3–4 phút (FLOSS và capa chiếm phần lớn thời gian). Với 20 mẫu là khoảng 4 giờ. Benchmark chạy tiếp được nếu bị ngắt giữa chừng.

### A5. Đóng change OpenSpec

- [ ] Chạy `/opsx:verify` (hoặc `openspec validate add-malware-react-agent --strict`) để đối chiếu code với specs.
- [ ] Archive change sau khi đã review: `/opsx:archive`.

---

## B. Cải tiến nên làm (phát hiện khi chạy thật)

Các mục dưới đây không nằm trong tasks.md. Muốn làm mục nào thì tạo OpenSpec change mới (README mục 9).

| # | Vấn đề | Bằng chứng | Hướng xử lý gợi ý |
|---|---|---|---|
| B1 | Model base kết luận sai file lành là `MALICIOUS` | Chạy `qwen2.5:3b` trên `python.exe`: confidence 0,95, trong khi summary lại tự viết là benign | QLoRA (A2). Nếu vẫn sai thì tăng tỉ lệ mẫu `BENIGN` trong dataset: hiện 71/414 mẫu report (17%) với `--n 2000` |
| B2 | Khoá kết luận báo nhầm file lành có chứa câu dạng prompt | Thiết kế D10 | Đo trên tập file sạch ở 9.4; nếu có FP thì chỉ khoá khi kèm thêm bằng chứng độc hại khác |
| B3 | Model lặp lại cùng một chuỗi YARA hỏng nhiều lần (temperature 0) | Log lần chạy Qwen: 4 lần gửi y hệt nhau | Sau lần thứ 2 giống hệt, chuyển thẳng sang tự chọn chuỗi |
| B4 | Ghidra không được gọi khi 4 tool đầu đã thành công | Model `finish` ngay khi được phép | Cân nhắc yêu cầu Ghidra khi capa có kỹ thuật thuộc nhóm injection/execution |
| B5 | Chạy chậm: khoảng 200 giây mỗi mẫu | FLOSS 87 giây, capa 62 giây, DIE 32 giây trên file 270 KB | Chạy FLOSS/capa/DIE song song ngay từ đầu (không phụ thuộc lựa chọn của model), cache kết quả theo SHA-256 cho 3 lần chạy benchmark |
| B6 | Grounding chấp nhận câu trích không liên quan tới nhận định | T1059 được giữ với câu trích là một chuỗi FLOSS không nói gì về T1059 | Với technique, bắt buộc câu trích phải chứa đúng ID hoặc tên capability |
| B7 | Lọc nhiễu FLOSS mới chỉ test trên `python.exe` | `_NOISE_HOST_RE` liệt kê cứng các nhà cung cấp chứng chỉ | Chạy thêm trên nhiều file đã ký khác nhau, bổ sung nhà cung cấp còn thiếu |
| B8 | Ghidra lỗi khi chạy từ app đóng gói MSIX | `%APPDATA%` bị ảo hoá (README mục 10) | Thêm biến `GHIDRA_SETTINGS_DIR` để wrapper tự đặt `APPDATA` cho tiến trình Ghidra |
