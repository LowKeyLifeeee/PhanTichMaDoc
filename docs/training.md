# QLoRA training on Google Colab

`src/training/train_qlora.py` builds a synthetic instruction-hierarchy dataset, fine-tunes
`Qwen/Qwen2.5-Coder-3B-Instruct` with 4-bit QLoRA, merges the adapter, checks it on a held-out split and
exports a `Q5_K_M` GGUF for Ollama.

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization, compute dtype bfloat16 (L4/A100) or float16 (T4) |
| LoRA | r=16, alpha=32, dropout=0.05 on q/k/v/o/gate/up/down projections |
| Sequence length | 3072 tokens, loss on the assistant completion only |
| Defaults | 2 epochs, lr 2e-4, batch 1 x grad-accum 16, cosine schedule |

The dataset is synthetic: tool outputs are generated and formatted by the same code the agent uses,
IOCs come from reserved ranges (`203.0.113.0/24`, `198.51.100.0/24`, `192.0.2.0/24`, `*.example`), and no
real sample is needed on Colab. Half the examples carry one of 6 injection techniques (direct override,
fake verdict demand, role impersonation, fake delimiter / chat-template tags, Base64/hex encoded, split
across strings) inside FLOSS, Ghidra or Procmon output. Targets keep the true verdict, list only grounded
IOCs/TTPs and put the injected strings in `prompt_injection_alerts`.

## 1. Build the dataset (any machine)

```
python src/training/train_qlora.py build --n 2000 --out data/train/dataset.jsonl --leak-check-dir data/samples
```

It prints the clean/adversarial split and the per-technique share, and fails if the split is outside
48-52%, a technique is below 10%, there are no BENIGN targets, or any generated IOC/injection string also
occurs in a benchmark sample under `--leak-check-dir`. Output: `dataset.jsonl` (train) and
`dataset.eval.jsonl` (5% held out).

## 2. Train on Colab

1. Runtime -> Change runtime type -> **T4 GPU** (or L4).
2. Upload the repo (or `git clone` it) and the two dataset files to `/content/PhanTichMaDoc/data/train/`.
3. Run these cells:

```
%cd /content/PhanTichMaDoc
!pip install -q -r requirements-train.txt pydantic psutil requests
!python src/training/train_qlora.py train --data data/train/dataset.jsonl --out outputs/qlora --dry-run
```

The dry run (20 examples, 2 steps, 4 held-out examples) checks the whole pipeline in a few minutes.
Then run the real training:

```
!python src/training/train_qlora.py train --data data/train/dataset.jsonl --out outputs/qlora
```

At the end it prints the held-out **JSON validity rate** and **adversarial verdict accuracy**
(also saved to `outputs/qlora/eval_metrics.json`). Outputs: `outputs/qlora/adapter/` (LoRA weights)
and `outputs/qlora/merged/` (adapter merged into fp16 base weights; the merge reloads the base model in
fp16 because merging into 4-bit weights loses precision).

## 3. Convert to GGUF Q5_K_M

```
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
!pip install -q -r /content/llama.cpp/requirements.txt
!cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=OFF && cmake --build /content/llama.cpp/build --target llama-quantize -j 4
!python src/training/train_qlora.py export --merged outputs/qlora/merged --out outputs/export --llama-cpp /content/llama.cpp
```

This runs `convert_hf_to_gguf.py` (f16) and `llama-quantize ... Q5_K_M`, and writes `outputs/export/Modelfile`
(same as the repo's `Modelfile`). Add `--print-only` to see the commands without running them.

Download `outputs/export/qwen25-coder-3b-malware-guard-Q5_K_M.gguf` (about 2.2 GB) and the `Modelfile`.

## 4. Load into Ollama (host)

Put both files in one folder on the host, then:

```
ollama create qwen25-coder-3b-malware-guard -f Modelfile
ollama run qwen25-coder-3b-malware-guard "Reply with {\"ok\": true}"
```

Use it from the VM with `--model qwen25-coder-3b-malware-guard`. The base model for comparison is
`ollama pull qwen2.5-coder:3b`.
