# Lab setup

The agent and every analysis tool run **inside an isolated analysis VM** (Host-Only network, no NAT, no internet). Only Ollama runs on the host, so the 3B model uses the RTX 4050 without the VM needing extra RAM. Only extracted feature text goes from the VM to the host; the sample file never leaves the VM.

```
+--------------------- Host (RTX 4050) ---------------------+
|  ollama serve   (OLLAMA_HOST=0.0.0.0:11434, firewall:     |
|                  allow inbound only from 192.168.56.0/24) |
+------------------------- vboxnet0 192.168.56.1 ----------+
                              |  Host-Only (no NAT)
+---------------- Analysis VM 192.168.56.101 --------------+
|  python src/agent/agent_react.py --sample ... --guard     |
|  FLOSS, capa, diec, Ghidra, YARA rules, samples, logs     |
+-----------------------------------------------------------+
```

## 1. VM network

1. Give the VM a single Host-Only adapter. Remove any NAT/bridged adapter.
2. On the host, allow port 11434 only from the Host-Only subnet, and block forwarding from that subnet to the internet.
3. Inside the VM, check that `curl http://192.168.56.1:11434/api/tags` works and that `curl https://example.com` fails.

The agent refuses to start if `OLLAMA_HOST` is not a loopback or private (RFC 1918) address.

## 2. Python environment (inside the VM)

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Install packages from a local wheel folder copied into the VM if the VM has no package mirror: on a connected machine run `pip download -r requirements.txt -d wheels`, copy `wheels/`, then `pip install --no-index --find-links wheels -r requirements.txt`.

Copy the Qwen tokenizer to `models/tokenizer/tokenizer.json` (from the `Qwen/Qwen2.5-Coder-3B-Instruct` model files). Without it the agent uses a digit-aware estimate (each digit = 1 token, Base64/hex runs at 1.2 chars/token, other text at 2.5 chars/token), which overcounts by 6-60% and so compacts the prompt earlier than needed.

## 3. Analysis tools

| Tool | Install | Environment variable |
|---|---|---|
| FLOSS | standalone binary from the FLOSS GitHub releases | `FLOSS_PATH` (default `floss` on PATH), `FLOSS_TIMEOUT` (s, default 300) |
| capa | standalone binary (bundles rules and signatures) | `CAPA_PATH` (default `capa`), `CAPA_RULES` and `CAPA_SIGS` (optional custom rule / signature dirs), `CAPA_TIMEOUT` (default 300) |
| Detect It Easy | `diec` console binary from the DIE release | `DIEC_PATH` (default `diec`), `DIEC_TIMEOUT` (default 120) |
| Ghidra | Ghidra 12.x + JDK 25 (Ghidra 11.x: JDK 21) | `GHIDRA_HOME` (folder containing `support/analyzeHeadless`), `GHIDRA_TIMEOUT` (s per headless run, default 300), `GHIDRA_SCRIPTS` (default `ghidra_scripts/`) |
| YARA | `yara-python` (pip) | `YARA_RULES_DIR` (base rules, default `rules/base/`) |

Other variables:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint, e.g. `http://192.168.56.1:11434` |
| `CLEAN_DIR` | `data/clean_files/` | Clean files used to reject false-positive YARA rules |
| `DYNAMIC_LOG_DIR` | `data/dynamic_logs/` | Procmon / INetSim logs per sample |
| `TOKENIZER_PATH` | `models/tokenizer/tokenizer.json` | Qwen tokenizer for exact token counts |
| `TOOL_TOKEN_CAP` | `450` | Token cap per tool output (hard maximum 600) |

## 4. Dynamic logs (manual detonation)

The agent never runs a sample. To include dynamic behavior, detonate the sample yourself in a disposable snapshot and copy the logs:

1. Revert the detonation VM to a clean snapshot. Point its DNS and default gateway to the INetSim VM (e.g. REMnux running `inetsim`).
2. Start Procmon with a filter on the sample's process name, run the sample, wait 2-5 minutes, stop capture.
3. Export Procmon as CSV with the default columns (`Time of Day, Process Name, PID, Operation, Path, Result, Detail`).
4. Copy INetSim's `service.log` (and optionally `report.*.txt`) for that session.
5. Put both in `data/dynamic_logs/<key>/` where `<key>` is the sample's SHA-256, file name, or file name without extension:

```
data/dynamic_logs/sampleG.exe/procmon.csv
data/dynamic_logs/sampleG.exe/service.log
```

6. Revert the detonation VM.

If no folder exists for a sample, the Procmon and INetSim tools report "no log available"; this is not counted as a tool error.

## 5. Data layout

```
data/samples/            samples to analyze (lab set, Sample G, injection test set)
data/clean_files/        known-clean binaries (FPR check for generated YARA rules)
data/ground_truth/       one JSON label file per sample (see docs/benchmark.md)
data/dynamic_logs/       optional Procmon/INetSim logs, see section 4
```
