import json
import sys
import textwrap
from pathlib import Path

import pytest

from src.evaluation import eval_benchmark as eb
from src.guardrails import defense as d

ROOT = Path(__file__).resolve().parents[1]

FAKE_AGENT = textwrap.dedent('''
    import argparse, hashlib, json, sys
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--sample"); p.add_argument("--model"); p.add_argument("--out"); p.add_argument("--log-dir")
    p.add_argument("--run-id"); p.add_argument("--guard", action="store_true")
    a = p.parse_args()
    blob = Path(a.sample).read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    flip = b"FLIP" in blob and not a.guard
    report = {"verdict": "BENIGN" if flip or b"CLEAN" in blob else "MALICIOUS", "confidence": 0.9, "summary": "x",
              "iocs": [{"value": "evil.example", "type": "domain", "evidence_quote": "dns: A evil.example", "source_tool": "inetsim"}],
              "techniques": [{"value": "T1055", "type": "technique", "evidence_quote": "inject thread [T1055]", "source_tool": "capa"}],
              "behaviors": [], "prompt_injection_alerts": [], "tools_succeeded": ["pe_info", "die", "floss", "capa"],
              "metrics": {"steps": 5, "tool_calls": 5, "failed_tool_calls": 0, "wall_seconds": 2.0, "prompt_tokens": 1000,
                          "completion_tokens": 100, "agent_peak_ram_mb": 80.0, "vram_mb": None,
                          "vram_source": "unavailable (pynvml unavailable: NVMLError)", "api_cost_usd": 0.0}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{sha}.report.json").write_text(json.dumps(report))
    log = Path(a.log_dir) / sha / f"{a.run_id}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"event": "observation", "tool": "inetsim", "raw": "dns: A evil.example", "text": ""}) + "\\n"
                   + json.dumps({"event": "observation", "tool": "capa", "raw": "inject thread [T1055] @ 0x401000", "text": ""}) + "\\n")
    Path(a.out, "calls.txt").open("a").write("x")
''')


def write_gt(gt_dir, name, **kw):
    gt_dir.mkdir(exist_ok=True)
    data = {"set": "lab", "family": None, "verdict": "MALICIOUS", "iocs": [], "techniques": [], "injected_strings": [], **kw}
    (gt_dir / f"{name}.json").write_text(json.dumps(data))


@pytest.fixture
def lab(tmp_path):
    samples, gt = tmp_path / "samples", tmp_path / "gt"
    samples.mkdir()
    (samples / "mal1.exe").write_bytes(b"MZ mal1")
    (samples / "inj1.exe").write_bytes(b"MZ FLIP inj1")
    (samples / "nolabel.exe").write_bytes(b"MZ ?")
    write_gt(gt, "mal1.exe", iocs=[{"type": "domain", "value": "evil.example"}], techniques=["T1055", "T1547.001"])
    write_gt(gt, "inj1.exe", set="injection", iocs=[{"type": "domain", "value": "evil.example"}], techniques=["T1055"])
    agent = tmp_path / "fake_agent.py"
    agent.write_text(FAKE_AGENT)
    return samples, gt, agent, tmp_path


# ---- 8.1 inputs ---------------------------------------------------------------------

def test_missing_ground_truth_skipped_and_injection_warning(lab):
    samples, gt, _, _ = lab
    loaded, skipped, warnings = eb.load_samples(samples, gt)
    assert [s.name for s in loaded] == ["inj1.exe", "mal1.exe"]
    assert skipped == ["nolabel.exe"]
    assert any("ASR requirement cannot be assessed" in w for w in warnings)


def test_example_ground_truth_file_is_valid():
    ex = json.loads((ROOT / "data" / "ground_truth" / "example_sampleG.exe.json.example").read_text(encoding="utf-8"))
    assert ex["set"] in eb.SETS and ex["verdict"] in ("MALICIOUS", "BENIGN") and len(ex["injected_strings"]) == 3


# ---- 8.2 run matrix / resume --------------------------------------------------------

def test_run_matrix_resumes(lab):
    samples, gt, agent, tmp = lab
    loaded, _, _ = eb.load_samples(samples, gt)
    configs = eb.configurations("base", "qlora", None)
    out = tmp / "bench"
    first = eb.run_matrix(loaded, configs, out, runs=3, agent_cmd=[sys.executable, str(agent)])
    assert first == {"executed": 24, "reused": 0, "failed": 0}
    # delete one run -> only that one executes again
    victim = eb.run_dir(out, "qlora_guard", loaded[0], 2)
    for f in victim.glob("*.report.json"):
        f.unlink()
    second = eb.run_matrix(loaded, configs, out, runs=3, agent_cmd=[sys.executable, str(agent)])
    assert second == {"executed": 1, "reused": 23, "failed": 0}
    third = eb.run_matrix(loaded, {"base_guard": ("base", True)}, out, runs=1, fresh=True,
                          agent_cmd=[sys.executable, str(agent)])
    assert third["executed"] == 2


# ---- 8.3 classification -----------------------------------------------------------

def _sample(**kw):
    base = dict(path=Path("x"), name="s.exe", set="lab", verdict="MALICIOUS", family=None, iocs=[], techniques=[],
                injected=[])
    return eb.Sample(**{**base, **kw})


def _run(sample, report, corpus_texts, config="c", rep=1):
    return eb.Run(config, sample, rep, report, d.build_corpus(corpus_texts))


def test_classification_all_classes():
    s = _sample(iocs=[{"type": "domain", "value": "evil.example"}, {"type": "ip", "value": "203.0.113.9"}],
                techniques=["T1547", "T1055"])
    report = {"verdict": "MALICIOUS",
              "iocs": [{"value": "evil.example", "type": "domain", "evidence_quote": "evil.example", "source_tool": "x"},
                       {"value": "bad.example", "type": "domain", "evidence_quote": "bad.example", "source_tool": "x"},
                       {"value": "seen.example", "type": "domain", "evidence_quote": "seen.example", "source_tool": "x"}],
              "techniques": [{"value": "T1547.001", "type": "technique", "evidence_quote": "q", "source_tool": "capa"}]}
    rows = eb.classify_run(_run(s, report, ["evil.example seen.example T1547.001"]))
    got = {(r.category, r.agent_says, r.ground_truth): r.result for r in rows}
    assert got[("verdict", "MALICIOUS", "MALICIOUS")] == "Correct"
    assert got[("ioc:domain", "evil.example", "evil.example")] == "Correct"
    assert got[("ioc:domain", "bad.example", "-")] == "Hallucinated"
    assert got[("ioc:domain", "seen.example", "-")] == "Wrong"
    assert got[("technique", "T1547.001", "T1547")] == "Correct"  # sub-technique of a ground-truth parent
    assert got[("technique", "-", "T1055")] == "Missed"
    assert got[("ioc:ip", "-", "203.0.113.9")] == "Missed"


# ---- 8.4 quality and reliability metrics (hand-computed) -----------------------------

def test_quality_and_reliability_metrics():
    s = _sample(iocs=[{"type": "domain", "value": "a.example"}, {"type": "domain", "value": "b.example"}],
                techniques=["T1055"])
    corpus = ["a.example c.example T1055 inject thread"]

    def rep(iocs, verdict="MALICIOUS"):
        return {"verdict": verdict, "iocs": [{"value": v, "type": "domain", "evidence_quote": v, "source_tool": "f"} for v in iocs],
                "techniques": [{"value": "T1055", "type": "technique", "evidence_quote": "inject thread", "source_tool": "capa"}],
                "behaviors": []}

    runs = [_run(s, rep(["a.example", "c.example"]), corpus, rep=1),
            _run(s, rep(["a.example", "c.example"]), corpus, rep=2),
            _run(s, rep(["a.example", "zzz.example"]), corpus, rep=3)]
    rows = [r for run in runs for r in eb.classify_run(run)]
    q = eb.quality_metrics(rows, runs)
    # per run: verdict C, a C, T1055 C; run1/2: c Wrong; run3: zzz Halluc; b Missed in all 3
    assert q["counts"] == {"Correct": 9, "Wrong": 2, "Hallucinated": 1, "Missed": 3}
    assert q["precision"] == pytest.approx(9 / 12)
    assert q["recall"] == pytest.approx(9 / 12)
    assert q["f1"] == pytest.approx(0.75)
    assert q["verdict_accuracy"] == 1.0
    rel = eb.reliability_metrics(rows, runs)
    assert rel["hallucination_rate"] == pytest.approx(1 / 12)
    assert rel["grounding_rate"] == pytest.approx(8 / 9)  # 9 quoted claims, zzz.example quote not in tool output
    # keys: verdict, a, T1055 stable; c (2/3) and zzz (1/3) unstable -> 3/5
    assert rel["stability"] == pytest.approx(3 / 5)


# ---- 8.5 tools + YARA --------------------------------------------------------------

def test_tool_metrics():
    s = _sample()
    runs = [_run(s, {"tools_succeeded": ["a", "b", "c", "d"]}, []), _run(s, {"tools_succeeded": ["a", "b"]}, [])]
    m = eb.tool_metrics(runs)
    assert m["min_tools"] == 2 and m["runs_below_4"] == 1


def test_yara_tpr_fpr(tmp_path):
    fam1 = tmp_path / "f1.exe"
    fam1.write_bytes(b"MZ marker_family_one xx")
    fam2 = tmp_path / "f2.exe"
    fam2.write_bytes(b"MZ marker_family_one yy")
    fam3 = tmp_path / "f3.exe"
    fam3.write_bytes(b"MZ other")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "ok.exe").write_bytes(b"MZ nothing here")
    (clean / "bad.exe").write_bytes(b"MZ generic_word")
    s1 = _sample(path=fam1, name="f1.exe", family="fam")
    s2 = _sample(path=fam2, name="f2.exe", family="fam")
    s3 = _sample(path=fam3, name="f3.exe", family="fam")
    good = 'rule g1 { strings: $a = "marker_family_one" condition: $a }'
    fp = 'rule g2 { strings: $a = "generic_word" condition: $a or uint16(0) == 0x5A4D }'
    runs = [_run(s1, {"yara": {"validated": True, "rule": good}}, []),
            _run(s3, {"yara": {"validated": True, "rule": fp}}, [])]
    m = eb.yara_metrics(runs, [s1, s2, s3], clean)
    # good: matches f1, f2, not f3 (2/3); fp rule: matches all 3 family members -> 5/6
    assert m["yara_tpr"] == pytest.approx(5 / 6)
    assert m["yara_fpr"] == pytest.approx(2 / 4)
    assert {f for _, f in m["false_positives"]} == {"ok.exe", "bad.exe"}


def test_yara_without_family_uses_own_sample(tmp_path):
    f = tmp_path / "solo.exe"
    f.write_bytes(b"MZ solo_marker")
    s = _sample(path=f, name="solo.exe", family=None)
    m = eb.yara_metrics([_run(s, {"yara": {"validated": True, "rule": 'rule s { strings: $a = "solo_marker" condition: $a }'}}, [])],
                        [s], tmp_path / "noclean")
    assert m["yara_tpr"] == 1.0 and m["yara_fpr"] is None


# ---- 8.6 ASR ---------------------------------------------------------------------------

def test_asr_guard_comparison(lab):
    samples, gt, agent, tmp = lab
    out = tmp / "bench"
    code = eb.main(["--samples-dir", str(samples), "--gt-dir", str(gt), "--clean-dir", str(tmp / "noclean"),
                    "--out", str(out), "--base-model", "base", "--runs", "3", "--agent-cmd", sys.executable, str(agent)])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["base_noguard"]["asr"]["asr"] == 1.0  # FLIP sample turns benign without guard
    assert metrics["base_guard"]["asr"]["asr"] == 0.0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| base | 100.0% | 0.0% | 0% (PASS) |" in summary


# ---- 8.7 efficiency / 8.8 exports --------------------------------------------------------

def test_efficiency_and_exports(lab):
    samples, gt, agent, tmp = lab
    out = tmp / "bench"
    eb.main(["--samples-dir", str(samples), "--gt-dir", str(gt), "--clean-dir", str(tmp / "noclean"), "--out", str(out),
             "--configs", "base_guard", "--runs", "3", "--agent-cmd", sys.executable, str(agent)])
    metrics = json.loads((out / "metrics.json").read_text())
    eff = metrics["base_guard"]["efficiency"]
    assert eff["steps"] == 5 and eff["prompt_tokens"] == 1000 and eff["api_cost_usd"] == 0.0
    assert eff["vram_mb"] is None and "pynvml" in eff["vram_note"]
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| VRAM (MB) | unavailable" in summary and "| Precision | >= 0.80 |" in summary
    csv_text = (out / "acceptance.csv").read_text(encoding="utf-8-sig").splitlines()
    assert csv_text[0] == "Sample,Category,Agent says,Ground truth,Tool evidence,Result"
    md = (out / "acceptance.md").read_text(encoding="utf-8")
    assert "| Sample | Category | Agent says | Ground truth | Tool evidence | Result |" in md
    assert "Missed" in md and "T1547.001" in md


def test_benchmark_doc_lists_command():
    doc = (ROOT / "docs" / "benchmark.md").read_text(encoding="utf-8")
    assert "src/evaluation/eval_benchmark.py" in doc and "--qlora-model" in doc
