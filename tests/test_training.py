import json
import random
import sys
from pathlib import Path

import pytest

from src.agent import agent_react as ar
from src.guardrails import defense as d
from src.tools import malware_tools as t
from src.training import train_qlora as tq

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return tq.build_dataset(400, seed=3)


# ---- 7.1 generators match the agent's formats ------------------------------------------

def test_targets_validate_against_agent_schemas(dataset):
    examples, _ = dataset
    for ex in examples:
        target = ex.completion[0]["content"]
        if ex.meta["kind"] == "report":
            d.ModelReport.model_validate_json(target)
        else:
            step = d.StepAction.model_validate_json(target)
            assert step.tool in ar.ALL_TOOLS
        assert ex.prompt[0]["role"] == "system" and ex.prompt[-1]["role"] == "user"


def test_prompts_fit_training_length(dataset):
    examples, _ = dataset
    longest = max(ar.Agent.prompt_tokens(ex.prompt + ex.completion) for ex in examples)
    assert longest <= tq.MAX_SEQ_LEN


def test_reserved_ioc_ranges_only(dataset):
    examples, _ = dataset
    for ex in examples:
        j = json.loads(ex.completion[0]["content"])
        for c in j.get("iocs", []):
            if c["type"] == "ip":
                assert c["value"].startswith(tuple(tq.IP_NETS))
            if c["type"] in ("domain", "url"):
                assert ".example" in c["value"]


def test_report_claims_are_grounded_in_prompt(dataset):
    examples, _ = dataset
    for ex in [e for e in examples if e.meta["kind"] == "report"]:
        text = "\n".join(m["content"] for m in ex.prompt)
        j = json.loads(ex.completion[0]["content"])
        for c in j["iocs"] + j["techniques"] + j["behaviors"]:
            assert c["evidence_quote"] in text


# ---- 7.2 distribution, injection targets -----------------------------------------------

def test_distribution_2000():
    _, stats = tq.build_dataset(2000, seed=7)
    assert tq.check_distribution(stats) == [], stats
    assert 0.48 <= stats["adversarial_ratio"] <= 0.52
    assert all(stats["per_technique_ratio"][k] >= 0.10 for k in tq.TECHNIQUES)
    assert stats["benign_report_targets"] > 0


def test_adversarial_reports_keep_verdict_and_list_injection(dataset):
    examples, _ = dataset
    adv_reports = [e for e in examples if e.meta["kind"] == "report" and e.meta["adversarial"]]
    assert adv_reports
    for e in adv_reports:
        j = json.loads(e.completion[0]["content"])
        assert j["verdict"] == "MALICIOUS"
        assert j["prompt_injection_alerts"]


@pytest.mark.parametrize("technique", tq.TECHNIQUES)
def test_scanner_detects_every_generated_injection(technique):
    rng = random.Random(1)
    for _ in range(10):
        strings = tq.injection_strings(technique, rng)
        assert d.scan_strings(strings), strings


# ---- 7.3 leakage check -------------------------------------------------------------------

def test_leakage_check_finds_planted_overlap(dataset, tmp_path):
    examples, _ = dataset
    some_ioc = next(c["value"] for e in examples if e.meta["kind"] == "report"
                    for c in json.loads(e.completion[0]["content"])["iocs"])
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "lab1.exe").write_bytes(b"MZ\0\0" + some_ioc.encode() + b"\0kernel32.dll\0")
    assert some_ioc in tq.leakage_check(examples, bench)


def test_leakage_check_clean(dataset, tmp_path):
    examples, _ = dataset
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "lab1.exe").write_bytes(b"MZ\0\0VirtualAllocEx\0kernel32.dll\0http://real-c2.ru/x\0")
    assert tq.leakage_check(examples, bench) == []


def test_build_cli_fails_on_leak(tmp_path, monkeypatch):
    examples, _ = tq.build_dataset(40, seed=7)
    ioc = next(c["value"] for e in examples if e.meta["kind"] == "report"
               for c in json.loads(e.completion[0]["content"])["iocs"])
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "x.bin").write_bytes(ioc.encode())
    code = tq.main(["build", "--n", "40", "--seed", "7", "--out", str(tmp_path / "ds.jsonl"),
                    "--leak-check-dir", str(bench)])
    assert code == 1


def test_build_cli_writes_train_and_eval(tmp_path):
    out = tmp_path / "ds.jsonl"
    assert tq.main(["build", "--n", "200", "--out", str(out)]) == 0
    train = out.read_text(encoding="utf-8").splitlines()
    held = out.with_name("ds.eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) + len(held) == 200 and len(held) >= 4
    assert {"prompt", "completion", "meta"} <= set(json.loads(train[0]))


# ---- 7.5 scoring logic used by the held-out evaluation ------------------------------------

def test_score_completion():
    target = json.dumps({"verdict": "MALICIOUS"})
    good = d.ModelReport(verdict="MALICIOUS", confidence=1, summary="x").model_dump_json()
    bad = d.ModelReport(verdict="BENIGN", confidence=1, summary="x").model_dump_json()
    assert tq.score_completion(good, target, "report", True) == (True, True)
    assert tq.score_completion(bad, target, "report", True) == (True, False)
    assert tq.score_completion("nope", target, "report", True) == (False, False)
    assert tq.score_completion('{"thought":"a","tool":"floss","args":{}}', "", "step", False) == (True, None)


# ---- 7.6 export -------------------------------------------------------------------------

def test_modelfile_and_export_commands(tmp_path):
    text = tq.modelfile_text()
    assert text.startswith(f"FROM ./{tq.GGUF_NAME}")
    for line in ("PARAMETER num_ctx 4096", "PARAMETER temperature 0", "PARAMETER top_p 0.1",
                 "PARAMETER repeat_penalty 1.05", '<|im_start|>{{ .Role }}'):
        assert line in text
    assert (ROOT / "Modelfile").read_text(encoding="utf-8") == text
    cmds = tq.export_commands(Path("merged"), tmp_path, Path("/content/llama.cpp"))
    assert cmds[0][1].endswith("convert_hf_to_gguf.py") and "--outtype" in cmds[0]
    assert cmds[1][-1] == "Q5_K_M" and cmds[1][-2].endswith(tq.GGUF_NAME)
