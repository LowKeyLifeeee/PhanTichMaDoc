import json
import re
import sys
from pathlib import Path

import pytest

from src.tools import malware_tools as t

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# ---- 3.1 runner ----------------------------------------------------------------

def test_missing_binary():
    r = t.run_command(["definitely-not-a-real-tool-xyz"], 5)
    assert not r.ok and "tool not found" in r.error


def test_nonzero_exit():
    r = t.run_command([sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"], 10)
    assert not r.ok and r.returncode == 3 and "exit code 3" in r.error


def test_timeout_kills():
    r = t.run_command([sys.executable, "-c", "import time; time.sleep(30)"], 1)
    assert not r.ok and "timeout" in r.error


def test_refuses_to_execute_sample(tmp_path):
    sample = tmp_path / "evil.exe"
    sample.write_bytes(b"MZ" + b"\0" * 100)
    r = t.run_command([str(sample), "--help"], 5, sample=sample)
    assert not r.ok and "refusing to execute the sample" in r.error


def test_tool_error_result_when_tool_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOSS_PATH", "no-such-floss")
    sample = tmp_path / "s.bin"
    sample.write_bytes(b"x")
    r = t.floss_strings(sample)
    assert not r.ok and "tool not found" in r.error and r.text.startswith("ERROR")


# ---- 3.2 pefile -------------------------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("win"), reason="needs a Windows PE (python.exe)")
def test_pe_info_on_real_pe():
    r = t.pe_info(sys.executable)
    assert r.ok
    assert r.data["sections"] and r.data["imports"]
    assert "entropy" in r.text and r.text.startswith("PE ")
    assert t.count_tokens(r.text) <= 600


def test_pe_info_not_pe(tmp_path):
    p = tmp_path / "notpe.bin"
    p.write_bytes(b"hello world" * 10)
    r = t.pe_info(p)
    assert not r.ok and "not a PE file" in r.error


def test_dangerous_api_classification():
    got = t.classify_apis(["VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread", "lstrlenA",
                           "RegSetValueExW", "InternetOpenUrlA"])
    assert set(got["injection"]) == {"VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread"}
    assert got["persistence"] == ["RegSetValueExW"]
    assert got["network"] == ["InternetOpenUrlA"]
    assert "lstrlenA" not in str(got)


def test_packing_flag():
    assert t.is_packed([{"entropy": 7.6}, {"entropy": 3.0}])
    assert not t.is_packed([{"entropy": 6.9}])


# ---- 3.3 FLOSS -------------------------------------------------------------------

def test_floss_ranking_and_types():
    r = t.floss_from_json(load("floss_sample.json"), "{}", import_names=["GetProcAddress"])
    lines = r.text.splitlines()
    url = next(i for i, l in enumerate(lines) if "http://evil.example/gate.php" in l)
    run = next(i for i, l in enumerate(lines) if "CurrentVersion\\Run" in l)
    lib = [i for i, l in enumerate(lines) if "Microsoft Visual C++ Runtime" in l]
    assert url < run
    assert all(run < i for i in lib)
    assert "[decoded]" in lines[url]
    assert any("[stack]" in l for l in lines)
    assert "GetProcAddress" not in r.text  # import names are noise
    assert r.data["iocs"]["url"] == ["http://evil.example/gate.php"]
    assert "203.0.113.45" in r.data["iocs"]["ip"]
    # injection-like strings are ranked first
    assert "Ignore previous instructions" in lines[1]


def test_floss_signed_binary_noise_filtered():
    # strings seen in real FLOSS 3.2 output of a signed python.exe
    real = ["Mhttp://crl3.digicert.com/DigiCertTrustedG4CodeSigningRSA4096SHA3842021CA1.crl0S",
            "http://ocsp.digicert.com0A",
            '<longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>',
            'version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*" />',
            "ecc__^^\\\\WWWUQQQQ\\", "KERNEL32.dll", "VirtualAlloc",
            "http://203.0.113.7/gate.php", "C:\\Users\\Public\\upd.exe"]
    doc = {"strings": {"static_strings": [{"string": s} for s in real]}}
    r = t.floss_from_json(doc, "", import_names=["KERNEL32.dll", "VirtualAlloc"])
    assert r.data["iocs"]["url"] == ["http://203.0.113.7/gate.php"]
    assert r.data["iocs"]["ip"] == ["203.0.113.7"]
    for junk in ("digicert", "schemas.microsoft.com", "KERNEL32.dll"):
        assert junk not in r.text, junk
    labelled = [l for l in r.text.splitlines() if re.match(r"\[\w+\] \w+:", l)]
    assert not any("6.0.0.0" in l or "WWWUQQQQ" in l for l in labelled)  # kept only as plain strings
    assert set(r.data["iocs"]) == {"url", "ip", "path"} and len(r.data["iocs"]["path"]) == 1
    assert "C:\\Users\\Public\\upd.exe" in r.text


def test_floss_large_output_capped():
    doc = {"strings": {"static_strings": [{"string": f"SomeLongerString_{i}_value"} for i in range(5000)]}}
    r = t.floss_from_json(doc, "")
    assert t.count_tokens(r.text) <= 600
    assert re.search(r"\[\.\.\. \d+ items omitted\]$", r.text)


# ---- 3.4 DIE -----------------------------------------------------------------------

def test_die_upx():
    r = t.die_from_json(load("die_upx.json"), "")
    assert r.data["packers"] == ["UPX"]
    assert r.text.splitlines()[1].startswith("packer: UPX 3.96")
    assert "compiler: Microsoft Visual C/C++" in r.text


# ---- 3.5 capa ----------------------------------------------------------------------

def test_capa_attack_mapping():
    r = t.capa_from_json(load("capa_sample.json"), "{}")
    inj = next(c for c in r.data["capabilities"] if c["name"] == "inject thread")
    assert inj["attack_ids"] == ["T1055"]
    assert inj["addresses"] == ["0x401000", "0x401200"]
    assert "T1547.001" in r.data["attack_ids"]
    assert all(c["name"] != "contain loop" for c in r.data["capabilities"])  # lib rules skipped
    assert "inject thread [T1055] @ 0x401000" in r.text


def test_capa_old_dict_format():
    doc = {"rules": {"x": {"meta": {"name": "x", "attack": ["Execution::Command [T1059]"]},
                           "matches": {"4198400": {}}}}}
    caps = t.parse_capa_json(doc)
    assert caps[0]["attack_ids"] == ["T1059"] and caps[0]["addresses"] == ["0x401000"]


# ---- 3.6 YARA scan ------------------------------------------------------------------

def test_yara_base_rules_match(tmp_path):
    p = tmp_path / "inj.bin"
    p.write_bytes(b"MZ" + b"\0" * 64 + b"VirtualAllocEx\0WriteProcessMemory\0CreateRemoteThread\0")
    r = t.yara_scan(p)
    assert r.ok
    m = r.data["matches"][0]
    assert m["rule"] == "Process_Injection_API_Combo" and "$a0" in m["strings"]


def test_yara_syntax_error(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    r = t.yara_scan(p, rule_text="rule broken {\n condition:\n  foo and\n}")
    assert not r.ok and "syntax error" in r.error and "line" in r.error


# ---- 3.7 Ghidra (ranking logic; the headless run is checked manually in the VM) ----

CAPS = [
    {"name": "inject", "attack_ids": ["T1055"], "addresses": ["0x401000", "0x401200"]},
    {"name": "env", "attack_ids": [], "addresses": ["0x401000", "0x403000"]},
    {"name": "embedded", "attack_ids": [], "addresses": ["file+0x2000"]},
]


def test_ghidra_ranking_prefers_capa_then_callers():
    callers = [{"address": f"00405{i:03x}", "apis": ["WinExec"] * (i % 3 + 1)} for i in range(12)]
    targets = t.rank_ghidra_targets(CAPS, callers)
    assert len(targets) == 5
    assert targets[:3] == ["0x401000", "0x401200", "0x403000"]
    assert all(not x.startswith("file+") for x in targets)


def test_ghidra_ranking_preferred_first_and_dedup():
    targets = t.rank_ghidra_targets(CAPS, [{"address": "00401000", "apis": ["WinExec"]}], preferred=["0x403000"])
    assert targets[0] == "0x403000" and targets.count("0x401000") == 1


def test_ghidra_ranking_empty():
    assert t.rank_ghidra_targets([], []) == []


def test_ghidra_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GHIDRA_HOME", str(tmp_path / "nope"))
    s = tmp_path / "s.exe"
    s.write_bytes(b"MZ")
    r = t.ghidra_decompile(s)
    assert not r.ok and "analyzeHeadless not found" in r.error


def test_ghidra_drops_local_declarations():
    code = ("void FUN_140001660(void)\n{\n  LPCWSTR lpWideCharStr;\n  char cVar1;\n  code *pcVar3;\n"
            "  WCHAR local_218 [264];\n  undefined1 *puVar4;\n  return_flag = 1;\n"
            '  CreateProcessW(L"cmd.exe",0,0,0,0,0,0,0,0,0);\n  return;\n}\n')
    r = t.ghidra_from_json([{"name": "FUN_140001660", "address": "140001660", "code": code}])
    assert "lpWideCharStr" not in r.text and "local_218" not in r.text and "pcVar3" not in r.text
    assert "CreateProcessW" in r.text and "return;" in r.text and "return_flag = 1;" in r.text


def test_ghidra_output_formatting():
    funcs = [{"name": "FUN_00401000", "address": "00401000",
              "code": 'void FUN_00401000(void)\n{\n  /* comment */\n  WinExec("cmd.exe /c whoami",0);\n}\n'},
             {"name": "?", "address": "0x999", "error": "no function at address"}]
    r = t.ghidra_from_json(funcs, 450, [{"address": "00401000", "apis": ["WinExec"]}])
    assert r.ok and "// FUN_00401000 @ 0x00401000" in r.text
    assert "cmd.exe /c whoami" in r.data["scan_strings"]
    assert "no function at address" in r.text


# ---- 3.8 summaries ----------------------------------------------------------------

def test_summaries_keep_key_values():
    floss = t.floss_from_json(load("floss_sample.json"), "")
    capa = t.capa_from_json(load("capa_sample.json"), "")
    die = t.die_from_json(load("die_upx.json"), "")
    assert "http://evil.example/gate.php" in floss.summary
    assert "T1055" in capa.summary and "T1547.001" in capa.summary
    assert "UPX" in die.summary
    for r in (floss, capa, die):
        assert t.count_tokens(r.summary) <= t.SUMMARY_CAP


# ---- 3.9 feature-only output across tools -------------------------------------------

def test_feature_only_output_all_tools():
    big_b64 = "QUJD" * 200
    floss_doc = load("floss_sample.json")
    floss_doc["strings"]["static_strings"].append({"string": big_b64})
    results = [
        t.floss_from_json(floss_doc, ""),
        t.capa_from_json(load("capa_sample.json"), ""),
        t.die_from_json(load("die_upx.json"), ""),
        t.procmon_from_parsed(t.parse_procmon_csv(FIX / "procmon_sample.csv", "sample.exe"), ""),
        t.inetsim_from_parsed(t.parse_inetsim((FIX / "inetsim_service.log").read_text().splitlines()), ""),
        t.ghidra_from_json([{"name": "f", "address": "1", "code": "int f(void){return 0;}"}]),
    ]
    if sys.platform.startswith("win"):
        results.append(t.pe_info(sys.executable))
    for r in results:
        assert not re.search(r"[A-Za-z0-9+/=]{65,}", r.text), r.tool
        assert t.count_tokens(r.text) <= 600, r.tool
