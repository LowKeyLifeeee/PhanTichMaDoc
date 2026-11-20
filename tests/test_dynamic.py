import shutil
from pathlib import Path

from src.tools import malware_tools as t

FIX = Path(__file__).parent / "fixtures"


def test_procmon_run_key_and_tree():
    parsed = t.parse_procmon_csv(FIX / "procmon_sample.csv", "sample.exe")
    assert parsed["persistence"][0]["key"].endswith("CurrentVersion\\Run\\Updater")
    assert "updater.exe" in parsed["persistence"][0]["data"]
    assert parsed["process_creates"][0]["cmdline"].startswith("cmd.exe /c del")
    assert parsed["deleted"][0]["path"].endswith("sample.exe")  # child process (cmd.exe) followed
    assert not any("RecentDocs" in r["key"] for r in parsed["registry_writes"])  # explorer.exe filtered out
    r = t.procmon_from_parsed(parsed, "")
    lines = r.text.splitlines()
    assert lines[1].startswith("persistence: RegSetValue HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run")
    assert any("file drop (executable)" in l for l in lines)


def test_inetsim_c2_beacon():
    parsed = t.parse_inetsim((FIX / "inetsim_service.log").read_text().splitlines())
    assert {"type": "A", "name": "evil.example"} in parsed["dns"]
    assert parsed["http"][0] == {"method": "POST", "host": "evil.example", "path": "/gate.php",
                                 "url": "http://evil.example/gate.php"}
    assert "smtp_25_tcp" in parsed["other_services"]
    r = t.inetsim_from_parsed(parsed, "")
    assert "http: POST http://evil.example/gate.php" in r.text
    assert "dns: A evil.example" in r.text


def test_inetsim_report_format():
    lines = ["2026-11-20 10:00:01  First simulated date in log file",
             "2026-11-20 10:00:01  DNS connection, type: A, class: IN, requested name: c2.example",
             "2026-11-20 10:00:02  HTTP connection, method: GET, URL: http://c2.example/payload.bin, file name: x"]
    parsed = t.parse_inetsim(lines)
    assert parsed["dns"] == [{"type": "A", "name": "c2.example"}]
    assert parsed["http"][0]["path"] == "/payload.bin"


def test_no_dynamic_log_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DYNAMIC_LOG_DIR", str(tmp_path))
    s = tmp_path / "s.exe"
    s.write_bytes(b"MZ")
    for r in (t.procmon_parse(s), t.inetsim_parse(s)):
        assert r.ok and not r.has_data and "No " in r.text


def test_dynamic_lookup_by_sample_name(monkeypatch, tmp_path):
    monkeypatch.setenv("DYNAMIC_LOG_DIR", str(tmp_path))
    d = tmp_path / "sample.exe"
    d.mkdir()
    shutil.copy(FIX / "procmon_sample.csv", d / "procmon.csv")
    shutil.copy(FIX / "inetsim_service.log", d / "service.log")
    s = tmp_path / "sample.exe.bin"
    s.write_bytes(b"MZ")
    sample = tmp_path / "x" / "sample.exe"
    sample.parent.mkdir()
    sample.write_bytes(b"MZ")
    assert t.procmon_parse(sample).has_data
    assert t.inetsim_parse(sample).has_data


def test_setup_doc_lists_every_env_var():
    src = (Path(t.__file__)).read_text(encoding="utf-8")
    import re
    names = set(re.findall(r'_env(?:_int)?\("([A-Z_]+)"', src)) | set(re.findall(r'os\.environ(?:\.get)?\(?\[?"([A-Z_]+)"', src))
    doc = (Path(__file__).resolve().parents[1] / "docs" / "setup.md").read_text(encoding="utf-8")
    missing = sorted(n for n in names if n not in doc)
    assert not missing, missing
