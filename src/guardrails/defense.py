"""Prompt-injection defenses, report schemas and the extractive grounding verifier.

Layer 1: spotlighting (random per-run delimiters) + heuristic injection scanner.
Layer 3: tool-argument blocking, extractive grounding of every claim, verdict lock.
Layer 2 (the QLoRA-tuned model) lives in ``src/training/train_qlora.py``.

This module has no dependency on the tool wrappers so both sides can import it.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

Verdict = Literal["MALICIOUS", "BENIGN"]
ClaimType = Literal[
    "ip", "domain", "url", "file_path", "registry", "mutex", "hash", "email",
    "command", "technique", "behavior", "other",
]


class Claim(BaseModel):
    value: str = Field(max_length=200)
    type: ClaimType
    evidence_quote: str = Field(max_length=300)
    source_tool: str = Field(max_length=32)


class StepArgs(BaseModel):
    addresses: list[str] = Field(default_factory=list, max_length=5)
    strings: list[str] = Field(default_factory=list, max_length=10)
    min_match: int | None = Field(default=None, ge=1, le=10)


class StepAction(BaseModel):
    thought: str = Field(max_length=300)
    tool: str = Field(max_length=32)
    args: StepArgs = Field(default_factory=StepArgs)


class ModelReport(BaseModel):
    """What the model writes at the end of the loop."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(max_length=600)
    iocs: list[Claim] = Field(default_factory=list, max_length=15)
    techniques: list[Claim] = Field(default_factory=list, max_length=15)
    behaviors: list[Claim] = Field(default_factory=list, max_length=10)
    prompt_injection_alerts: list[str] = Field(default_factory=list, max_length=10)


class YaraOutcome(BaseModel):
    rule: str = ""
    validated: bool = False
    reason: str = ""
    attempts: int = 0
    fallback: bool = False


class GroundingStats(BaseModel):
    proposed: int = 0
    kept: int = 0
    dropped: int = 0


class DroppedClaim(BaseModel):
    category: str
    claim: Claim
    reason: str


class Report(ModelReport):
    """Final report written to disk (model report + code-attached fields)."""

    sample: str
    sha256: str
    model: str
    guard: bool
    yara: YaraOutcome | None = None
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    grounding: GroundingStats = Field(default_factory=GroundingStats)
    verdict_override: str | None = None
    tool_shortfall: str | None = None
    tools_succeeded: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def _inline_refs(schema: dict) -> dict:
    """Return a copy of a Pydantic JSON schema with every $ref inlined.

    Ollama converts the schema to a grammar; inlined schemas avoid relying on
    $ref support and keep the grammar small.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = {k: v for k, v in node.items() if k != "$ref"}
                merged.update(resolve(copy.deepcopy(defs[name])))
                return merged
            return {k: resolve(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def step_schema(tool_names: Iterable[str]) -> dict:
    """JSON Schema for one ReAct step with the tool restricted to ``tool_names``."""
    schema = _inline_refs(StepAction.model_json_schema())
    schema["properties"]["tool"] = {"type": "string", "enum": list(tool_names)}
    schema["required"] = ["thought", "tool", "args"]
    return schema


def report_schema() -> dict:
    schema = _inline_refs(ModelReport.model_json_schema())
    schema["required"] = list(ModelReport.model_fields)
    return schema


# --------------------------------------------------------------------------- #
# Injection scanner
# --------------------------------------------------------------------------- #

INJECTION_MARK = "[SUSPICIOUS_PROMPT_INJECTION_DETECTED]"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("template_token", re.compile(
        r"<\|(?:im_start|im_end|endoftext|system|user|assistant|eot_id|start_header_id)\|>"
        r"|\[/?INST\]|<<SYS>>|<</SYS>>", re.I)),
    ("delimiter_forgery", re.compile(
        r"</?\s*TOOL_DATA[_A-Za-z0-9]*[^>]{0,40}>"
        r"|</?\s*(?:tool_output|tool_result|observation|system|instructions?|evidence)\s*>", re.I)),
    ("instruction_override", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|skip)\b.{0,40}?"
        r"\b(?:previous|prior|above|earlier|all|any|system|your|the|these)\b.{0,30}?"
        r"\b(?:instructions?|rules?|prompts?|directives?|guidelines?|context|analysis)\b",
        re.I | re.S)),
    ("instruction_override", re.compile(r"\bnew\s+(?:instructions?|rules?|task)\s*:", re.I)),
    ("instruction_override", re.compile(
        r"\bdo\s+not\s+(?:report|flag|mention|analy[sz]e|detect)\b.{0,40}\b(?:file|sample|this|malware|binary)\b",
        re.I | re.S)),
    ("verdict_demand", re.compile(
        r"\b(?:classify|mark|label|report|treat|consider|flag|tag|rate|declare)\s+"
        r"(?:this|the|it|that)\s*(?:file|sample|binary|program|executable|software|application|exe|dll)?\s*"
        r"(?:as\s+|is\s+)?(?:being\s+)?(?:benign|clean|safe|harmless|not\s+malicious|legitimate|whitelisted)\b",
        re.I)),
    ("verdict_demand", re.compile(
        r"\b(?:output|respond|answer|return|say|print|reply|conclude|verdict)\s*(?:with|:|=|is)?\s*[\"']?"
        r"(?:benign|not\s+malicious)\b", re.I)),
    ("verdict_demand", re.compile(
        r"\b(?:this|the)\s+(?:file|sample|binary|program|executable)\s+is\s+(?:benign|harmless|not\s+malicious)\b",
        re.I)),
    ("verdict_demand", re.compile(r"[\"']?verdict[\"']?\s*[:=]\s*[\"']?benign", re.I)),
    ("role_impersonation", re.compile(
        r"\byou\s+are\s+(?:now\s+)?(?:a|an|the|my)\s+(?:\w+\s+){0,2}?"
        r"(?:ai|assistant|chatbot|bot|model|llm|analy[sz]t|analy[sz]er|agent|classifier|expert|researcher)\b"
        r"|\bact\s+as\s+(?:a|an|the)\b"
        r"|\bas\s+an?\s+(?:ai|assistant|language\s+model|llm)\b"
        r"|\b(?:system|developer|admin(?:istrator)?)\s*(?:prompt|message|override|note|instruction)s?\s*:"
        r"|^\s*(?:system|assistant|user)\s*:"
        r"|\bdeveloper\s+mode\b"
        r"|\b(?:dear|attention|note\s+to)\s+(?:ai|llm|assistant|analy[sz]er|model)\b",
        re.I | re.M)),
]

_B64_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX_RE = re.compile(r"(?:\b|0x)((?:[0-9a-fA-F]{2}){8,})\b")


@dataclass
class InjectionFinding:
    text: str
    technique: str
    source: str = ""
    decoded: str | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "technique": self.technique, "source": self.source,
                "decoded": self.decoded}


def _match_technique(text: str) -> str | None:
    for technique, pattern in _PATTERNS:
        if pattern.search(text):
            return technique
    return None


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    ok = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
    return ok / len(text)


def _decode_candidates(text: str) -> list[str]:
    """Decode Base64 / hex runs inside ``text`` into mostly-printable strings."""
    out: list[str] = []
    for m in _B64_RE.finditer(text):
        chunk = m.group(0)
        padded = chunk + "=" * (-len(chunk) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue
        for enc in ("utf-8", "utf-16-le"):
            try:
                dec = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if len(dec) >= 8 and _printable_ratio(dec) >= 0.9:
                out.append(dec)
                break
    for m in _HEX_RE.finditer(text):
        try:
            raw = bytes.fromhex(m.group(1))
        except ValueError:
            continue
        for enc in ("utf-8", "utf-16-le"):
            try:
                dec = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if len(dec) >= 8 and _printable_ratio(dec) >= 0.9:
                out.append(dec)
                break
    return out


def looks_like_injection(text: str) -> bool:
    """Cheap check used by the string ranker: plain or encoded injection."""
    if _match_technique(text):
        return True
    return any(_match_technique(d) for d in _decode_candidates(text))


def scan_strings(strings: Iterable[str], source: str = "", window: int = 3) -> list[InjectionFinding]:
    """Scan a sequence of strings for injection indicators.

    Each string is checked on its own, then decoded Base64/hex runs are checked,
    then joins of up to ``window`` adjacent strings are checked to catch an
    instruction split across several strings.
    """
    items = [s for s in strings if isinstance(s, str) and s.strip()]
    findings: list[InjectionFinding] = []
    seen: set[str] = set()
    hit_idx: set[int] = set()

    def add(f: InjectionFinding) -> None:
        if f.text not in seen:
            seen.add(f.text)
            findings.append(f)

    for i, s in enumerate(items):
        tech = _match_technique(s)
        if tech:
            add(InjectionFinding(s, tech, source))
            hit_idx.add(i)
            continue
        for dec in _decode_candidates(s):
            if _match_technique(dec):
                add(InjectionFinding(s, "encoded_instruction", source, decoded=dec))
                hit_idx.add(i)
                break

    for size in range(2, window + 1):
        for i in range(0, len(items) - size + 1):
            idx = set(range(i, i + size))
            if idx & hit_idx:
                continue
            parts = items[i:i + size]
            if any(len(p) < 3 for p in parts):
                continue
            joined = " ".join(p.strip() for p in parts)
            if _match_technique(joined):
                add(InjectionFinding(joined, "split_instruction", source))
                hit_idx |= idx
    return findings


# --------------------------------------------------------------------------- #
# Escaping, flagging, spotlighting
# --------------------------------------------------------------------------- #

_TEMPLATE_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
_TOOL_TAG_RE = re.compile(r"<(\s*/?\s*TOOL_DATA)", re.I)


def escape_template_tokens(text: str) -> str:
    """Break chat-template special tokens so Ollama cannot parse them as specials."""
    return _TEMPLATE_TOKEN_RE.sub(r"< |\1| >", text)


def escape_delimiters(text: str) -> str:
    """Neutralize any TOOL_DATA tag inside tool output so it cannot close the wrapper."""
    return _TOOL_TAG_RE.sub(r"&lt;\1", text)


def flag_text(text: str, findings: list[InjectionFinding]) -> str:
    """Prefix every line that carries an injection finding with the marker."""
    if not findings:
        return text
    keys = []
    for f in findings:
        for piece in re.split(r"\s{2,}|\n", f.text):
            piece = piece.strip()
            if len(piece) >= 6:
                keys.append(piece[:40].casefold())
    out = []
    for line in text.splitlines():
        low = line.casefold()
        if _match_technique(line) or any(k in low for k in keys):
            if not line.startswith(INJECTION_MARK):
                line = f"{INJECTION_MARK} {line}"
        out.append(line)
    encoded = [f for f in findings if f.technique == "encoded_instruction"]
    if encoded:
        out.append(f"{INJECTION_MARK} {len(encoded)} encoded string(s) decode to instructions aimed at the analyst.")
    return "\n".join(out)


class Spotlight:
    """Per-run random delimiter wrapper."""

    def __init__(self, suffix: str | None = None) -> None:
        self.suffix = suffix or secrets.token_hex(2)

    @property
    def tag(self) -> str:
        return f"TOOL_DATA_{self.suffix}"

    def wrap(self, tool: str, text: str) -> str:
        return f'<{self.tag} tool="{tool}">\n{text}\n</{self.tag}>'


# --------------------------------------------------------------------------- #
# Guard: the per-run pipeline applied to tool results
# --------------------------------------------------------------------------- #

@dataclass
class Guard:
    enabled: bool
    spotlight: Spotlight = field(default_factory=Spotlight)
    findings: list[InjectionFinding] = field(default_factory=list)

    def observe(self, tool: str, text: str, scan_items: Iterable[str]) -> str:
        """Turn a tool's model-facing text into the Observation shown to the model."""
        text = escape_template_tokens(text)
        if not self.enabled:
            return text
        new = scan_strings(list(scan_items), source=tool)
        # lines of the display text catch anything the tool did not list for scanning;
        # skip lines that only repeat a finding already made from the scan items
        for f in scan_strings(text.splitlines(), source=tool):
            if not any(o.text in f.text or f.text in o.text for o in new):
                new.append(f)
        known = [o.text for o in self.findings]
        for f in new:
            if not any(k in f.text or f.text in k for k in known):
                self.findings.append(f)
                known.append(f.text)
        text = escape_delimiters(text)
        text = flag_text(text, new)
        return self.spotlight.wrap(tool, text)

    def wrap_only(self, tool: str, text: str) -> str:
        text = escape_template_tokens(text)
        if not self.enabled:
            return text
        return self.spotlight.wrap(tool, escape_delimiters(text))

    @property
    def alerts(self) -> list[str]:
        return [f.text for f in self.findings]

    def check_args(self, args: StepArgs) -> str | None:
        """Return a block reason if any argument carries a flagged injection string."""
        if not self.enabled:
            return None
        flagged = [a.casefold() for a in self.alerts]
        for value in [*args.strings, *args.addresses]:
            v = value.casefold().strip()
            if not v:
                continue
            if _match_technique(value) or any(v in f or f in v for f in flagged if len(v) >= 6):
                return f"argument blocked: '{value[:80]}' matches a flagged prompt-injection string"
        return None


# --------------------------------------------------------------------------- #
# Extractive grounding
# --------------------------------------------------------------------------- #

_TECH_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)


def normalize(text: str) -> str:
    t = text.casefold()
    t = t.replace("hxxps", "https").replace("hxxp", "http")
    t = t.replace("[.]", ".").replace("(.)", ".").replace("[:]", ":").replace("[://]", "://")
    t = t.replace("\\\\", "\\")
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip("\"'` ")


def _json_leaves(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            out.append(str(k))
            _json_leaves(v, out)
    elif isinstance(node, list):
        for v in node:
            _json_leaves(v, out)
    elif node is not None:
        out.append(str(node))


def build_corpus(raw_outputs: Iterable[str]) -> str:
    """Normalized searchable text of everything the tools produced in a run.

    JSON raw outputs are also flattened into their string leaves, so a quote
    of ``HKCU\\Software`` matches the JSON-escaped ``HKCU\\\\Software``.
    """
    parts: list[str] = []
    for raw in raw_outputs:
        if not raw:
            continue
        parts.append(raw)
        stripped = raw.lstrip()
        if stripped[:1] in "{[":
            try:
                leaves: list[str] = []
                _json_leaves(json.loads(raw), leaves)
                parts.append("\n".join(leaves))
            except (ValueError, RecursionError):
                pass
    return normalize("\n".join(parts))


_LEADING_TAGS_RE = re.compile(r"^(?:\[[^\]]{1,40}\]\s*)+")
_LEADING_LABEL_RE = re.compile(
    r"^(?:url|ip|domain|registry|path|command|email|mutex|crypto_ransom|injection_like|file drop \(executable\)|"
    r"file write|persistence|process|dns|http|service)\s*:\s*", re.I)


def quote_variants(quote: str) -> list[str]:
    """The normalized quote, and the same quote without display labels the agent adds to tool output
    ("[static]", "[decoded] url:", the injection marker). The rest must still match tool output verbatim."""
    q = normalize(quote)
    out = [q]
    stripped = _LEADING_LABEL_RE.sub("", _LEADING_TAGS_RE.sub("", q.replace(normalize(INJECTION_MARK), "")).strip())
    if stripped and stripped != q:
        out.append(stripped)
    return out


def quote_in_corpus(quote: str, corpus: str) -> bool:
    return any(v and v in corpus for v in quote_variants(quote))


def technique_id(value: str) -> str | None:
    m = _TECH_RE.search(value)
    return m.group(0).upper() if m else None


@dataclass
class GroundingResult:
    report: ModelReport
    dropped: list[DroppedClaim]
    stats: GroundingStats


def ground_report(report: ModelReport, corpus: str, attack_ids: Iterable[str]) -> GroundingResult:
    """Keep only claims whose value and evidence quote appear verbatim in tool output."""
    ids = {i.upper() for i in attack_ids}
    parents = {i.split(".")[0] for i in ids}
    dropped: list[DroppedClaim] = []
    kept: dict[str, list[Claim]] = {"iocs": [], "techniques": [], "behaviors": []}
    proposed = 0
    for category in kept:
        for claim in getattr(report, category):
            proposed += 1
            reason = None
            quote = normalize(claim.evidence_quote)
            if not quote:
                reason = "empty evidence_quote"
            elif not quote_in_corpus(claim.evidence_quote, corpus):
                reason = "evidence_quote not found in tool output"
            elif category == "techniques":
                tid = technique_id(claim.value)
                if tid is None:
                    reason = "no ATT&CK ID in technique value"
                elif tid not in ids and tid not in parents:
                    reason = f"{tid} not reported by capa"
            elif category == "iocs" and normalize(claim.value) not in corpus:
                reason = "value not found in tool output"
            elif category == "behaviors" and not claim.value.strip():
                reason = "empty behavior"
            if reason:
                dropped.append(DroppedClaim(category=category, claim=claim, reason=reason))
            else:
                kept[category].append(claim)
    new_report = report.model_copy(update=kept)
    stats = GroundingStats(proposed=proposed, kept=proposed - len(dropped), dropped=len(dropped))
    return GroundingResult(new_report, dropped, stats)


def apply_verdict_lock(report: ModelReport, alerts: list[str]) -> tuple[ModelReport, str | None]:
    """With guard on: any injection alert means the verdict is MALICIOUS."""
    if alerts and report.verdict != "MALICIOUS":
        note = (f"verdict changed from {report.verdict} to MALICIOUS: "
                f"{len(alerts)} prompt-injection string(s) found in tool output")
        return report.model_copy(update={"verdict": "MALICIOUS"}), note
    return report, None


def merge_alerts(scanner_alerts: list[str], model_alerts: list[str], corpus: str) -> list[str]:
    """Scanner findings plus model-reported alerts that really appear in tool output."""
    out = list(dict.fromkeys(scanner_alerts))
    for a in model_alerts:
        if a and normalize(a) in corpus and a not in out:
            out.append(a)
    return out[:10]
