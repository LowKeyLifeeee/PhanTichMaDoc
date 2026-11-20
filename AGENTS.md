# AGENTS.md

Read [README.md](README.md) in full before your first edit, command or commit in this repo. README is the single source of truth for safety rules, contribution rules and known pitfalls; this file only points there.

Hard guardrail: malware samples stay inside the isolated VM and are only read: never executed, uploaded or committed.

Return to the matching README section when you:

- touch a sample, log or anything under `data/`: section 1 (safety rules)
- change behaviour: section 9 (contribution rules), then the specs in `openspec/changes/add-malware-react-agent/`
- hit a tool, Ghidra or token-budget error: section 10 (known errors)
- add an environment variable: `docs/setup.md`
