# YIELD-1: throughput and yield of the synthesis pipeline (2026-10-29)

Data: `crucible/data/synth/yield/` (gitignored): `failures-m4-rsi1.json`, `probe-alt-1/`, `study-oracle-{1,2}/`,
`study-alt-1/`, `lemmas-1/`, `smoke-{1,2}/`, `yield-1/` (the arms), `speed-*/` (serial vs parallel).
Swarm only, no Qwen in any role, the student never called. Builders as in M4 (DeepSeek proposes, writes oracles and
repairs; GLM writes contracts); ALT Nemotron-3-Ultra ×2 + gpt-oss, ATK Nemotron-3-Super + gpt-oss, panel Nemotron-3-Ultra
+ gpt-oss, abstractor gpt-oss. Code: `synth/batch.py`, `synth/offload.py`, `synth/yieldkit.py`, `synth/failures.py`,
`synth.py batch | batch-report | failures | lemmas`.

<!-- RESULTS -->
