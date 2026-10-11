# M1 taxonomy study (2026-10-12)

What the local RL-environment corpus contains, and what Crucible's World can and cannot express. Read `07` first.

| File | Contents |
|---|---|
| `00-crucible-inventory.md` | Crucible's World capabilities today, from the code |
| `01-harbor-swe-ecosystem.md` | Harbor (86 adapters in 17 families), SWE-smith, mini-swe-agent, Repo2RLEnv, GitSwarm, autoresearch |
| `02-openenv-fineenvs.md` | OpenEnv (41 environments), FineEnvs, the OpenEnv interface contract, the multi-harness design |
| `03-prime-intellect.md` | verifiers v0 and v1, prime-environments (109), research-environments (106), prime-agent |
| `04-nemo-gym-enterprise-mimo-karotte.md` | NeMo-Gym (145 servers in 18 families) and its protocol, EnterpriseOps-Gym, MiMo v2.6, Karotte |
| `05-trainer-env-interfaces.md` | How 11 RL trainers connect to environments, and the 26 requirements for plugging into all of them |
| `06-articles-and-datasets.md` | Lessons from the 07 articles and the 08 dataset cards; 81 requirements (MUST, SHOULD, IDEA) |
| `07-capability-matrix.md` | The merged matrix: 22 families, 20 primitives with owner modules, article MUSTs against Crucible, code to port |

Studies 01–06 were read-only subagent passes over the code, with file:line evidence; 00 and 07 were written from them
and from Crucible's source.
