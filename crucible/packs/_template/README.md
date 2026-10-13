# __PACK_NAME__ (a Crucible pack)

One workspace task: write `greeting.txt` with a seeded greeting. It needs no Docker and no model.

```bash
crucible pack check packs/__PACK_NAME__                    # static: manifest, placeholders, entry point, hooks
crucible pack check packs/__PACK_NAME__ --level semantic   # gold reaches the top reward, the untouched task does not
crucible pack check packs/__PACK_NAME__ --level accept     # the M9 Acceptance Contract: provisional (A10 needs paid frontier runs)
crucible bench run packs/__PACK_NAME__                     # a model-free benchmark report (scripted gold and no-op)
```

Next steps:
- Replace the task in `src/__PACK_PKG__/studio.py` with your domain's Worlds, and keep its acceptance capabilities true
  (stated properties, checks with provenance, designed negatives with a naive floor and a valid alternative, attacks and
  tamper attempts with effect probes, the containment audit and a property audit): a missing one fails `--level accept`.
- Add `ingest:` sources to `pack.yaml` so `crucible label` can turn your workflow logs into datasets.
