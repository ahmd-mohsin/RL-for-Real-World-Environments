# __PACK_NAME__ (a Crucible pack)

One workspace task: write `greeting.txt` with a seeded greeting. It needs no Docker and no model.

```bash
crucible pack check packs/__PACK_NAME__                    # static: manifest, placeholders, entry point, hooks
crucible pack check packs/__PACK_NAME__ --level semantic   # gold reaches the top reward, the untouched task does not
crucible bench run packs/__PACK_NAME__                     # a model-free benchmark report (scripted gold and no-op)
```

Next steps:
- Replace the task in `src/__PACK_PKG__/studio.py` with your domain's Worlds.
- Add `ingest:` sources to `pack.yaml` so `crucible label` can turn your workflow logs into datasets.
