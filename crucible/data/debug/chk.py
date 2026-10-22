from pathlib import Path
from crucible.factory.langs import get

n = get('node')
for r in ("react-hook-form__react-hook-form", "honojs__hono", "toss__es-toolkit", "pmndrs__jotai", "cheeriojs__cheerio", "immerjs__immer"):
    t = Path('data/debug/src') / r
    print(r, n.image_info(t)["runner"], n.run_script(t, ['x.test.ts'])[-140:])
