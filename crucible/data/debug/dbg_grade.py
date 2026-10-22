"""Debug: grade a stored window's oracle as steps and print the report's failing parts."""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import lh_debug  # noqa: E402

wid, split = sys.argv[1], sys.argv[2]


async def main():
    from crucible.factory.lanes.windows import WindowStore
    store = WindowStore(lh_debug.ROOT / "gen" / "store" / split, split=split)
    rec = store.window(wid)
    gen = lh_debug.generator([rec["repo"]], split)
    t = gen.task(wid)
    rep = await t.grade_patch(t.oracle, trail=[])
    m = rep.meta
    print("reward", rep.reward, "phi", m.get("phi"), "full", m.get("full_pass"), "end_full", m.get("end_state_full_pass"))
    print("gates", rep.gates)
    print("failing", m.get("failing_tests")[:10], "apply", m.get("apply_errors"), "protected", m.get("protected_writes"))
    print("path", json.dumps(m.get("path"))[:800])


asyncio.run(main())
