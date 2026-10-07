#!/bin/bash
# Verifier entry point: applies the agent's patch to a pristine PCL tree and grades it.
set -u
mkdir -p /logs/verifier /logs/artifacts
fail() { echo "{\"reward\": 0.0, \"error\": \"$1\"}" > /logs/verifier/report.json; echo 0 > /logs/verifier/reward.txt; cat /logs/verifier/reward.txt; exit 0; }
[ -d /opt/legacy_rt ] || fail "verifier image lacks the original build (/opt/legacy_rt)"
WORK=$(mktemp -d)
cp -r /opt/originals/pcl "$WORK/pcl"
cd "$WORK/pcl"
git init -q && git add -A && git -c user.name=v -c user.email=v@v commit -qm base
if [ -s /logs/artifacts/model.patch ]; then
  git apply --whitespace=nowarn /logs/artifacts/model.patch || fail "patch does not apply"
fi
cd /
python3 /tests/grade.py --candidate "$WORK/pcl" --originals /opt/originals --out /logs/verifier/report.json
python3 -c "import json; print(json.load(open('/logs/verifier/report.json'))['reward'])" > /logs/verifier/reward.txt
rm -rf "$WORK"
cat /logs/verifier/reward.txt
