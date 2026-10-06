#!/bin/bash
# Verifier entry point: applies the agent's patch to a pristine copy and grades it.
set -u
mkdir -p /logs/verifier /logs/artifacts
WORK=$(mktemp -d)
cp -r /opt/originals/mdlite "$WORK/mdlite"
cd "$WORK/mdlite"
git init -q && git add -A && git -c user.name=v -c user.email=v@v commit -qm base
if [ -s /logs/artifacts/model.patch ]; then
  git apply --whitespace=nowarn /logs/artifacts/model.patch || { echo '{"reward": 0.0, "error": "patch does not apply"}' > /logs/verifier/report.json; echo 0 > /logs/verifier/reward.txt; exit 0; }
fi
python /tests/grade.py --candidate "$WORK/mdlite" --originals /opt/originals --out /logs/verifier/report.json
python -c "import json; print(json.load(open('/logs/verifier/report.json'))['reward'])" > /logs/verifier/reward.txt
cat /logs/verifier/reward.txt
