#!/bin/sh
mkdir -p /logs/verifier
if [ "$(cat /app/answer.txt 2>/dev/null)" = "42" ]; then c=1; else c=0; fi
echo "{\"correctness\": $c, \"style\": 0.5}" > /logs/verifier/reward.json
