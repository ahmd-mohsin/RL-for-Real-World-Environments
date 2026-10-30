# synth run finance_ap / 20261029-160300-full2

| stage | tasks |
|---|---|
| proposed | 20 |
| oracle_executed | 20 |
| contract | 20 |
| refined | 20 |
| final: certified | 2 |
| final: provisional | 2 |
| final: rejected | 16 |

Tasks with a confirmed counterexample, by falsifier class (round 1 / any round):

- TAINT: 1 / 1

Repairs accepted: 0 on 0 tasks; refused 0; declined 1. ALT passed on 4 tasks.

Reasons:

- ALT:no_pass: 16
- MUT:upper: 8
- ALT:suspects: 4
- TAINT:confirmed: 1
- archive:reopened: 1

Confirmed defects (archived counterexamples):

- D5: 1

Cost (USD):

- ALT: 0.51016
- ATK: 0.0526
- abstract: 0.00053
- adjudicate: 0.12318
- calibrate: 0.02579
- contract: 0.21574
- oracle: 0.10491
- propose: 0.0202
- repair: 0.01769
- total: 1.07079

| task | seed | status | rounds | repairs | confirmed (any round) | reasons |
|---|---|---|---|---|---|---|
| t01 | 0 | rejected | 1 | 0 | {} | ALT:no_pass |
| t02 | 0 | certified | 1 | 0 | {} |  |
| t03 | 0 | provisional | 1 | 0 | {} | MUT:upper=0.0503>0.05 |
| t04 | 0 | rejected | 1 | 0 | {} | MUT:upper=0.0673>0.05; ALT:no_pass; ALT:suspects=3 |
| t05 | 0 | provisional | 1 | 0 | {} | MUT:upper=0.0570>0.05 |
| t06 | 1 | rejected | 1 | 0 | {'TAINT': 1} | TAINT:confirmed=1; ALT:no_pass; archive:reopened=1 |
| t07 | 1 | rejected | 1 | 0 | {} | ALT:no_pass; ALT:suspects=1 |
| t08 | 1 | rejected | 1 | 0 | {} | ALT:no_pass; ALT:suspects=2 |
| t09 | 1 | rejected | 1 | 0 | {} | MUT:upper=0.0658>0.05; ALT:no_pass |
| t10 | 1 | rejected | 1 | 0 | {} | ALT:no_pass |
| t11 | 2 | rejected | 1 | 0 | {} | ALT:no_pass |
| t12 | 2 | rejected | 1 | 0 | {} | MUT:upper=0.0658>0.05; ALT:no_pass |
| t13 | 2 | rejected | 1 | 0 | {} | ALT:no_pass |
| t14 | 2 | rejected | 1 | 0 | {} | ALT:no_pass |
| t15 | 2 | rejected | 1 | 0 | {} | ALT:no_pass; ALT:suspects=2 |
| t16 | 3 | rejected | 1 | 0 | {} | MUT:upper=0.0550>0.05; ALT:no_pass |
| t17 | 3 | rejected | 1 | 0 | {} | MUT:upper=0.0618>0.05; ALT:no_pass |
| t18 | 3 | rejected | 1 | 0 | {} | ALT:no_pass |
| t19 | 3 | certified | 1 | 0 | {} |  |
| t20 | 3 | rejected | 1 | 0 | {} | MUT:upper=0.0582>0.05; ALT:no_pass |
