# synth run it_helpdesk / 20261029-160257-full2

| stage | tasks |
|---|---|
| proposed | 20 |
| oracle_executed | 19 |
| contract | 19 |
| refined | 19 |
| final: certified | 2 |
| final: dropped | 1 |
| final: provisional | 1 |
| final: rejected | 16 |

Tasks with a confirmed counterexample, by falsifier class (round 1 / any round):

- MUT: 4 / 6
- NULL: 2 / 2
- RED: 7 / 11
- TAINT: 6 / 7
- TWIN: 11 / 13

Repairs accepted: 21 on 10 tasks; refused 1; declined 7. ALT passed on 6 tasks.

Reasons:

- ALT:no_pass: 13
- TWIN:confirmed: 10
- MUT:upper: 9
- archive:reopened: 8
- RED:confirmed: 8
- MUT:suspects: 7
- MUT:confirmed: 4
- TAINT:confirmed: 4
- ALT:suspects: 2
- NULL:confirmed: 2
- ALT:confirmed: 1
- oracle:dead: 1

Confirmed defects (archived counterexamples):

- D9: 40
- D3: 34
- D4: 31
- D2: 29
- D8: 11
- D1: 8
- D5: 8
- D7: 2

Cost (USD):

- ALT: 0.22674
- ATK: 0.03384
- abstract: 0.00099
- adjudicate: 0.64191
- calibrate: 0.02086
- contract: 0.37087
- oracle: 0.16261
- propose: 0.01921
- repair: 0.46286
- total: 1.93989

| task | seed | status | rounds | repairs | confirmed (any round) | reasons |
|---|---|---|---|---|---|---|
| t01 | 0 | certified | 1 | 0 | {} |  |
| t02 | 0 | provisional | 3 | 2 | {'RED': 1, 'TWIN': 1} | MUT:upper=0.0950>0.05 |
| t03 | 0 | rejected | 1 | 0 | {'TWIN': 2} | TWIN:confirmed=2; MUT:upper=0.0570>0.05; ALT:no_pass |
| t04 | 0 | rejected | 3 | 2 | {'MUT': 33, 'RED': 6, 'TAINT': 1, 'TWIN': 3} | MUT:confirmed=13; RED:confirmed=3; TWIN:confirmed=2 |
| t05 | 0 | rejected | 1 | 0 | {} | MUT:upper=0.0570>0.05; ALT:no_pass; ALT:suspects=2 |
| t06 | 0 | rejected | 3 | 2 | {'RED': 3, 'TAINT': 1, 'TWIN': 2, 'MUT': 11} | ALT:no_pass; MUT:suspects=3 |
| t07 | 0 | rejected | 3 | 3 | {'MUT': 10, 'RED': 2} | RED:confirmed=1; MUT:suspects=1; archive:reopened=1 |
| t08 | 0 | rejected | 1 | 0 | {'TWIN': 1} | TWIN:confirmed=1; ALT:no_pass; archive:reopened=1 |
| t09 | 0 | rejected | 3 | 3 | {'TWIN': 3, 'RED': 6, 'MUT': 23} | MUT:confirmed=12; RED:confirmed=3; TWIN:confirmed=1 |
| t10 | 0 | dropped | 0 | 0 | {} | oracle:dead |
| t11 | 0 | rejected | 1 | 0 | {} | MUT:upper=0.0644>0.05; MUT:suspects=1 |
| t12 | 0 | rejected | 2 | 1 | {'RED': 4, 'TAINT': 4, 'TWIN': 4} | TAINT:confirmed=2; TWIN:confirmed=2; ALT:no_pass |
| t13 | 0 | certified | 1 | 0 | {} |  |
| t14 | 0 | rejected | 3 | 3 | {'TAINT': 2, 'TWIN': 4, 'RED': 4} | RED:confirmed=2; TWIN:confirmed=2; ALT:no_pass |
| t15 | 0 | rejected | 3 | 3 | {'RED': 9, 'TAINT': 3, 'MUT': 4, 'TWIN': 6} | MUT:confirmed=1; RED:confirmed=4; TAINT:confirmed=1 |
| t16 | 0 | rejected | 1 | 0 | {'RED': 1, 'TWIN': 3} | RED:confirmed=1; TWIN:confirmed=3; ALT:no_pass |
| t17 | 0 | rejected | 2 | 1 | {'RED': 5, 'TAINT': 2, 'NULL': 11, 'MUT': 22} | MUT:confirmed=11; NULL:confirmed=5; RED:confirmed=4 |
| t18 | 0 | rejected | 1 | 0 | {'RED': 1, 'NULL': 9, 'TWIN': 2} | NULL:confirmed=9; RED:confirmed=1; TWIN:confirmed=2 |
| t19 | 0 | rejected | 1 | 0 | {'TAINT': 1, 'TWIN': 2} | TAINT:confirmed=1; TWIN:confirmed=2; ALT:no_pass |
| t20 | 0 | rejected | 2 | 1 | {'TWIN': 1} | MUT:upper=0.0950>0.05; MUT:suspects=1 |
