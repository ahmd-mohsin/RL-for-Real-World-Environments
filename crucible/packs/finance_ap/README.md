# finance_ap (a Crucible pack)

Accounts payable, the thesis MVP domain. Each seeded world holds pending invoices that must be decided under the AP-3
3-way-match policy (purchase order, goods receipts, invoice) with fraud controls: blocked vendors, duplicate invoices
in the forms real duplicates take, purchase orders that do not exist or belong to another vendor, unverified
bank-detail changes, a price tolerance, and missing receipts.

Generator v3 (pack 0.3.0, task ids `@3`) draws per world:
- the number of invoices (7 to 14) and the base rates (45 to 70% of the drawn invoices trip no rule);
- the price tolerance (0.5 to 10%, inclusive at the bar) and the order of the three hold rules, both stated only in
  the policy document;
- compound cases (two rules apply, so the order decides), near misses (exactly at the tolerance, one cent over it,
  partial receipts that add up, an earlier same-number invoice that was rejected, the same number from another
  vendor), duplicate variants an exact match misses, and an infeasible match.

Every world also holds (campaign B7: over 2,000 v2 worlds, ignoring the hold order solved 57.8%, whole-unit prices
68%, amount-matched duplicates 54.5%): two compound cases on the adjacent pairs of its hold order, so only the world's
own order decides both; a variant duplicate billed at a restated amount; a price one cent over the bar that a whole-unit
comparison passes; and a payee the policy stops. In 85% of worlds that payee is a **vendor alias**, a blocked vendor
re-registered under a new id and name with the same bank account (AP-3 rule 1; the vendor master tool `list_vendors`
finds it). Over 2,000 v3 worlds no shortcut solves more than about 1/6 of worlds (any fixed hold order 16-18%, a
memorized 5% 14%, ignoring aliases 15%; whole units, amount-matched duplicates, another company's tolerance, the
reverse order and exact-match duplicates 0%).

Every label is the AP-3 policy applied to the generated data (`world.decide`), checked against an independent
implementation over 1,000 seeds in the tests.

The grade is computed from the ERP state:
- a **control gate:** approving an invoice that the policy should have stopped scores 0;
- an **outcome gate:** every invoice decided as AP-3 says, with the policy's reason code. The reward is 1 only when
  both gates pass; the report's `outcome` keeps the share decided correctly.

Progress Φ moves per invoice and ends at that share: a control failure drops it to 0. The tasks carry the M9
acceptance capabilities (stated properties, checks, negatives, mutants, answer-key strings, isolation audit, tamper
attempts, attacks, property audit) for `crucible pack check --level accept`.

`studio.adversaries()` lists the shortcut policies a sound grader holds down, each with an `expect_max` bound on its
mean reward: approve-everything, hold-everything, exact-match duplicates, duplicates only at the same amount, another
company's tolerance, whole-unit prices, the rules in reverse order, and paying a blocked vendor's alias.

```bash
crucible pack check packs/finance_ap --level semantic   # gold reaches 1.0, the untouched world does not; designed negatives grade exactly
crucible bench run packs/finance_ap --out runs          # oracle vs approve-everything vs no-op and the adversaries, every step metric
```
