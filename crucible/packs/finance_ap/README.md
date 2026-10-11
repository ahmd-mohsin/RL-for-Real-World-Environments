# finance_ap (a Crucible pack)

Accounts payable, the thesis MVP domain. Each seeded world holds pending invoices that must be decided under a
3-way-match policy (purchase order, goods receipt, invoice) with fraud controls: duplicate invoices, blocked vendors,
unverified bank-detail changes, a price tolerance that is drawn per world, and missing receipts.

The grade is computed from the ERP state:
- a **control gate:** approving an invoice that the policy should have stopped scores 0;
- **outcome:** the share of invoices decided correctly, with the policy's reason code.

Progress Φ moves per invoice.

```bash
crucible pack check packs/finance_ap --level semantic   # gold reaches 1.0, the untouched world does not; designed negatives grade exactly
crucible bench run packs/finance_ap --out runs          # oracle vs approve-everything vs no-op, every step metric
```
