# Federation rerun: claims_intake -> payments

Reruns: ../payments.md, with the numbers its seam measured end to end.

## Option
a2a_agent: holds. The four answers come from the same owner facts.

## Shape
Recorded: hub: Claims routes it

Holds. The hub sends it, as the record says.

## Predicted against measured
| Line | Predicted | Measured |
|---|---|---|
| re-sent per task | 4 fields (claim_id, amount, repair_estimate, payment_memo) | 4 fields (claim_id, amount, repair_estimate, payment_memo): 545 bytes a request, about 36 tokens |
| latency | a queue we do not run (per-seam p99, Ch 14) | an in-process floor, printed by the run and never committed |
| checkpoint visibility | Finance's steps stay in Finance's checkpoint | Finance's steps in Finance's checkpoint; the task in Claims' ledger |
| authority | (not predicted) | a 2-link chain, 1220 bytes a request |
| receipts | (not predicted) | 3 for 1 task, in Claims' log |
| cost per task | (not predicted) | nothing priced on any cost ledger |

## Still open
- budget waived: Finance states no price per payment; Claims pays the claim, not the call
- evidence waived: Finance's terms state none; the payment ledger is Finance's book of record

## Finding
The record stands, with the lines above added.
