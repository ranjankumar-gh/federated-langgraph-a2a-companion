# Federation rerun: claims_intake -> fraud_screening

Reruns: ../fraud_screening.md, with the numbers its seam measured end to end.

## Option
a2a_agent: holds. The four answers come from the same owner facts.

## Shape
Recorded: hub: Claims routes it

Holds. The hub sends it, as the record says.

## Predicted against measured
| Line | Predicted | Measured |
|---|---|---|
| re-sent per task | 4 fields (claim_id, amount, days_to_report, prior_claims) | 4 fields (claim_id, amount, days_to_report, prior_claims): 544 bytes a request, about 19 tokens |
| latency | a queue we do not run (per-seam p99, Ch 14) | an in-process floor, printed by the run and never committed |
| checkpoint visibility | Risk's steps stay in Risk's checkpoint | Risk's steps in Risk's checkpoint; the task in Claims' ledger |
| authority | (not predicted) | a 2-link chain, 1236 bytes a request |
| receipts | (not predicted) | 3 for 3 tasks, in Claims' log |
| cost per task | (not predicted) | 0.67 on Claims' cost ledger |

## Still open
- nothing

## Finding
The record stands, with the lines above added.
