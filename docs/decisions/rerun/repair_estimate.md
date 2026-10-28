# Federation rerun: claims_intake -> repair_estimate

Reruns: ../repair_estimate.md, with the numbers its seam measured end to end.

## Option
a2a_agent: holds. The four answers come from the same owner facts.

## Shape
Recorded: hub: Claims routes it

Holds. The hub sends it, as the record says.

## Predicted against measured
| Line | Predicted | Measured |
|---|---|---|
| re-sent per task | 2 fields (claim_id, damaged_parts) | 2 fields (claim_id, damaged_parts): 524 bytes a request, about 15 tokens |
| latency | a queue we do not run (per-seam p99, Ch 14) | an in-process floor, printed by the run and never committed |
| checkpoint visibility | Northgate's steps stay in Northgate's checkpoint | Northgate's steps in Northgate's checkpoint; the task in Claims' ledger |
| authority | (not predicted) | a 2-link chain, 1164 bytes a request |
| receipts | (not predicted) | 1 for 1 task, in Claims' log |
| cost per task | (not predicted) | 3.50 on Claims' cost ledger |

## Still open
- nothing

## Finding
The record stands, with the lines above added.
