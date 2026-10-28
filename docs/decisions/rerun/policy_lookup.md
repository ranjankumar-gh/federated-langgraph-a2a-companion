# Federation rerun: claims_intake -> policy_lookup

Reruns: ../policy_lookup.md, with the numbers its seam measured end to end.

## Option
subgraph: holds. The four answers come from the same owner facts.

## Shape
Recorded: subgraph inside the Claims graph

Holds. Not a seam: nothing crosses, and nothing is measured.

## Predicted against measured
| Line | Predicted | Measured |
|---|---|---|
| re-sent per task | 2 fields (claim_id, damaged_parts) | none: the subgraph reads the claim in place |
| request body (CLM-1001) | 306 bytes | 306 bytes as the incident deployed it (Chapter 2's test, rerun in the same suite); none as a subgraph |

## Still open
- nothing

## Finding
The record stands, with the lines above added.
