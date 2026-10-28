# Federation decision: claims_intake -> policy_lookup

Status: refused as a seam

## Context
claims_intake (Claims) calls policy_lookup (Claims).

## The Federation Test
| Question | Answer | Decided by (caller / callee) |
|---|---|---|
| owner | no | Claims / Claims |
| cadence | yes | claims-train / claims-policy-pipeline |
| trust_domain | no | meridian-idp / meridian-idp |
| data_boundary | no | meridian-claims / meridian-claims |

## Decision
subgraph. Not a seam: no owner line.
Warning: cadence differs. Under one owner that is a deployment or access
problem, not a reason for a seam.

## Options refused
- mcp_tool: it has steps worth checkpointing, and they belong in our checkpoint
- a2a_agent: one owner (Claims) on both sides: a seam's costs, none of its reasons

## Predicted costs
- re-sent per task: 2 fields (claim_id, damaged_parts)
- latency: a round trip per claim instead of a function call
- request body (CLM-1001): 310 bytes
- lost from the Claims checkpoint: the policy revision read, and 4 subgraph checkpoints

## Shape
subgraph inside the Claims graph

## Paged
claims-oncall

## Revisit when
another owner calls the policy lookup, or the policy administration team takes it over.
