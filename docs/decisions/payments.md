# Federation decision: claims_intake -> payments

Status: accepted

## Context
claims_intake (Claims) calls payments (Finance).

## The Federation Test
| Question | Answer | Decided by (caller / callee) |
|---|---|---|
| owner | yes | Claims / Finance |
| cadence | yes | claims-train / finance-train |
| trust_domain | no | meridian-idp / meridian-idp |
| data_boundary | yes | meridian-claims / meridian-payments |

## Decision
a2a_agent. A seam. Harden first: contract (Ch 4, 15), authority (Ch 5-8)

## Options refused
- subgraph: Finance deploys and releases its own runtime; a subgraph compiles it into ours
- mcp_tool: Finance runs its own state, people and task lifecycle behind the call

## Predicted costs
- re-sent per task: 4 fields (claim_id, amount, repair_estimate, payment_memo)
- latency: a queue we do not run (per-seam p99, Ch 14)
- checkpoint visibility: Finance's steps stay in Finance's checkpoint

## Shape
hub: Claims routes it

## Paged
finance-oncall

## Revisit when
Finance and Claims merge their release trains and data boundary.
