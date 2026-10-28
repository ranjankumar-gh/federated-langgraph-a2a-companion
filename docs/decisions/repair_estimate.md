# Federation decision: claims_intake -> repair_estimate

Status: accepted

## Context
claims_intake (Claims) calls repair_estimate (Northgate).

## The Federation Test
| Question | Answer | Decided by (caller / callee) |
|---|---|---|
| owner | yes | Claims / Northgate |
| cadence | yes | claims-train / northgate |
| trust_domain | yes | meridian-idp / northgate-idp |
| data_boundary | yes | meridian-claims / northgate |

## Decision
a2a_agent. A seam. Harden first: contract (Ch 4, 15), authority (Ch 5-8)

## Options refused
- subgraph: Northgate deploys and releases its own runtime; a subgraph compiles it into ours
- mcp_tool: Northgate runs its own state, people and task lifecycle behind the call

## Predicted costs
- re-sent per task: 2 fields (claim_id, damaged_parts)
- latency: a queue we do not run (per-seam p99, Ch 14)
- checkpoint visibility: Northgate's steps stay in Northgate's checkpoint

## Shape
hub: Claims routes it

## Paged
northgate-support, via the partner manager

## Revisit when
Meridian buys Northgate or brings estimating in-house.
