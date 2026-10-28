# Federation decision: claims_intake -> fraud_screening

Status: accepted

## Context
claims_intake (Claims) calls fraud_screening (Risk).

## The Federation Test
| Question | Answer | Decided by (caller / callee) |
|---|---|---|
| owner | yes | Claims / Risk |
| cadence | yes | claims-train / risk-monthly |
| trust_domain | no | meridian-idp / meridian-idp |
| data_boundary | yes | meridian-claims / meridian-risk |

## Decision
a2a_agent. A seam. Harden first: contract (Ch 4, 15), authority (Ch 5-8)

## Options refused
- subgraph: Risk deploys and releases its own runtime; a subgraph compiles it into ours
- mcp_tool: Risk runs its own state, people and task lifecycle behind the call

## Predicted costs
- re-sent per task: 4 fields (claim_id, amount, days_to_report, prior_claims)
- latency: a queue we do not run (per-seam p99, Ch 14)
- checkpoint visibility: Risk's steps stay in Risk's checkpoint

## Shape
hub: Claims routes it

## Paged
risk-oncall

## Revisit when
Risk moves onto the Claims release train, or Claims takes over the fraud model.
