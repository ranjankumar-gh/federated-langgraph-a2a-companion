# Federation decision: fraud_screening -> estimate_history

Status: accepted

## Context
fraud_screening (Risk) calls estimate_history (Northgate).

## The Federation Test
| Question | Answer | Decided by (caller / callee) |
|---|---|---|
| owner | yes | Risk / Northgate |
| cadence | yes | risk-monthly / northgate |
| trust_domain | yes | meridian-idp / northgate-idp |
| data_boundary | yes | meridian-risk / northgate |

## Decision
a2a_agent. A seam. Harden first: contract (Ch 4, 15), authority (Ch 5-8)

## Options refused
- subgraph: Northgate deploys and releases its own runtime; a subgraph compiles it into ours
- mcp_tool: Northgate runs its own state, people and task lifecycle behind the call

## Predicted costs
- re-sent per task: 1 field (claim_id)
- latency: a queue we do not run (per-seam p99, Ch 14)
- checkpoint visibility: Northgate's steps stay in Northgate's checkpoint

## Shape
peer: Risk's screening calls Northgate directly; the task is in Risk's record, not Claims'

## Paged
risk-oncall

## Revisit when
Risk agrees to share its fraud signals with Claims (route it through the hub), or a second peer seam is proposed.
