# Federation rerun: fraud_screening -> estimate_history

Reruns: ../estimate_history.md, with the numbers its seam measured end to end.

## Option
a2a_agent: holds. The four answers come from the same owner facts.

## Shape
Recorded: peer: Risk's screening calls Northgate directly; the task is in Risk's record, not Claims'

Fails. Only claim_id crossed out and shop and estimates came back, none of fraud_signals. Which claims were sent was a model's choice, so the choosing is the one signal that would cross; a hub that asks for every claim it screens carries none.

## Predicted against measured
| Line | Predicted | Measured |
|---|---|---|
| re-sent per task | 1 field (claim_id) | 1 field (claim_id): 430 bytes a request, about 6 tokens |
| latency | a queue we do not run (per-seam p99, Ch 14) | an in-process floor, printed by the run and never committed |
| checkpoint visibility | Northgate's steps stay in Northgate's checkpoint | Northgate's steps in Northgate's checkpoint; the task in no ledger: Risk's crossing records only |
| authority | (not predicted) | a 3-link chain, 1708 bytes a request |
| receipts | (not predicted) | 2 for 2 tasks, in Risk's log |
| cost per task | (not predicted) | nothing priced on any cost ledger |

## Still open
- ledger waived: a tool call has no thread to link; the task is in Risk's crossing records
- budget waived: Northgate prices a history read at 0.5; Risk budgets nothing per call

## Finding
Superseded: hub. Claims asks Northgate for the history of every claim it screens, before screening, and sends it to Risk as a fifth field under Risk's terms revision 2027.08. Meridian's data-class policy lets estimate data cross to Risk, and the adjuster's root names the field.

Measured on the same claims: 3 history tasks for 2 (one for every screened claim, at the 0.5 a task Northgate's terms state), a 2-link chain for 3, the task in Claims' ledger, and 86 more bytes and 20 more tokens on every screening request. Risk's assistant no longer chooses when to look.

Retired at Risk: its outbound gateway, its breaker on Northgate, the estimate_history entry on its ownership map, and both waivers.
