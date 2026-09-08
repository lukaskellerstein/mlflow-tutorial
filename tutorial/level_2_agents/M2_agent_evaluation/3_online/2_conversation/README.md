# L2-M2.3.2 — Online, Conversation Scope

**One production session, judged whole.**

The judge answers *"did this customer get helped?"* — a question no single trace
can answer, asked of traffic you did not choose.

## The two lessons

| # | Lesson | Unit |
|:--|:--|:--|
| 1 | [`1_online_session_scoring`](1_online_session_scoring/) | one session |
| 2 | [`2_live_ab_testing`](2_live_ab_testing/) | one session, two arms |

## What actually changes from turn scope

Very little in the code, and that is the point. Same registration, same
`start()`, same sampling machinery. `Scorer.is_session_level_scorer` is what
makes the server hand the judge a **list of traces sharing a session id**
instead of a single trace.

Built-in session scorers are `BUILTIN` kind, so they register against an
open-source tracking server — verified, not assumed.

## What changes outside the code

**Cost.** A turn judge reads one trace. A session judge reads every turn of the
conversation, so its prompt — and its bill — grow with conversation length.
Sample lower here than you would at turn level.

**Latency to signal.** A session cannot be scored until it looks finished, so
session-level findings arrive later than turn-level ones. Turn scoring tells you
a reply was bad within minutes; session scoring tells you a customer was failed,
and only once they stopped talking.

Run both. They answer different questions and neither substitutes for the other.
