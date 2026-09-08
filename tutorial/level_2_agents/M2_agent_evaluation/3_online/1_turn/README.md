# L2-M2.3.1 — Online, Turn Scope

**One production trace, judged on its own.**

The judge answers *"was this reply good?"* about a single request and its
response. It never sees the turn before or after.

## The two lessons

| # | Lesson | Unit |
|:--|:--|:--|
| 1 | [`1_online_scoring`](1_online_scoring/) | one trace |
| 2 | [`2_live_ab_testing`](2_live_ab_testing/) | one trace, two arms |

It covers registering a `make_judge` judge, starting it against a gateway
endpoint, setting and changing the sample rate, and reading the scorer's
server-side state back.

## Why it must be a registered judge

An inline `@scorer` function is `DECORATOR` kind. It deserialises via `exec()`,
so an open-source tracking server refuses to store it — which means it can never
run server-side, which means it can never run online at all. `make_judge`
produces `INSTRUCTIONS` kind, and that registers.

Built-in scorers are `BUILTIN` kind and register too. That is what makes
[`../2_conversation/`](../2_conversation/) possible.

## What this scope cannot ask

Whether the customer behind those traces actually got helped. That needs the
whole session — [`../2_conversation/`](../2_conversation/).
