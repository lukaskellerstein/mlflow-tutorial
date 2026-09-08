#!/usr/bin/env python
"""Seed the MLflow AI Gateway — every alias this tutorial calls.

This file is BOTH the configuration and the code that applies it. There used to
be a `models.yaml` beside it; the split bought nothing. Nothing else read the
YAML, the file was 60% comments explaining the script, and a reader had to hold
two files open to answer one question. One file, one answer.

There is one gateway now, and it is the MLflow server itself:

    http://127.0.0.1:5555/gateway/mlflow/v1/chat/completions

MLflow has no gateway config file of its own — its gateway lives in the tracking
database and arrives over an API. So compose runs this script after the MLflow
server reports healthy, on every `up -d`.

It is idempotent. Existing model definitions and endpoints are reused; the secret
is rewritten on each run, so a rotated key reaches the gateway with no manual
step.

ADDING an alias below needs only `podman compose up -d`. CHANGING one needs a
rebuild, because an existing endpoint is reported as `reused` and quietly keeps
the model it already had:

    podman compose run --rm mlflow-seed --reset --prune

---- One provider: Unsloth Studio, on this machine ---------------------------

Every alias resolves to a local model. There is no OpenRouter, no OpenAI, no
hosted anything — no key to leak, no bill to watch, no network needed, and no
way for a lesson to quietly run somewhere other than where it says it runs.

That last one is the real win, and it used to be a live problem. The local
aliases carried an error fallback to OpenRouter, so an unloaded model did not
fail the call — it silently answered from a DIFFERENT model. A sweep whose
independent variable can change without telling you is worse than one that
crashes. Now a lesson that cannot reach Unsloth fails, says so, and you fix the
one real cause.

---- The three objects -------------------------------------------------------

LiteLLM, which this replaced, kept an alias, its model and its credential in ONE
entry. MLflow splits that into three, and the split is why 8 aliases need only 3
model definitions — five aliases share one local model instead of repeating it.

    secret            the credential + base URL      one per provider account
      |
    model definition  provider + the real model id   one per DISTINCT model
      |
    endpoint          the alias a caller names       one per alias

---- Two things about Unsloth that are load-bearing --------------------------

  1. `Settings -> API -> Model auto-switch` MUST be on. Unsloth holds ONE model
     at a time, and with auto-switch off every alias but the loaded one fails
     with `400 ... 'Switch model by request' is off`. With it on, a call to
     another alias unloads the current model and loads the new one — 4-14 s once
     the file is in the page cache. Every alias below is reachable in one
     session because of this setting, and only because of it.
  2. It needs a key on EVERY route, `/v1/models` included. `UNSLOTH_API_KEY`
     comes from the environment; leave it blank and this script seeds nothing
     rather than building endpoints that 401 at call time.

Unsloth is also why this repo no longer strips `response_format`. LMStudio
honoured a JSON schema by compiling a decoding grammar that masks EOS, so a
judge that missed a closing quote rambled to max_tokens — which is what the old
`additional_drop_params: ["response_format"]` line existed to prevent. Unsloth
returns clean JSON with `finish_reason: stop` for the same request (verified
2026-08-27), so the workaround is gone and every structured-output path works,
including `mlflow.genai.simulators.generate_test_cases()`.

---- What this gateway cannot do ---------------------------------------------

    max_input_tokens          no equivalent, so `gemma-tight` cannot guard a
                              prompt before the call. Overflow fails at the
                              model.
    drop_params               no equivalent — every parameter is forwarded as
                              sent. No longer needed; see above.
    per-route max_tokens      no equivalent, and this one bites. A gateway
                              endpoint holds a model and nothing else — there is
                              nowhere to put a default inference parameter. It
                              matters because Gemma 4 REASONS before it answers
                              and the reasoning is charged against max_tokens: on
                              an unlucky run the whole budget goes on thinking
                              and the call returns HTTP 200 with `content: ""`
                              and `finish_reason: "length"`. Measured on one
                              lesson prompt: 1170 completion tokens on one run,
                              the full 4096 on the next. Every caller sets its
                              own max_tokens, and any caller that cannot tolerate
                              an empty answer retries.
    per-route timeout         one global figure instead,
                              MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS in compose.yml.

    fallback chains           MLflow supports them; this config has none, because
                              with one provider there is nowhere to fall back to.
                              The wiring is gone from this script too. If it ever
                              comes back, the trap is worth knowing in advance: a
                              FALLBACK linkage is stored, and shown in the UI,
                              whether or not you pass a `FallbackConfig` — but
                              the gateway only wraps the primary in a fallback
                              provider when that object is there. Leave it out
                              and the chain looks right everywhere except in
                              production.

Usage:
    python seed_gateway.py [--reset] [--prune]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.tracking._tracking_service.utils import _get_store

DEFAULT_TRACKING_URI = "http://mlflow:5000"


@dataclass(frozen=True)
class Secret:
    """One provider account: a credential, and the base URL it applies to.

    Declarative — it names environment variables rather than holding values, so
    a key never sits in this file and never reaches a log line.
    """

    provider: str
    api_key_env: str
    api_base_env: str = ""
    api_base_default: str = ""

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def api_base(self) -> str:
        if not self.api_base_env:
            return ""
        return os.environ.get(self.api_base_env, "") or self.api_base_default


@dataclass(frozen=True)
class Model:
    """One distinct upstream model, addressed through a secret."""

    name: str
    secret: str
    model: str


@dataclass(frozen=True)
class Endpoint:
    """One alias a lesson names."""

    alias: str
    model: str


# =============================================================================
# Configuration
# =============================================================================

# ---- Gateway-side tracing ---------------------------------------------------
#
# On, every request through every endpoint becomes an MLflow trace in an
# auto-created `gateway/<alias>` experiment — no callback, no autolog call, no
# client-side code.
#
# OFF here, deliberately. This repo is a tutorial about tracing: 8 aliases means
# 8 extra experiments, and every lesson call would appear twice — once in the
# lesson's own experiment and once under `gateway/`. A learner comparing trace
# counts would be counting the same call twice. Flip it to True to see the
# feature, then `podman compose run --rm mlflow-seed --reset`.
USAGE_TRACKING = False

# ---- Secrets: one per provider account --------------------------------------
#
# `provider="openai"` means "speaks the OpenAI protocol", not "is OpenAI".
# api_base is the only thing separating Unsloth from api.openai.com.
#
# The base URL belongs in auth_config, which is what api_base_env feeds. Put it
# anywhere else and MLflow ignores it without a word — the server then calls
# api.openai.com and reports an authentication error about a key it never sent.
SECRETS: dict[str, Secret] = {
    "unsloth": Secret(
        provider="openai",
        api_key_env="UNSLOTH_API_KEY",
        # As seen FROM INSIDE the MLflow container: the server dials Unsloth, you
        # do not, so "localhost" here would be the container itself.
        api_base_env="UNSLOTH_API_BASE",
        api_base_default="http://host.containers.internal:8888/v1",
    ),
}

# ---- Model definitions: one per DISTINCT model ------------------------------
#
# Three models cover every alias. Download them in Unsloth; do not load one by
# hand, because auto-switch does that on demand.
#
# The `-qat` suffix is the quantisation-aware-trained build and it is deliberate.
MODELS: list[Model] = [
    Model("unsloth-gemma-26b-qat", "unsloth", "unsloth/gemma-4-26B-A4B-it-qat-GGUF"),
    Model("unsloth-gemma-31b-qat", "unsloth", "unsloth/gemma-4-31B-it-qat-GGUF"),
    Model("unsloth-nomic-embed", "unsloth", "second-state/Nomic-embed-text-v1.5-Embedding-GGUF"),
]

# ---- Endpoints: the names every lesson calls --------------------------------
#
# Named for the JOB, not the model. `gemma-chat`, `gemma-judge` and `gemma-agent`
# all resolve to one model today; the split is what lets them stop being one
# later, without a sweep through forty lessons.
ENDPOINTS: list[Endpoint] = [
    Endpoint("gemma-chat", "unsloth-gemma-26b-qat"),  # the application under observation
    Endpoint("gemma-judge", "unsloth-gemma-26b-qat"),  # LLM-as-judge, scorers, simulators
    Endpoint("gemma-agent", "unsloth-gemma-26b-qat"),  # agent loops and tool calling
    Endpoint("gemma-tight", "unsloth-gemma-26b-qat"),  # context-overflow demos
    # MLflow's judge aligner asks for these two BY NAME and gives you no way to
    # change it, so the aliases exist to satisfy those calls with local weights.
    Endpoint("gpt-4.1-mini", "unsloth-gemma-26b-qat"),
    Endpoint("text-embedding-3-small", "unsloth-nomic-embed"),
    # The denser local model. The sweep lessons compare it against the 26B above,
    # and auto-switch pays 4-14 s each time the sweep crosses between them.
    Endpoint("gemma-31b-local", "unsloth-gemma-31b-qat"),
    # Embeddings for RAG and the vector DB.
    Endpoint("nomic-embed", "unsloth-nomic-embed"),
]


# =============================================================================
# Applying it
# =============================================================================


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def apply_secrets(store, secrets: dict[str, Secret]) -> dict[str, str]:
    """Create or refresh one secret per provider account. Returns name -> id."""
    existing = {s.secret_name: s.secret_id for s in store.list_secret_infos()}
    built: dict[str, str] = {}

    for name, secret in secrets.items():
        if not secret.api_key:
            # Creating a secret with no key would build endpoints that 401 at
            # call time, which is a worse failure than not having them.
            print(f"  {name:12s} SKIPPED  -- ${secret.api_key_env} is not in the environment")
            continue

        # The base URL belongs in auth_config. Put it in secret_value and MLflow
        # ignores it without a word -- the server then calls api.openai.com and
        # reports an auth failure about a key it never sent.
        auth_config = {"api_base": secret.api_base} if secret.api_base else {}

        if name in existing:
            store.update_gateway_secret(
                secret_id=existing[name],
                secret_value={"api_key": secret.api_key},
                auth_config=auth_config,
            )
            built[name] = existing[name]
            print(f"  {name:12s} refreshed  provider={secret.provider}")
            continue

        created = store.create_gateway_secret(
            secret_name=name,
            secret_value={"api_key": secret.api_key},
            provider=secret.provider,
            auth_config=auth_config,
        )
        built[name] = created.secret_id
        print(f"  {name:12s} created    provider={secret.provider}  ->  {secret.api_base or 'provider default'}")
    return built


def apply_models(store, models: list[Model], secrets: dict[str, Secret], secret_ids: dict[str, str]) -> dict[str, str]:
    """Create one model definition per distinct model. Returns name -> id."""
    existing = {d.name: d.model_definition_id for d in store.list_gateway_model_definitions()}
    built: dict[str, str] = {}

    for model in models:
        if model.secret not in secret_ids:
            print(f"  {model.name:28s} SKIPPED  -- needs the '{model.secret}' secret")
            continue
        if model.name in existing:
            built[model.name] = existing[model.name]
            print(f"  {model.name:28s} reused")
            continue
        created = store.create_gateway_model_definition(
            name=model.name,
            secret_id=secret_ids[model.secret],
            provider=secrets[model.secret].provider,
            model_name=model.model,
        )
        built[model.name] = created.model_definition_id
        print(f"  {model.name:28s} created  ->  {model.model}")
    return built


def apply_endpoints(
    store,
    endpoints: list[Endpoint],
    model_ids: dict[str, str],
    usage_tracking: bool,
    reset: bool,
) -> list[tuple[str, str]]:
    """Create one endpoint per alias. Returns (alias, status) per alias."""
    existing = {e.name: e.endpoint_id for e in store.list_gateway_endpoints()}
    rows: list[tuple[str, str]] = []

    for endpoint in endpoints:
        if endpoint.model not in model_ids:
            print(f"  {endpoint.alias:24s} SKIPPED  -- needs model definition '{endpoint.model}'")
            continue
        if endpoint.alias in existing and not reset:
            print(f"  {endpoint.alias:24s} reused   (--reset to rebuild)")
            rows.append((endpoint.alias, "reused"))
            continue
        if endpoint.alias in existing:
            store.delete_gateway_endpoint(existing[endpoint.alias])

        store.create_gateway_endpoint(
            name=endpoint.alias,
            model_configs=[
                GatewayEndpointModelConfig(
                    model_definition_id=model_ids[endpoint.model],
                    # The ENUM, not the string "PRIMARY".
                    linkage_type=GatewayModelLinkageType.PRIMARY,
                    weight=1,
                    fallback_order=0,
                )
            ],
            usage_tracking=usage_tracking,
        )
        print(f"  {endpoint.alias:24s} created  ->  {endpoint.model}")
        rows.append((endpoint.alias, "created"))
    return rows


def report_extras(store, prune: bool) -> None:
    """Everything MLflow holds that this file no longer names.

    All THREE object types, in dependency order: an endpoint references a model
    definition, which references a secret, so deleting the other way round is
    refused. Sweeping only endpoints -- which is all this did originally -- leaves
    orphan model definitions and, worse, orphan SECRETS: a removed provider's API
    key stays live in the tracking database, referenced by nothing and visible to
    anyone with UI access. Removing a provider has to mean removing its key.
    """
    found_any = False

    for label, items, delete in (
        (
            "endpoint",
            [
                (e.name, e.endpoint_id)
                for e in store.list_gateway_endpoints()
                if e.name not in {x.alias for x in ENDPOINTS}
            ],
            store.delete_gateway_endpoint,
        ),
        (
            "model",
            [
                (d.name, d.model_definition_id)
                for d in store.list_gateway_model_definitions()
                if d.name not in {m.name for m in MODELS}
            ],
            store.delete_gateway_model_definition,
        ),
        (
            "secret",
            [(s.secret_name, s.secret_id) for s in store.list_secret_infos() if s.secret_name not in SECRETS],
            store.delete_gateway_secret,
        ),
    ):
        for name, object_id in items:
            found_any = True
            if prune:
                delete(object_id)
                print(f"  {label:8s} {name:28s} DELETED  (--prune)")
            else:
                print(f"  {label:8s} {name:28s} not in this file -- left alone (--prune removes it)")

    if not found_any:
        print("  none -- MLflow holds exactly what this file names")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Seed the MLflow AI Gateway from {Path(__file__).name}.")
    parser.add_argument("--reset", action="store_true", help="delete and rebuild every endpoint")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete endpoints, model definitions and secrets this file no longer names",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    store = _get_store()

    print(f"seeding {tracking_uri}  ->  {len(SECRETS)} secret, {len(MODELS)} models, {len(ENDPOINTS)} endpoints")

    section("Step 1: secrets")
    secret_ids = apply_secrets(store, SECRETS)
    if not secret_ids:
        print("\nnothing to seed: UNSLOTH_API_KEY did not reach the container.", file=sys.stderr)
        print("It is the only credential this stack needs. Copy it from Unsloth", file=sys.stderr)
        print("Settings -> API, export it, and run `podman compose up -d` again.", file=sys.stderr)
        return 1

    section("Step 2: model definitions")
    model_ids = apply_models(store, MODELS, SECRETS, secret_ids)

    section("Step 3: endpoints")
    rows = apply_endpoints(store, ENDPOINTS, model_ids, USAGE_TRACKING, args.reset)

    section("Step 4: objects this file does not name")
    report_extras(store, args.prune)

    created = sum(1 for _, status in rows if status == "created")
    print(
        f"\ndone: {len(secret_ids)} secret, {len(model_ids)} model definitions, {len(rows)} endpoints ({created} new)"
    )
    print(f'call one:  POST {tracking_uri}/gateway/mlflow/v1/chat/completions  with  "model": "{rows[0][0]}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
