#!/usr/bin/env bash
# Claude CLI transport wrapper: the SDK on the host spawns this instead of a
# local `claude` binary, so the agent process -- and every tool it runs --
# executes inside the SWE-Bench evaluation container, next to the repo.
#
# SWE_CONTAINER_ID is set per instance via ClaudeAgentOptions.env;
# CLAUDE_CODE_OAUTH_TOKEN must be in the host environment (see README).
set -euo pipefail

: "${SWE_CONTAINER_ID:?SWE_CONTAINER_ID not set -- agent.build_options() passes it via ClaudeAgentOptions.env}"
: "${CLAUDE_CODE_OAUTH_TOKEN:?CLAUDE_CODE_OAUTH_TOKEN not set -- run 'claude setup-token' once and store it (see README)}"

# IS_SANDBOX=1: the CLI refuses --dangerously-skip-permissions as root; this
# container genuinely is a disposable sandbox, which is the flag's intended use.
exec "${SWE_BENCH_RUNTIME:-podman}" exec -i \
  -w /workspace/repo \
  -e "CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}" \
  -e IS_SANDBOX=1 \
  "${SWE_CONTAINER_ID}" \
  claude "$@"
