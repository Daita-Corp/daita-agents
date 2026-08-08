# Subscription model sources

Daita supports two explicit subscription provider identities:

```text
codex:<model>
claude-code:<model>
```

They remain distinct from API-billed `openai:<model>` and
`anthropic:<model>` routes. The two subscription transports also have
different authentication contracts; they are not interchangeable CLI wrappers.

## Codex through a ChatGPT subscription

Choose **Codex subscription** during model setup. Daita will:

1. request a device code from OpenAI;
2. show `https://auth.openai.com/codex/device` and the one-time code;
3. wait for the user to approve Daita in the browser;
4. store the resulting OAuth bundle in the OS keychain; and
5. validate the selected model through the ChatGPT Codex Responses transport.

The Codex CLI is not required, and Daita does not read or mutate
`~/.codex`. Access-token refresh and refresh-token rotation are owned by Daita;
only a keychain reference is written to the agent's configuration.

OpenAI documents Codex's ChatGPT and device-code authentication methods at
<https://learn.chatgpt.com/docs/auth#login-on-headless-devices>. OpenClaw and
Hermes Agent use the same broad shape: application-owned OAuth credentials and
direct ChatGPT Responses transport rather than one `codex exec` process per
model turn.

## Claude through a Claude Code subscription

Claude subscription-plan execution currently uses Claude Code's programmatic
print mode. Install and sign in to Claude Code first:

```bash
claude auth login
daita
```

Then choose **Claude Code subscription**. Daita invokes the official client in
a temporary directory with client-native tools, MCP servers, browser access,
session persistence, and API-key endpoint overrides disabled. Daita does not
copy Claude Code's saved login.

This distinction is intentional. OpenClaw also uses a Claude CLI backend for
subscription-plan execution. A Claude setup/OAuth token sent directly to the
Anthropic Messages API can have different availability or billing semantics,
so Daita does not silently substitute that path for the signed-in Claude Code
plan.

## Runtime boundary

For both providers, Daita remains the only agent loop:

```text
Daita transcript -> subscription model transport -> canonical model response
       ^                                                  |
       |                                                  v
Daita tool result <- Daita DataToolRuntime <- proposed Daita tool call
```

The transport can propose only tools projected in the canonical request.
`AgentLoop` records the response, and `DataToolRuntime` validates and executes
each data read through the same catalog-scoped path used by API providers. The
subscription transport never receives direct source access or a second tool
execution path.

## Operational notes

- Validation consumes a small amount of subscription allowance.
- Subscription availability, model access, and allowance resets belong to the
  connected plan.
- Daita records returned token usage, but subscription allowance has no
  per-request list price. Estimated-dollar cost ceilings therefore fail closed
  for these unpriced routes.
- Subscription transports are currently exposed to Daita as atomic model calls;
  the terminal does not advertise provider streaming for them.
- Codex authentication failures are repaired by reconnecting ChatGPT through
  Daita. Installing or running `codex login` is not part of that repair path.
- Claude Code failures are repaired by updating the official client and running
  `claude auth login` again.
