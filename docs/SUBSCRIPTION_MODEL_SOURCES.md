# Subscription model sources

Daita has three explicit subscription provider identities:

```text
codex:<model>
claude-code:<model>
grok-build:<model>
```

They remain distinct from API-billed `openai:<model>`, `anthropic:<model>`,
`gemini:<model>`, and `grok:<model>` routes. Selecting a CLI subscription never
silently falls back to an API key, Vertex AI, a custom provider, or a custom
base URL.

## Codex through a ChatGPT subscription

Choose **Codex subscription** during model setup. Daita requests a device code,
shows the OpenAI verification URL and one-time code, waits for approval, stores
the OAuth bundle in the OS keychain, and validates the model through the
ChatGPT Codex Responses transport.

The Codex CLI is not required, and Daita does not read or mutate `~/.codex`.
Only a keychain reference is written to the agent configuration.

## Claude through a Claude Code subscription

Install and sign in to Claude Code first:

```bash
claude auth login
daita
```

Choose **Claude Code subscription**. Daita invokes the official client in a
temporary directory with native tools, MCP, browser access, slash commands,
session persistence, and API-key endpoint overrides disabled. Daita does not
copy Claude Code's saved login.

## Grok Build subscription

Install the official client and sign in:

```bash
curl -fsSL https://x.ai/cli/install.sh | bash
grok login
grok version
daita
```

Choose **Grok Build subscription** and `grok-4.5`. Grok Build does not accept
prompt content from stdin, so Daita writes the bounded request to an owner-only
temporary prompt file and passes only that file path as an argument. Before the
first request, Daita feature-detects the documented headless controls and runs
`grok inspect --json` under the exact inference environment. Inference starts
only when inspection confirms API-key authentication is disabled, no config
layer contributes settings, and no instructions, permissions, hooks, plugins,
marketplaces, MCP servers, LSP servers, or non-bundled skills/agents were
discovered. The subscription route accepts only Daita-reviewed built-in model
identities; it currently accepts `grok-4.5`.

Each run uses owner-only temporary working and process-home directories. Grok's
real `GROK_HOME` remains in place so the official client coordinates through
its normal lock and can atomically rotate `auth.json`; Daita never opens or
copies that credential. The child environment forces
`GROK_DISABLE_API_KEY_AUTH=1` and omits API keys and custom provider variables.
Because an active Grok config cannot be proven harmless without reproducing the
client's config semantics, even a settings-only active config is rejected.

The invocation also:

- replaces the native system prompt with Daita's model-only contract;
- supplies an empty native-tool allowlist plus explicit native-tool and
  permission denials;
- disables planning, subagents, memory, web search/fetch, automatic updates,
  and alternate-screen UI;
- limits the native harness to one model turn; and
- passes Daita's response schema through Grok's native `--json-schema`
  contract; and
- reads the documented `streaming-json` event contract, requiring no advertised
  native tools, one successful `end_turn`, native validated structured output,
  and model-usage confirmation for the requested built-in model.

If the client emits an unexpected event, exceeds one model turn, or reports
usage for a different model, Daita fails the request closed. The child
environment omits `XAI_API_KEY`, custom chat-proxy/OIDC/auth-provider variables,
database secrets, and unrelated provider keys. Daita does not read, copy,
persist, refresh, or deliberately mutate the saved credential; Grok owns its
normal official OAuth refresh while it owns the subprocess, and `grok login`
remains the repair path.

The current Grok Build headless client does not expose a no-session-persistence
flag. It may therefore retain its normal session or log records in the user's
Grok home, including the bounded request and response. Daita still disables
Grok memory and executes from a disposable directory, and Daita does not import
those client records into agent state. This is a client limitation rather than
an implied ephemeral-session guarantee.

Official references:

- <https://docs.x.ai/build/overview>
- <https://docs.x.ai/build/cli/headless-scripting>
- <https://docs.x.ai/build/cli/reference>

## Runtime and security boundary

All three sources use Daita's one direct loop:

```text
Daita transcript -> subscription model transport -> canonical model response
       ^                                                  |
       |                                                  v
Daita tool result <- Daita DataToolRuntime <- proposed Daita tool call
```

The CLI adapters translate canonical messages and projected tools into one
bounded response envelope. A model may propose a Daita tool call, but only
`CapabilityRegistry` and `DataToolRuntime` validate and execute it. The clients
receive no source client, executor, database credential, or alternate tool
path.

Subprocesses are launched directly without a shell. Daita bounds request,
arguments, stdout, stderr, wall time, JSON depth/node count, response text, tool
arguments, and tool-call count. It rejects duplicate-key/non-finite JSON and
terminal controls, and terminates the subprocess group on cancellation,
timeout, or output overflow. Raw client diagnostics are classified at the
adapter boundary and are not retained in normalized exceptions.

Returned token usage is recorded when present. Dollar cost is always marked
`subscription_billing`/unavailable, even if a client emits an API-style dollar
field, so estimated-cost ceilings fail closed instead of treating subscription
usage as free.

## Manual end-to-end smoke tests

These checks consume a small amount of allowance. They are intentionally not
part of the deterministic test suite.

### Grok Build

1. Run `grok login`, then `grok version`.
2. Run `daita`, create a disposable agent, and choose **Grok Build
   subscription**.
3. Select `grok-4.5`; if prompted for unreviewed limits, enter the limits
   appropriate to the installed client/model configuration.
4. Acknowledge the allowance warning and validate.
5. Attach a disposable read-only SQLite/CSV source and ask a question that
   requires catalog inspection and one data read.
6. Exit, run `daita` again, reopen the agent, and ask a follow-up without
   repeating setup.

## Troubleshooting

- **Missing or incompatible Grok Build:** reinstall/update the official client
  and run `grok login`.
- **OAuth/configuration error from Grok:** remove per-model API keys, custom
  providers, MCP servers, skills, and any other active Grok config layer,
  including user, project, managed, requirements, or system configuration;
  otherwise run `grok login` and retry. Daita rejects these layers before
  inference rather than trying to reinterpret their precedence.
- **Allowance/rate limit:** wait for the connected plan's allowance reset or
  choose another explicitly configured provider.
- **Local access error:** launch Daita from a normal terminal where the official
  client can access its own login state.
- **Validation:** each validation request consumes a small amount of the
  connected allowance.
