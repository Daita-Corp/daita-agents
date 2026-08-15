# Subscription model sources

Daita can use an existing ChatGPT, Claude Code, or Grok Build subscription
instead of a separately billed model API key. Subscription calls consume the
allowance of the connected account, and the models available to that account
still depend on the provider and plan.

Subscription routes are distinct from API-key routes:

| Daita option | Uses | Sign-in |
| --- | --- | --- |
| **Codex subscription** | A ChatGPT account with access to the selected Codex model | Completed inside Daita |
| **Claude Code subscription** | The installed Claude Code client and its saved login | Run `claude auth login` first |
| **Grok Build subscription** | The installed Grok Build client and its saved login | Run `grok login` first |

Daita never silently replaces one of these routes with an API key, custom
endpoint, or different provider. Select an API-backed provider during model
setup when pay-as-you-go API billing is preferred.

## Configure Codex through ChatGPT

The Codex CLI is not required.

1. Run `daita` and create or open an agent.
2. During model setup, choose **Codex subscription**.
3. Choose one of the models offered by Daita.
4. Open the verification URL shown by Daita and enter the one-time code.
5. Return to Daita and wait for model validation to complete.

Daita stores the resulting OAuth credential in the operating-system keychain.
The agent configuration stores only a reference to that keychain entry. Daita
does not read or modify `~/.codex`, and it refreshes its own saved connection
when necessary.

## Configure Claude Code

Install Claude Code using the
[official setup instructions](https://code.claude.com/docs/en/getting-started),
then sign in before starting Daita:

```bash
claude auth login
daita
```

Choose **Claude Code subscription** during model setup, then choose one of the
models offered by Daita. The Claude Code executable must remain installed and
its login must remain valid.

Daita invokes the signed-in client in a temporary directory with its native
tools, browser integration, MCP servers, slash commands, and session persistence
disabled. Daita does not copy Claude Code's saved credential into the agent
home or keychain.

## Configure Grok Build

Install Grok Build using the
[official Grok Build instructions](https://docs.x.ai/build/overview), then sign
in and confirm that the client runs:

```bash
grok login
grok version
daita
```

Choose **Grok Build subscription** during model setup. Daita accepts only the
built-in Grok models it has explicitly reviewed, which may differ from the
provider's newest model. Choose from the models Daita displays instead of
entering a custom provider or model.

Before each use, Daita checks the installed client's headless controls and the
configuration reported by `grok inspect`. Active custom providers, API-key
authentication, instructions, permissions, hooks, plugins, marketplaces, MCP
servers, or other extensions cause validation to fail because Daita cannot
guarantee its model-only boundary in their presence.

Daita does not copy Grok's saved credential. The Grok client continues to own
its login and normal credential refresh. Grok Build currently has no control
that guarantees headless sessions are not persisted, so the client may retain
the bounded request and response in its own session or log files. Daita does
not import those records into the agent home.

## What the model receives

Subscription models receive the same Daita model request as API-backed models.
This can include the current conversation, bounded catalog descriptions, and
data returned by Daita tools when needed to answer the question. Do not treat a
subscription route as local-only processing.

The provider client does not receive a database password, source client, or
direct connection to an attached data source. It can propose Daita tool calls,
but Daita validates and executes those calls through its normal catalog and
permission boundaries. Native client tools are not an alternate path to the
filesystem, shell, browser, or database.

## Credentials, allowance, and cost limits

- **Codex subscription:** Daita owns a ChatGPT OAuth credential in the OS
  keychain and stores only its reference in agent configuration.
- **Claude Code and Grok Build:** the official client owns its saved login;
  Daita invokes that client without copying the credential.
- **Allowance:** setup validation and normal requests consume a small amount of
  the connected subscription allowance.
- **Dollar estimates:** subscription usage is not treated as zero-cost API
  usage. Daita records token usage when the provider reports it, but marks the
  dollar estimate unavailable. A run that requires a complete dollar estimate
  can therefore stop rather than assume the request is free.

## Troubleshooting

### The client is missing or incompatible

Update or reinstall the official client, sign in again, and retry model setup:

```bash
claude auth login
# or
grok login
```

For Grok Build, `grok version` must also succeed. Daita fails closed when the
installed client does not expose the headless controls it requires.

### Authentication failed

- For Codex, start model setup again and repeat the device-code sign-in.
- For Claude Code, run `claude auth login` outside Daita.
- For Grok Build, run `grok login` outside Daita.

If an official client works in a normal terminal but not in Daita, launch Daita
from that same terminal so the client can access its saved login state.

### The selected model is unavailable

The connected plan may not include the model, or the provider may have changed
its availability. Reopen model setup and choose another model offered by Daita.
Do not substitute a custom provider identity under a subscription route.

### Subscription allowance is exhausted

Wait for the provider's allowance to reset or explicitly configure a different
subscription or API-backed provider.

### Grok configuration could not be isolated

Run `grok inspect` to review the configuration Grok discovers. Remove or disable
custom providers, API keys, instructions, permissions, hooks, plugins,
marketplaces, MCP servers, skills, and other active configuration layers, then
run `grok login` and retry. Daita rejects these layers instead of attempting to
reinterpret their precedence.
