# Daita Terminal TUI Design and Implementation Plan

Date: 2026-07-23  
Status: Proposed implementation plan  
Primary interactive command: `daita`

## 1. Objective

Turn Daita's current line-oriented terminal chat into a polished,
transcript-first terminal UI that feels deliberate and pleasant to use while
preserving the product's direct architecture:

```text
user message -> model -> zero or more tool calls -> ordered tool results
             -> model -> answer
```

The target experience is inspired by the visual hierarchy of Oh My Pi:

- a quiet, readable conversation transcript;
- bordered cards for model-requested tool work;
- live but restrained run status;
- a persistent multiline composer;
- a compact status line;
- expandable detail rather than an always-noisy trace; and
- strong keyboard navigation.

This is a presentation improvement over the existing public `Agent` API and
terminal controller. It does not add a second agent loop, workflow runtime,
session runtime, event bus, telemetry store, pending approval system, or
alternate source of catalog truth.

## 2. Product decisions

### 2.1 The interface is transcript-first, not dashboard-first

Daita should not look like a monitoring dashboard with permanent sidebars and
multiple panes competing for attention. The conversation remains the primary
surface. Model and tool activity appears inline where it occurred, and the
composer and status line remain anchored at the bottom.

The default ready-state layout is:

```text
 DAITA  atlas                                      PostgreSQL · 3 sources
────────────────────────────────────────────────────────────────────────

 You
 Which region leads paid revenue this quarter?

 ╭─ ✓ Search catalog · 42ms ───────────────────────────────────────────╮
 │ Found orders, customers, regions, and payments                     │
 ╰────────────────────────────────────────────────────────────────────╯

 ╭─ ✓ Query PostgreSQL · 184ms · 6 rows ──────────────────────────────╮
 │ SELECT region, SUM(amount) AS revenue ...                          │
 │                                                                    │
 │ region          revenue                                            │
 │ EMEA            $4.2M                                              │
 │ North America   $3.7M                                              │
 │                                              Ctrl+O expand          │
 ╰────────────────────────────────────────────────────────────────────╯

 Daita
 EMEA leads paid revenue with $4.2M, approximately 14% ahead of
 North America.

╭──────────────────────────────────────────────────────────────────────╮
│ › Ask a follow-up…                                                   │
╰──────────────────────────────────────────────────────────────────────╯
 atlas · gpt-5.6-sol · ● ready          2 steps · 1.8k tokens · $0.02
```

The terminal emulator owns window chrome. Daita must not draw imitation macOS
traffic-light buttons, a fake title bar, or other operating-system chrome.

### 2.2 Green is the Daita identity

Green is the primary brand and interaction color. It identifies Daita,
focus, selectable content, the prompt, active controls, and the structural
edges of the application.

Green must not make every successful item visually loud. Success uses a green
check glyph plus primarily neutral text. Selection uses green plus weight,
position, or a focus border. States must remain understandable without color.

Semantic roles:

| Role | Treatment |
| --- | --- |
| Daita identity | Brand green, bold where appropriate |
| Active selection | Focus green plus `›`, bold, underline, or focus border |
| Composer focus | Brand-green border and prompt glyph |
| Running activity | Muted green spinner plus neutral label |
| Successful work | Green `✓` glyph plus neutral label |
| Data identifiers | Cyan/blue for tables, columns, sources, and paths |
| Approval or warning | Amber plus explicit label or glyph |
| Failure | Red plus `!` or `×` and explicit failure text |
| Secondary metadata | Cool gray |
| Primary prose | Terminal default foreground |
| Background | Terminal default background |

Target true-color references:

```text
brand green     #22C55E
focus green     #4ADE80
muted green     #15803D
data cyan       #38BDF8
warning amber   #F59E0B
error red       #F87171
muted text      #71717A
```

These hex values are design references, not a requirement to paint an assumed
dark background. Prefer semantic ANSI colors at lower color depths so the
user's terminal palette can adapt. Use true color only when the output reports
support. Honor `NO_COLOR`.

Example distinction:

```text
› PostgreSQL                 selected: green + bold
✓ Query PostgreSQL · 184ms   success: green glyph, neutral label
```

### 2.3 Use `prompt_toolkit` for interaction and Rich for content rendering

Keep `prompt_toolkit` as the terminal application owner. It already exists as
a default production dependency and owns the current arrow-key selectors. Use
it for:

- the terminal application and redraw loop;
- input and output abstraction;
- keyboard bindings;
- multiline composer editing;
- focus and scrolling;
- completion menus;
- terminal resize handling;
- alternate-screen admission and restoration; and
- deterministic injected input/output tests.

Add Rich as a default production dependency and use it only for rendering:

- model Markdown;
- syntax-highlighted SQL and code;
- width-aware result tables;
- panels and rules;
- wrapping and truncation; and
- semantic theme tokens.

Do not introduce Textual in this stage. Textual is a capable application
framework, but replacing the working onboarding, selector, cancellation, and
test seams would create a much larger migration than the requested transcript
UI requires. Do not combine two independent application/event loops.

Both terminal dependencies remain lazy. Importing `daita`, `daita.cli`, or
executing headless commands must not import `prompt_toolkit` or Rich. A missing
or damaged production dependency must raise the existing normalized
application-repair guidance:

```text
pipx reinstall daita-agents
```

Do not add a terminal extra. The installed application remains
batteries-included.

### 2.4 The interactive TUI and headless CLI remain distinct surfaces

When stdin, stdout, and stderr are interactive terminals, the default `daita`
command launches the TUI.

Existing advanced/headless subcommands retain their current contracts:

- `daita create`
- `daita attach`
- `daita sources`
- `daita run`
- `daita chat`
- `daita memory`
- `daita skills`

Their argument parsing, stdout/stderr routing, JSON/JSONL behavior, exit codes,
and lazy import behavior must not change as an incidental consequence of the
TUI.

Unsupported terminals, non-interactive streams, or failures before enhanced
terminal admission retain a deterministic plain-text path. Never print Rich
markup or raw terminal control sequences into redirected output.

## 3. Current state

The current terminal controller in `src/daita/terminal.py` already owns:

- agent selection and creation;
- model onboarding and replacement;
- source onboarding and catalog repair;
- readiness projection;
- local slash commands;
- exact once-only approval prompts;
- run interruption; and
- terminal-safe plain-text rendering.

`src/daita/terminal_selection.py` already owns bounded `prompt_toolkit`
single-select and multi-select behavior with numbered fallbacks.

The current chat path is still line-oriented:

```text
You › <input>

Daita
<answer>

<steps> steps · <tokens> tokens · <cost>
```

This is functionally sound but lacks:

- persistent composition;
- live run feedback;
- visual grouping of tools and results;
- Markdown, syntax, and table rendering;
- slash-command completion;
- transcript scrolling;
- responsive visual hierarchy; and
- a coherent Daita theme.

The contract to improve belongs to terminal presentation. It does not belong
to `AgentLoop`, `DataContextBuilder`, `DataToolRuntime`, provider adapters, or
the state store.

## 4. Architecture

### 4.1 Dependency direction

The TUI is a projection over existing public behavior:

```text
Agent public API + AgentObserver + ApprovalHandler
                         │
                         ▼
            ephemeral terminal view state
                         │
                         ▼
        transcript / tool cards / composer / status
```

The TUI may:

- call public `Agent` methods;
- receive bounded `AgentEvent` records;
- implement the existing in-process `ApprovalHandler`;
- load a completed run through `Agent.transcript(run_id)`;
- display current catalog/source summaries through public APIs; and
- cancel the foreground `agent.run(...)` task.

The TUI must not:

- call providers, source clients, capabilities, or executors directly;
- interpret model text as trusted UI instructions;
- create a second transcript or persistence format;
- persist cursor, focus, expansion, spinner, or selection state;
- retry or repair a model run;
- reorder tool results;
- synthesize a second answer;
- create durable events or telemetry;
- approve an action without the existing exact approval callback; or
- turn displayed content into authorization.

### 4.2 Smallest implementation owner

Keep `src/daita/terminal.py` as the terminal workflow/controller. It continues
to decide which public action happens next.

Add one focused presentation module:

```text
src/daita/terminal_tui.py
```

It initially owns:

- lazy terminal-rendering imports;
- the interactive application layout;
- ephemeral view-state records;
- event-to-view-state projection;
- the composer and key bindings;
- theme construction;
- safe Rich rendering;
- live tool-card projection; and
- terminal restoration.

Do not begin with a generic widget hierarchy, screen router, form framework, or
new package tree. If the focused module later becomes materially difficult to
maintain, split only the pure content renderers into:

```text
src/daita/terminal_rendering.py
```

Do not split merely to create symmetry.

`src/daita/terminal_selection.py` remains the bounded selector owner during the
first vertical slice. Onboarding selectors can adopt the shared theme without
moving provider, model, source, schema, or repair semantics out of
`terminal.py`.

### 4.3 Ephemeral view state

Define small terminal-only records, for example:

```text
TerminalViewState
  agent label
  model label
  source summary
  conversation label
  transcript blocks
  tool cards keyed by call_id
  current run status
  current-process usage totals
  focused overlay, if any
  composer buffer
  scroll/expansion state

ToolCardState
  call_id
  capability_id, when available
  safe display label
  state: queued | running | approval | succeeded | failed
  duration
  error code, when present
  bounded hydrated details, when complete
```

This state is disposable. Relaunching Daita reconstructs durable facts through
the public API; it does not restore TUI focus or animations.

Tool cards are keyed by `call_id`. During execution, first-seen event order is
used for live display. After completion, hydrate and normalize cards in the
canonical call order recorded in the exact transcript so concurrent execution
never changes presentation order.

### 4.4 Observer bridge

Construct one terminal event bridge before creating or opening the selected
agent, and inject it as the existing `observer` every time the terminal
controller calls `Agent.create(...)` or `Agent.open(...)`, including reopen
after a model change.

The synchronous observer must return promptly:

```text
AgentEvent -> nonblocking enqueue -> return
```

It must not render, perform source I/O, read transcripts, or await work inside
the observer callback. The interactive application consumes queued events,
updates `TerminalViewState`, and invalidates the `prompt_toolkit` application.
Observer failures remain unable to affect execution.

Map events as follows:

| Event | TUI effect |
| --- | --- |
| `run.started` | Mark run active and start restrained status animation |
| `model.completed` | Update model timing and per-run token indicators |
| `tool.started` | Create or mark one card running |
| `approval.requested` | Mark card waiting and focus the approval surface |
| `approval.decided` | Record approved, denied, or failed outcome |
| `tool.completed` | Mark success/failure and record duration/error code |
| `run.completed` | Stop animation and apply final usage/exit state |

Event data is metadata only. Do not expand the observation contract with raw
prompts, tool arguments, data results, memory, skill content, or secrets for
the convenience of the TUI.

### 4.5 Completed transcript hydration

After `agent.run(...)` returns, call:

```python
transcript = await agent.transcript(result.run_id)
```

Use the exact completed transcript to hydrate bounded display details:

- tool arguments appropriate for presentation;
- SQL statements;
- selected source/resource identifiers;
- successful result summaries;
- tabular previews;
- structured error details; and
- final model text.

The transcript remains the durable source. The hydrated card is a safe,
bounded projection and is never written back.

Do not query a source again merely to populate a card. Do not infer a missing
result. Every card must correspond to the canonical transcript.

### 4.6 Approval bridge

The TUI implements the existing once-only, in-process approval callback.

When the callback receives an `ApprovalRequest`:

1. enqueue an approval presentation request;
2. focus a scrollable approval panel;
3. render the exact frozen arguments within the existing review bound;
4. offer explicit `Approve once` and `Deny` actions;
5. await the user's decision;
6. return the existing `ApprovalDecision`; and
7. close the panel and return focus to the transcript/composer.

Escape, Ctrl-C, rendering failure, application shutdown, or invalid input deny
closed. Never add approve-later state, a resume token, global approval,
remembered approval, or automatic confirmation.

The panel must make the action and scope obvious:

```text
 APPROVAL REQUIRED

 Tool          skill_save
 Capability    skills.write

 Exact arguments
 ╭────────────────────────────────────────────────────────────────────╮
 │ { ... }                                                            │
 ╰────────────────────────────────────────────────────────────────────╯

 [A] Approve once                                      [D] Deny
```

## 5. Layout and visual behavior

### 5.1 Header

The header is one quiet line. It includes:

- `DAITA` in brand green;
- the selected agent name;
- a compact source summary on sufficiently wide terminals; and
- no fake window chrome.

Do not repeat detailed readiness information on every turn. `/status` can open
or render the full current model, source, catalog, sync, and conversation
facts.

### 5.2 Transcript viewport

The transcript is vertically scrollable and owns most of the terminal height.
It contains:

- user messages;
- tool cards in canonical order;
- approval outcomes;
- Daita answers; and
- bounded local command results.

Avoid boxing ordinary user and assistant prose. Whitespace, labels, and color
provide sufficient hierarchy. Reserve borders for:

- tool work;
- approval;
- the composer; and
- focused setup selections.

When the user has not scrolled away, new content keeps the viewport pinned to
the bottom. If the user scrolls upward, new activity must not yank the
viewport. Show a small `new activity ↓` indicator until the user returns to the
bottom.

### 5.3 Composer

The composer is always visible while chat is ready. It supports:

- multiline editing;
- visible focus;
- slash-command completion;
- current-process input history;
- paste;
- bounded input validation;
- disabled submission while an exact approval owns focus; and
- a running-state hint without hiding the user's draft.

Suggested empty-state placeholder:

```text
› Ask about your data…
```

Do not persist shell history, composer drafts, or prompts as a separate
history mechanism. Submitted user messages continue to persist only through
the existing run transcript.

### 5.4 Status line

The bottom status line has two regions.

Left:

```text
atlas · gpt-5.6-sol · ● ready
```

Right:

```text
2 steps · 1.8k tokens · $0.02
```

During a run:

```text
atlas · gpt-5.6-sol · ◐ querying
```

Use the current result and bounded observation events for usage and state.
Do not invent progress percentages. A running model request has an activity
state, not an estimated completion percentage.

At narrow widths, collapse in this order:

1. hide cost;
2. hide tokens;
3. shorten model label;
4. hide source summary;
5. retain agent, state glyph, and state word.

### 5.5 Tool cards

Cards should answer three questions at a glance:

1. What is Daita doing?
2. Did it work?
3. Is more detail available?

Example collapsed states:

```text
╭─ ◐ Query PostgreSQL ────────────────────────────────────────────────╮
│ Running…                                                           │
╰────────────────────────────────────────────────────────────────────╯

╭─ ✓ Query PostgreSQL · 184ms · 6 rows ──────────────────────────────╮
│ SELECT region, SUM(amount) AS revenue ...              Ctrl+O more │
╰────────────────────────────────────────────────────────────────────╯

╭─ ! Query PostgreSQL · 31ms ────────────────────────────────────────╮
│ unknown_column · column "reveneu" is not in the current catalog    │
╰────────────────────────────────────────────────────────────────────╯
```

Successful cards collapse by default. Failed cards expand automatically.
Approval cards remain expanded until decided.

Presentation names derive from stable capability identity when it is
available. Do not infer capability semantics by parsing arbitrary tool-name
strings. For an unavailable or unknown tool, display the bounded sanitized
tool name without inventing behavior.

Potential built-in presentation labels include:

| Capability family | Label |
| --- | --- |
| Catalog search | Search catalog |
| Catalog inspection | Inspect schema |
| Catalog traversal | Follow relationships |
| SQLite query | Query SQLite |
| PostgreSQL query | Query PostgreSQL |
| Local file read | Read data file |
| Memory update | Update memory |
| Skill view | Read skill |
| Skill save | Save skill |
| Skill delete | Delete skill |

This mapping is presentation only. It must not select or execute a capability.

### 5.6 Model answers

Render assistant text as sanitized Markdown:

- headings are visually restrained;
- lists remain readable;
- inline code is distinct;
- fenced code is syntax highlighted;
- tables fit the current terminal width;
- raw HTML is not interpreted as terminal control;
- model-provided ANSI/OSC/control characters are neutralized first; and
- generated hyperlinks are disabled initially.

Do not enable Rich markup parsing for untrusted plain strings. A table name
containing brackets must remain data rather than a style instruction.

### 5.7 Data result previews

Use compact tables when the tool result has a clear row/column shape. The
initial preview should be intentionally bounded:

```text
collapsed SQL/details       1 logical line
expanded SQL/code           at most 80 visible lines
expanded text/JSON          at most 16 KiB
initial table preview       at most 10 rows and 12 columns
expanded table preview      at most 50 rows and 20 columns
cell preview                at most 240 display characters
```

If content exceeds a presentation bound, show an explicit indicator:

```text
… 34 more rows in the recorded tool result
```

The TUI truncation does not alter the exact transcript.

### 5.8 Responsive modes

Support three presentation widths:

| Width | Mode |
| --- | --- |
| 100 columns or more | Full metadata, normal cards, two-sided status |
| 70-99 columns | Compact metadata and fewer preview columns |
| Below 70 columns | Line-oriented cards, stacked metadata, minimal status |

Never assume a fixed terminal height. The transcript must retain at least one
visible row after accounting for the header, composer, and status line. Very
small terminals show a clear resize hint rather than raising or corrupting
the screen.

### 5.9 Color, Unicode, and accessibility

Requirements:

- honor `NO_COLOR`;
- use terminal default foreground and background;
- downgrade true color to 256/16-color semantic equivalents;
- use text and glyphs in addition to color for every state;
- use Unicode borders and glyphs only when supported;
- provide ASCII equivalents such as `+`, `-`, `>`, `OK`, and `!`;
- do not rely on blinking text;
- do not use low-contrast green prose for long passages; and
- restore cursor visibility, terminal mode, and alternate screen on every
  normal, failed, interrupted, or cancelled exit.

## 6. Keyboard contract

Default chat bindings:

| Key | Behavior |
| --- | --- |
| Enter | Submit a nonempty composer |
| Ctrl-J | Insert a newline |
| Tab | Open or advance completion |
| Escape | Close completion or the focused overlay |
| Up/Down | Navigate completion/history when appropriate; otherwise move cursor |
| Page Up/Page Down | Scroll transcript |
| Home/End | Move within composer; with transcript focus, jump as appropriate |
| Ctrl-O | Toggle expanded detail for completed tool cards |
| Ctrl-L | Redraw the application |
| Ctrl-C during run | Cancel the foreground run and return to composer |
| Ctrl-C while idle | Clear/interruption feedback without silently exiting |
| Ctrl-D on empty composer | Exit cleanly |
| `a` in approval | Approve the exact request once |
| `d` or Escape in approval | Deny |

Do not depend on Shift-Enter because terminal support is inconsistent.

Slash-command completion includes the current local command surface:

```text
/model
/sources
/source add
/source refresh <id>
/catalog
/settings
/new
/resume <id>
/memory
/user
/skills
/status
/conversation
/help
/exit
```

Completions are terminal-local. They are not included in model context or
written to the transcript unless the user submits a normal model message.

## 7. Setup and onboarding

The first implementation slice may enter the full-screen transcript
application only after the existing sequential setup reaches `Ready`.

Then apply the same visual system to onboarding:

- agent selection;
- provider selection;
- provider-specific model selection;
- source-type selection;
- PostgreSQL schema multi-selection;
- empty-catalog repair;
- bounded text fields;
- hidden credentials; and
- validation progress/results.

`terminal.py` continues to own the sequence. Do not introduce persisted wizard
state or a screen router. Each selector/form returns one stable domain value
to the current controller.

Hidden credentials remain owned by a hidden input boundary. Never render,
enqueue, log, persist, or include them in view state.

## 8. Security and trust requirements

All terminal-visible external content is untrusted:

- model text;
- catalog names and descriptions;
- source display names;
- table and column names;
- file content;
- row values;
- tool arguments and results;
- memory documents; and
- skill documents.

Before rendering:

1. normalize line endings;
2. remove or visibly neutralize C0/C1 controls except admitted newlines/tabs;
3. neutralize ANSI CSI, OSC, DCS, bidi overrides, and terminal title changes;
4. apply the existing display bounds;
5. escape Rich markup for plain fields;
6. parse Markdown only in the designated model-answer renderer; and
7. let the terminal library produce all actual control sequences.

Do not let untrusted content:

- change style outside its renderable;
- inject key bindings;
- create a command;
- trigger a tool;
- grant approval;
- change focus;
- alter the terminal title;
- write to the clipboard;
- open a URL automatically; or
- escape the application layout.

## 9. Implementation stages

### Stage 1: focused shell vertical slice

Implement one ready-agent chat turn through the new TUI:

- header;
- transcript viewport;
- user message;
- multiline composer;
- Daita answer;
- status line;
- brand-green theme;
- safe Markdown rendering;
- Ctrl-C cancellation;
- Ctrl-D exit; and
- clean terminal restoration.

Keep current onboarding and local commands functional. Do not add tool-detail
hydration before the shell is deterministic.

Focused tests:

- ready agent enters TUI;
- one text-only turn renders user and Daita content;
- controls in model output are neutralized;
- composer bounds are preserved;
- cancellation returns to composer;
- EOF exits and restores output; and
- headless imports do not load terminal dependencies.

### Stage 2: live execution cards

Add:

- observer bridge;
- queued event consumption;
- run/model/tool status;
- per-call cards keyed by `call_id`;
- concurrent live tool ordering;
- durations and error states;
- run totals; and
- quiet animation.

Focused tests:

- observer callback performs nonblocking enqueue only;
- all seven current event kinds project correctly;
- one failed tool does not hide a sibling tool;
- cards settle in canonical call order;
- observer/rendering failure does not alter `LoopExit`; and
- run cancellation settles all live UI state.

### Stage 3: transcript hydration and rich data rendering

Add:

- completed transcript load;
- SQL/code rendering;
- JSON/text details;
- row/column previews;
- explicit truncation;
- collapse/expand behavior; and
- failed-card auto-expansion.

Focused tests:

- no source is queried during hydration;
- card details come from the exact run transcript;
- SQL and data controls cannot escape the renderer;
- result previews obey every bound;
- expanded/collapsed state is not persisted; and
- transcript call/result ordering is retained.

### Stage 4: approval and local command surfaces

Add:

- focused exact-approval panel;
- fail-closed key handling;
- themed `/status`, `/sources`, `/catalog`, and `/settings`;
- slash completion;
- process-local input history; and
- `/new` and `/resume` conversation projection.

Focused tests:

- the complete exact frozen arguments are reviewable within the bound;
- approval returns exactly one existing `ApprovalDecision`;
- Escape, cancellation, and rendering failure deny;
- secrets never appear in output or view state;
- local commands never reach the model transcript; and
- resumed conversations remain agent-scoped.

### Stage 5: onboarding, responsive modes, and fallback

Add:

- shared visual treatment for existing selectors;
- themed setup fields and validation status;
- narrow/normal/wide modes;
- `NO_COLOR`;
- 16/256/true-color projection;
- ASCII glyph fallback;
- very-small-terminal handling; and
- deterministic non-TTY fallback.

Focused tests:

- widths below 70, from 70-99, and at least 100 columns;
- resize while idle, running, and approving;
- cursor and screen restoration after every exit path;
- numbered fallback retains stable identities;
- model/source onboarding writes no duplicate state; and
- a ready returning agent skips onboarding exactly as today.

## 10. Expected file changes

The first coherent implementation is expected to touch:

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Add bounded Rich production dependency |
| `src/daita/terminal_tui.py` | New focused interactive presentation owner |
| `src/daita/terminal.py` | Delegate ready chat presentation and inject observer/approval bridges |
| `src/daita/terminal_selection.py` | Adopt theme or shared presentation behavior only where necessary |
| `src/daita/_installation.py` | Reuse existing pipx repair guidance if needed |
| `tests/test_terminal_tui.py` | Focused TUI state, rendering, input, event, and restoration tests |
| `tests/test_terminal_chat.py` | Preserve controller/chat contracts |
| `tests/test_terminal_acceptance.py` | Public first-run and returning-run vertical slices |
| `tests/test_architecture.py` | Permit lazy imports only at focused terminal presentation boundaries |
| `tests/test_packaging.py` | Assert Rich is a default production dependency and remains lazy |
| `README.md` | Show the finished interactive experience and key bindings |

Do not add a second root package, alternate agent implementation, renderer
plugin registry, theme marketplace, UI persistence table, or compatibility
decoder.

## 11. Verification

During development, run the narrowest relevant tests first:

```bash
pytest tests/test_terminal_tui.py -v
pytest tests/test_terminal_chat.py -v
pytest tests/test_terminal_selection.py -v
pytest tests/test_terminal_acceptance.py -v
pytest tests/test_architecture.py -v
pytest tests/test_packaging.py -v
```

Before handoff:

```bash
pytest tests/ -m "not requires_llm and not requires_db"
python -m black --check src tests
python -m mypy src/daita tests
```

Also perform manual TTY checks in:

- a true-color terminal;
- a 256-color terminal;
- `NO_COLOR=1`;
- a terminal narrower than 70 columns;
- a terminal resized during a run;
- a text-only model answer;
- a multi-tool data answer;
- a structured tool failure;
- an exact approval request;
- Ctrl-C during model work;
- Ctrl-D while idle; and
- redirected/non-TTY output.

Live model or PostgreSQL tests require the repository's existing explicit
authorization and credentials. Do not use repeated paid runs to diagnose
deterministic rendering behavior.

## 12. Definition of done

The TUI is complete when:

1. `daita` on an interactive terminal launches a polished green-identity
   transcript UI.
2. The composer, transcript, header, tool cards, and status line remain usable
   across supported terminal sizes.
3. Text-only answers, tool-heavy answers, failures, cancellations, and
   approvals each have a clear visual state.
4. Live cards are driven by existing bounded observation events.
5. Detailed cards are hydrated only from the exact completed transcript.
6. Tool calls and results remain in canonical order.
7. The direct model/tool loop and runtime boundaries are unchanged.
8. Exact approvals remain once-only, in-process, and fail closed.
9. Untrusted content cannot inject terminal controls, Rich markup, commands,
   focus changes, approval, clipboard writes, or automatic links.
10. `NO_COLOR`, limited-color, Unicode-limited, narrow, and non-TTY paths are
    deterministic and readable.
11. The terminal is restored after normal exit, interruption, failure, and
    cancellation.
12. Headless commands preserve their parsing, output, exit, import, and
    packaging contracts.
13. `prompt_toolkit` and Rich remain lazy terminal-only imports.
14. No TUI state, event stream, progress record, or duplicate transcript is
    persisted.
15. Focused tests, the deterministic suite, architecture tests, packaging
    tests, Black, and mypy pass.

## 13. External design references

- Oh My Pi repository and interactive TUI:
  <https://github.com/can1357/oh-my-pi>
- `prompt_toolkit` full-screen/custom application documentation:
  <https://python-prompt-toolkit.readthedocs.io/en/stable/pages/full_screen_apps.html>
- Rich Markdown:
  <https://rich.readthedocs.io/en/stable/markdown.html>
- Rich tables:
  <https://rich.readthedocs.io/en/stable/tables.html>

The references inform interaction and presentation only. Do not copy Oh My
Pi's coding-agent runtime, tools, session model, extension system, or
TypeScript TUI implementation into Daita.
