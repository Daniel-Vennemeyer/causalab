---
name: research-session
description: Bootstrap a session directory under agent_logs/ for an interactive research workflow. Creates plan/, run/, result/ subfolders, a README, and a .current marker that downstream research skills (setup-task, plan-experiment, setup-methods, setup-analyses, run-experiment, interpret-experiment, replicate-paper, document-issues) read at Step 0. Optionally pass --worktree to isolate the session in a fresh git worktree (recommended when running multiple Codex instances against the same checkout). Invoke explicitly at the start of a research workflow — do not invoke for one-off codebase questions or developer-mode work.
args: '[topic] [--worktree]'
---

# Research Session Skill

Bootstraps a session directory under `agent_logs/` that scopes a single research workflow. Downstream research skills read `agent_logs/.current` at Step 0 to find the active session.

## Required Reading

Before running this skill, read the sibling contract document:

- `.Codex/skills/research-session/CONVENTIONS.md` — canonical layout, what goes where, how downstream skills consume the session.

## Autonomous vs Interactive Mode

**Check the `CAUSALAB_AUTONOMOUS` environment variable:**
```bash
echo $CAUSALAB_AUTONOMOUS
```

- **If set to "1"**: Run autonomously. Do not prompt for the topic — infer 1–3 words from the surrounding research context (research objective, task name, analysis being run). Skip the re-invocation prompt and always create a fresh session.
- **If not set or empty**: Run interactively. Prompt for the topic if missing; prompt for a re-invocation choice if a session is already active.

## Arguments

Two optional arguments. Order does not matter.

**Topic** — a 1–3 word slug describing the research focus.

- `/research-session locate-weekdays` → topic = `locate-weekdays`
- `/research-session "locate weekdays sweep"` → topic = `locate-weekdays-sweep` (whitespace slugified to hyphens, capped at 3 words)
- `/research-session` → in interactive mode, prompt the user; in autonomous mode, infer from context.

The topic should describe the research focus, not a placeholder like `auto`, `untitled`, or `test`.

**`--worktree`** — opt-in flag to isolate the session in a fresh git worktree.

- `/research-session locate-weekdays --worktree` → create the session inside a new worktree under `.Codex/worktrees/<session-name>/`.
- Use when running multiple Codex instances against the same checkout. Each agent gets its own worktree, and each worktree gets its own `agent_logs/.current` marker — eliminating the cross-talk that arises when two agents share one `.current` file.
- Tradeoffs: artifacts and `agent_logs/{session}/` live inside the worktree (not the main checkout); the HTML-preview server rooted at the main checkout will not see them; cross-session comparison requires merging the worktree branch back or addressing files via the worktree path.
- Default is off — single-agent workflows keep the simpler ergonomics of editing `agent_logs/` directly in the main checkout.

## Workflow

### Step 0: Check for an existing active session

If `--worktree` was passed, skip this step entirely — the new worktree gets its own `agent_logs/.current`, so any active session in the parent checkout is irrelevant. Proceed to Step 1.

```bash
cat agent_logs/.current 2>/dev/null
```

If the file exists and points at a valid directory under `agent_logs/`:

**Interactive mode:**
> "An active session already exists: `{session_name}`.
>
> Options:
> 1. **New** — start a fresh session, replace `.current`
> 2. **Continue** — keep using the existing session, exit without changes
> 3. **Abort** — exit without changes"

If the user chooses *continue* or *abort*, print the active session path and stop.

**Autonomous mode:** Always start a new session. Skip the prompt.

### Step 1: Resolve the topic

**Interactive mode:**
- If args provided: slugify (lowercase, replace whitespace/underscores with `-`, strip non-alphanumeric except `-`, cap at 3 hyphen-separated words).
- If no args: prompt the user — *"What's the topic for this session? (1–3 words describing the research focus)"* — then slugify.

**Autonomous mode:**
- Infer the topic from the surrounding context. Look at:
  - the research objective in the conversation,
  - the task name being set up or analyzed (`causalab/tasks/<task>/`),
  - the analysis being run (`baseline`, `locate`, `subspace`, …).
- Combine into 1–3 words. Examples: `locate-weekdays`, `subspace-hierarchical`, `replicate-monosemanticity`.
- Never use placeholder strings like `auto`, `untitled`, `test`.

### Step 2: Generate the disambiguator

Pick one adjective and one noun uniformly at random from the wordlists below and join with `-`. Example: `pensive-mongoose`.

If the resulting directory `agent_logs/{date}--{topic}--{adj}-{noun}/` already exists (same-day same-topic collision), re-roll up to 5 times.

**Adjectives (50):**
agile, ample, brisk, calm, candid, clever, cosmic, crisp, curious, daring,
dapper, deft, eager, fervent, fluent, frank, gallant, gentle, glad, hardy,
jolly, keen, lively, lucid, mellow, merry, mighty, nimble, noble, pensive,
placid, plucky, prudent, quick, quiet, regal, robust, serene, sharp, silent,
sleek, snug, spry, stoic, sunny, swift, tender, tidy, vivid, witty

**Nouns (50):**
acorn, antler, aurora, badger, basalt, beacon, bramble, cactus, canyon, cedar,
comet, dolphin, ember, falcon, ferret, finch, fjord, geode, glacier, harbor,
heron, ibex, jasper, kestrel, lantern, marlin, meadow, mongoose, narwhal, nebula,
opal, otter, panda, prairie, quartz, raven, river, sable, salmon, sparrow,
spruce, sumac, tundra, vista, walnut, willow, wombat, yarrow, yew, zephyr

### Step 2.5: Enter a worktree (if `--worktree` was passed)

If `--worktree` was passed, call `EnterWorktree` with `name` set to the resolved `SESSION_NAME` (e.g. `2026-05-04--locate-weekdays--pensive-mongoose`). The session-name format uses only characters allowed by `EnterWorktree` (letters, digits, dots, dashes). Typical names land around 45 chars and fit within the 64-char limit; if a pathologically long topic pushes the session name past 64 chars, truncate the topic (preserving the leading words) before passing it to `EnterWorktree` while keeping the full name for the `agent_logs/` directory.

After this call, the session's CWD is `<repo>/.Codex/worktrees/<SESSION_NAME>/`. All subsequent steps run inside the worktree, and the relative paths in Steps 3–6 (`agent_logs/...`) resolve against the worktree's checkout. The new branch (auto-named by `EnterWorktree`) is the place where this session's commits will land — promotion to shipped paths happens when the branch is merged back, not via a `git mv` on `main`.

Skip this step entirely if `--worktree` was not passed — the session is created in the main checkout as before.

If already inside a worktree (e.g. the user manually entered one before invoking the skill), skip the `EnterWorktree` call and continue — the session dir will be created in the existing worktree.

### Step 3: Create the session directory

```bash
SESSION_NAME="$(date +%Y-%m-%d)--{topic}--{adj}-{noun}"
mkdir -p "agent_logs/${SESSION_NAME}/plan"
mkdir -p "agent_logs/${SESSION_NAME}/run"
mkdir -p "agent_logs/${SESSION_NAME}/result"
```

### Step 4: Write `README.md`

Write `agent_logs/{SESSION_NAME}/README.md` with this structure (single paragraph + cheatsheet):

```markdown
# {SESSION_NAME}

{One-paragraph statement of intent — what this session is investigating, why, and the expected
deliverable. In autonomous mode, derive this from the research context. In interactive mode, ask
the user for one sentence after creating the directory and append it here.}

## Layout

- `plan/` — research objective, task-spec drafts, approval-checkpoint logs
- `run/` — resolved-config snapshot (`--cfg job` output), `run.log`, slurm logs
- `result/` — `REPORT.md` (single consolidated interpretation written by `/interpret-experiment`), `figures/` for embedded plots/tables
- `code/` — session-local Python + Hydra (via `/setup-methods`, `/setup-analyses`, `/run-experiment`)
- `artifacts/` — raw experiment outputs at `{task}/{model}/{analysis}/...`
- `issues.md` — top-level issue log spanning all phases (managed by `/document-issues`)
```

### Step 5: Touch `issues.md`

```bash
touch "agent_logs/${SESSION_NAME}/issues.md"
```

Empty file ready for `/document-issues` to populate.

### Step 6: Write the marker file

```bash
echo "${SESSION_NAME}" > agent_logs/.current
```

### Step 7: Print the session path

```
Research session created: agent_logs/{SESSION_NAME}/
```

If `--worktree` was used, also print the worktree path so the user knows where to look on disk:

```
Worktree: .Codex/worktrees/{SESSION_NAME}/
(use ExitWorktree when done — `keep` to preserve the branch, `remove` to discard)
```

### Step 8: Hand off based on whether a research objective is in context

Inspect the surrounding conversation for a research objective. An objective is "in context" if any of the following is true:

- the user's invoking message contains a clearly stated research question, hypothesis, or goal beyond the topic slug,
- the user provided a path to a research-objective markdown file,
- prior turns in this conversation already laid out an objective the user wants to act on.

The topic slug alone (e.g. `locate-weekdays`) does **not** count — that's a label, not an objective.

**If an objective is in context:**

Immediately invoke `/plan-experiment` and pass the objective through (as a markdown blob or path, per that skill's input contract). Do not print a menu — just announce the handoff in one sentence and invoke the skill.

**If no objective is in context:**

Print exactly these two options and stop:

```
No research objective detected. Pick one:

  1. Formulate a research objective, then run /plan-experiment to crystallize it
     into RESEARCH_OBJECTIVE.md and PLAN.md.
  2. Run /getting-started to learn what causalab can do (tasks, analyses,
     workflow) before committing to an objective.
```

Wait for the user's choice. Do not auto-invoke either skill in this branch.

## README intent paragraph (interactive mode)

After Step 4 in interactive mode, prompt:

> "One sentence: what is this session investigating, and what's the deliverable?"

Append the user's response to `README.md` between the heading and the layout cheatsheet.

In autonomous mode, generate the intent paragraph from the surrounding research context without prompting.

## Restrictions

- **Raw experiment artifacts live under `${SESSION_DIR}/artifacts/{task}/{model}/`** in research mode (per `CONVENTIONS.md` "Output routing" and `ARCHITECTURE.md` §3 invariant 7). The session is a self-contained bundle.
- **Skills write only to the active session dir.** Files outside `agent_logs/{SESSION_NAME}/` are never edited automatically — promotion to `causalab/` is a manual `git mv` step, except for `/setup-task`'s explicit approval-gated package scaffold (see `CONVENTIONS.md` "Core invariant").
- **Auto-invoke only `/plan-experiment`, only when an objective is in context** (see Step 8). All other downstream skills (`/setup-task`, `/run-experiment`, `/interpret-experiment`, …) must be invoked by the user — they each check `.current` themselves at Step 0. (`/interpret-experiment` is the one exception that auto-invokes from `/run-experiment` Step 9 once an experiment finishes.)
- **Do not create a session for codebase-only questions** (typo fixes, refactors, dependency bumps). This skill is for research workflows only.
- **Topic must describe research focus.** Reject placeholders like `auto`, `untitled`, `test`, `tmp`, `foo`.
