# pmndrs/gi: start here

A WebGPU-native GI platform with DDGI as its first technique (AMD GI-1.0's radiance cache is the second, on its own branch), published as `@pmndrs/gi` from the repo `pmndrs/gi` (DECISIONS 2026-11-10). The workspace folders and private package names still say `ddgi` until the radiance-cache branch merges. The core is plain WebGPU and WGSL, and engines plug in through adapters; three.js r187 is first. Sibling libraries: `@pmndrs/upscaler`, `@pmndrs/denoiser`.

## Read in this order

1. **This file.**
2. **[TASKS.md](TASKS.md):** find your task row. It is your scope.
3. **The files your task row links,** plus the matching sections of [SPEC.md](SPEC.md) (what and why) and [PLAN.md](PLAN.md) (order, and how "done" works).
4. **The last few entries for your task in [PROGRESS.md](PROGRESS.md):** what the previous agent did and left.
5. **[docs/NOTES.md](docs/NOTES.md):** facts already learned. Check before you investigate something.
6. **[docs/DECISIONS.md](docs/DECISIONS.md):** settled choices. Don't reopen them inside a task. Raise a conflict in PROGRESS instead.

## When you finish a session

1. Add a PROGRESS entry using the template there. Say plainly what you verified, on which device and browser, and what you didn't.
2. Update your task's status in TASKS.md (`review` when you think it's done; the main session marks `done`).
3. Add any durable fact to NOTES.md.
4. Put numbers in PROGRESS or `bench/results/*.csv`. Never in new report files.

## Rules

- **Stay inside your task.** Note anything else you find in PROGRESS under "Next".
- **Respect the boundary:** `ddgi-core` never imports an engine, and lint enforces it.
- **Readable code:** normal formatting, names that say what they are, and comments that explain *why*. No minified one-liners, magic parameter indices or test hooks in production shaders.
- **Verify GPU work on a real GPU** with `pnpm test:gpu` and say which adapter. "It builds" is not "it works".
- **No checks that only check other reports.** No byte-identical-repeat requirements, since temporal DDGI is stochastic.
- **Changing a threshold or target is a decision:** add it to DECISIONS.md with the reason.
- **Plain language** in docs, UI text and status: Dennis reads these.

## Agents

Subagent definitions are in [.claude/agents/](.claude/agents/). [PLAN.md](PLAN.md#how-work-flows) lists which agent does what.

The main session (strongest model) owns design, task breakdown and review. Cheaper agents do scoped work.
