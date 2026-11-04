# Progress log

Newest entries go at the top. One entry per work session on a task, under 15 lines. This is how agents hand work to each other across contexts, so write for someone with no memory of the conversation.

Template:

```
## YYYY-MM-DD · T0.x · agent (model)
Did: what changed, with file paths.
Verified: the command or check, and its result. Say which device and browser.
Not verified: anything not checked, and why.
Numbers: the measurements, or a link to the CSV rows.
Next: what's left or blocked, and who should take it.
```

Rules:

- "Verified" means you ran it and saw it pass. "It builds" is not "it works".
- Keep numbers here or in `bench/results/*.csv`. Don't create report files.
- If you learned something that will matter again, add it to `docs/NOTES.md` too.

---

## 2026-11-04 · setup · main session (opus)

Did:
- Created the repo with the spec, the audit, the research notes and the native RTXGI submodule.
- Analyzed three.js r187, HomeFig's DDGI branch, `@pmndrs/upscaler` and `@pmndrs/denoiser`.
- Wrote `PLAN.md`, `TASKS.md`, `docs/NOTES.md`, `docs/DECISIONS.md`, `CLAUDE.md` and the subagent definitions in `.claude/agents/`.

Installed on the Windows laptop:
- RenderDoc 1.47.0, via winget;
- NVIDIA FLIP (`flip-evaluator` 1.7, via pip).

Already present: PIX 2603.25, Blender, Vulkan SDK 1.3.283, Nsight Systems 2022.4, Nsight Compute 2022.3.

Not done: Nsight Graphics, which needs an NVIDIA developer-account login. Dennis to install it from developer.nvidia.com/nsight-graphics.

Next: T0.1 (implementer). T0.11 can start in parallel in the main session.
