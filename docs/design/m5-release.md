# M5, second half: the examples site and release prep

Written 2026-11-10 by the lead (Mac), after the coverage pass (T5.1-T5.5). SPEC M5's last two lines: "Examples site" and "Release prep for a pmndrs-style public home". Prototype level, as the coverage pass: each piece works end to end and passes one check; nothing is tuned.

The model for both is `@pmndrs/upscaler` (`~/Developer/upscaler`), Dennis's sibling package: its numbered examples gallery on GitHub Pages, its CI, and its npm publish workflow with Trusted Publishing. Copy what fits rather than invent.

**Nothing is published in this pass.** No npm publish, no tag, no repo transfer, no Pages deploy turned on. The workflows are written and checked locally; Dennis turns them on (DECISIONS: outward-facing actions need his OK).

## Defaults the lead chose (Dennis can overrule)

1. **Hosting:** GitHub Pages from this repo, as upscaler does. Today that is `dennissmolek.github.io/webgpu-ddgi/`; after the move to `pmndrs/ddgi` it is `pmndrs.github.io/ddgi/`. The base path comes from `PAGES_BASE`.
2. **Only assets we can redistribute and that fit Pages** (1 GB a site, 100 MB a file). The site uses the procedural scenes (ours, MIT) and three primitives. No Sponza (Intel's licence forbids redistribution, DECISIONS), no Bistro (CC BY 4.0, but far too big). The Poly Haven tree (CC0) only if it fits under 20 MB.
3. **Version:** the package stays `0.0.0` in the tree. The first publish is `0.1.0`, after v1's sign-off, cut by the publish workflow.
4. **`@pmndrs/bvh` stays bundled** into `@pmndrs/ddgi` until pmndrs/bvh publishes its own package. Then it becomes a dependency (a later task, coordinated with that package's agent).
5. **The internal docs stay** (SPEC, PLAN, TASKS, PROGRESS, DECISIONS, NOTES, the design notes). They are how the work runs. The README links only the public docs (getting started, the API docs, the adapter guide, examples, THIRD_PARTY).

## T5.6: the examples site

**What:** a gallery of small, single-purpose pages, each the shortest code that shows one feature, with a landing page that links them all. It replaces `examples/three` (whose two pages become examples 01 and 02) and links `examples/webgpu/hello` as the no-engine example.

- **Layout:** as upscaler's: `examples/index.html` (the landing page), `examples/NN-name/{index.html,main.ts}`, `examples/shared/` for the few helpers every page needs (renderer and camera setup, a small stats line, the procedural scene loader), `examples/vite.config.ts` with the library aliased to its source (`@pmndrs/ddgi`, `/core`, `/three`), `examples/README.md` with a table of what each one shows. `pnpm examples` serves it; `pnpm examples:build` builds it into `examples/dist` with `PAGES_BASE`.
- **Each `main.ts` reads like user code:** imports from `@pmndrs/ddgi/three`, no test hooks, no imports from `tools/`. Scenes come from the procedural builders in `scenes/` (or three primitives), through one shared helper.
- **The set:**
  1. `01-hello`: `GIVolume` in a lit room, the dozen lines from the README's getting started.
  2. `02-composite`: the drop-in composite with `ddgi()` and `@pmndrs/upscaler` (today's `composite.html`).
  3. `03-lights`: move, dim, recolour and toggle lights in Cornell or two-room; GI follows with no blackout.
  4. `04-moving-objects`: a door swinging and objects moved in two-room or the cabinet row; the far room stays dark.
  5. `05-time-of-day`: the courtyard with the sun animated and the sky.
  6. `06-materials`: the glass wall, the alpha checker and the uv board (textures at GI hits).
  7. `07-presets`: `low`, `medium`, `high` switched live, with probes, memory and GI time shown.
  8. `08-saved-bake`: save the field, reload the page from it, and start with finished GI.
  9. `09-lightprobegrid`: the export to three's `LightProbeGrid`, wiped against live GI.
  10. `10-debug`: probes, their states and the irradiance atlas, through the existing debug outputs.
  - `webgpu/hello`: plain WebGPU, no engine (the adapter guide's example), built into the same site.
- **A Pages workflow** (`.github/workflows/deploy-pages.yml`), as upscaler's: on push to `main` and by hand. It is committed but deploys only once Dennis enables Pages for the repo.

**Done when:**
- every example runs on the M5 Pro in headless Chrome, with GI visible and no errors in the console: one GPU page (extend `three-examples`, or a new `examples-site` page) loads each built example and checks that it reports a frame with non-black GI and no errors;
- `pnpm examples:build` with `PAGES_BASE=/webgpu-ddgi/` builds a site whose links and assets resolve under that base (served locally from a subdirectory and loaded by the same page);
- the built site is under 50 MB;
- screenshots of each example are looked at by the lead, then by Dennis.

## T5.7: release prep

**What:** everything a pmndrs package needs before its first publish, short of publishing.

- **The tarball:** `packages/ddgi` gets its README (the root README's user-facing half, or a copy step at build), `LICENSE` and `THIRD_PARTY.md` (tinybvh's MIT text must ship with the WASM; it is already embedded in the bundle, and the notice file says so). `homepage` and `bugs` in `package.json`, as upscaler's.
- **A packed-consumer check,** as upscaler's `verify-packed-subpaths`: `npm pack` the package, install the tarball in a temporary project with only `three@0.187` and `@webgpu/types`, and check that `@pmndrs/ddgi`, `/core` and `/three` import under Node, and that a small TypeScript file using each typechecks against the packed types. `/core` must also resolve with three not installed. It runs in `pnpm test` (or as its own script CI calls).
- **CI** (`.github/workflows/ci.yml`): on pull requests and pushes to `main`, Ubuntu and Windows: install, lint, typecheck, unit tests, build, the packed-consumer check. No GPU checks: GitHub's runners have no GPU, and the GPU suite stays on the target and correctness devices (say so in the workflow and in `docs/releasing.md`).
- **The publish workflow** (`.github/workflows/publish.yml`), adapted from upscaler's: a `vX.Y.Z` tag, or a manual run with an explicit version, publishes `packages/ddgi` with npm Trusted Publishing (OIDC, no token) and provenance, then creates a GitHub Release. Our commit messages aren't Conventional Commits, so drop the `auto` bump: the version is always given. Release notes come from a hand-written `CHANGELOG.md` section for that version, not from commits. `prepublishOnly` runs lint, typecheck, tests and build.
- **`CHANGELOG.md`** with an unreleased `0.1.0` section: what v1 contains, in plain words, from the milestone notes.
- **`docs/releasing.md`:** how to cut a release, what Dennis has to do once (register the trusted publisher on npmjs.com for `pmndrs/ddgi` and `publish.yml`, move the repo to `pmndrs`, enable Pages), and what CI does and doesn't check.
- **The README as a public home,** pmndrs style: one-line description, a link to the examples site, a screenshot, install, the getting-started (already there), browser support (WebGPU: Chrome and Edge 113+, Safari 26+; the target and correctness devices), the docs links, credits and licence. Internal workflow notes move below a "Working on this repo" heading.

**Done when:**
- the packed-consumer check passes locally;
- the CI workflow's steps all pass when run locally in order (`act` is not required);
- `publish.yml` and `ci.yml` are valid YAML and pass `actionlint` if it is installed (say if it isn't);
- the tarball holds README, LICENSE and THIRD_PARTY, and its size is recorded in PROGRESS;
- Dennis reads the README and `docs/releasing.md`.

## T5.8: small leftovers (lead, inline)

From TASKS Later: export `DDGILightGroups`; fix the stale comments in `SceneApi.updateMesh` and the procedural sky. Done when `pnpm test` (with docs-cover-exports) and typecheck pass.
