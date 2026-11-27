# Releasing @pmndrs/gi

**Merging to `main` never publishes.** A release happens only when you ask for one, and it runs entirely in GitHub Actions ([`publish.yml`](../.github/workflows/publish.yml)), with no step on your computer. The workflow is adapted from `@pmndrs/upscaler`'s, with two differences:

- **You always say the version.** Upscaler can work out the next version from its commit messages ("Conventional Commits"). Ours aren't written that way, so there is no `auto`: the version comes from the tag, or you type it.
- **The release notes are written by hand,** in [CHANGELOG.md](../CHANGELOG.md): one `## X.Y.Z` section per version. The workflow refuses a version without one.

The published package, `@pmndrs/gi`, is `packages/ddgi`, not the repository root (which is a private workspace). The folder keeps its old name until the workspace folders are renamed ([DECISIONS](DECISIONS.md), 2026-11-10). The repository installs with pnpm; versioning and publishing use the npm CLI, because npm's Trusted Publishing is an npm CLI feature.

**Nothing has been published yet, and the workflow can't publish until the one-time steps below are done.**

## Once, before the first release (Dennis)

1. **Make the repository `pmndrs/gi` public.** It exists (private) and is `origin` since 2026-11-10; `packages/ddgi/package.json`, the README and the package's links already point at it. npm checks that `repository.url` matches the repository the workflow runs in, so publish only from `pmndrs/gi`. npm attaches provenance only for public repositories ("Provenance generation is not supported for private repositories", npm's trusted-publishing docs), and the README's links and the examples site need it public too.
2. **Register the trusted publisher on npmjs.com**, for `@pmndrs/gi`, under the package's Settings → Trusted publishing, as a GitHub Actions publisher:
   - Organization or user: `pmndrs`
   - Repository: `gi`
   - Workflow filename: `publish.yml`
   - Environment: leave empty
   This needs publish rights in npm's `pmndrs` organisation. There is no token or secret, and npm attaches provenance by itself. npm says a new trusted-publisher setting must publish within 2 days, or it lapses.

   **Not checked:** whether npmjs.com lets you register a publisher for a package name that has never been published. Its settings page belongs to an existing package. If it doesn't offer it for `@pmndrs/gi`, publish `0.1.0` once by hand from a clean checkout of the tagged commit (`pnpm install --frozen-lockfile`, then in `packages/ddgi`: `npm version 0.1.0 --no-git-tag-version` and `npm publish --access public`, which runs every check first), then register the publisher. Every later release goes through the workflow. The run for that tag then finds 0.1.0 already on npm, skips publishing, and still creates the GitHub Release.
3. **Turn on GitHub Pages** for the examples site (Settings → Pages → Source: GitHub Actions). That is [`deploy-pages.yml`](../.github/workflows/deploy-pages.yml), not this workflow; it builds with `PAGES_BASE=/gi/`, for `https://pmndrs.github.io/gi/`.
4. **Check that Actions may push to `main`.** The workflow pushes a version commit (and, from the button, a tag) with its own token. `main` has no branch protection today. If protection is added, allow GitHub Actions to bypass it, or the button and the catch-up commit fail (a tag you push yourself still publishes).

## Cutting a release

1. **Write the notes.** In `CHANGELOG.md`, write the version's section (`## 0.1.0 (2026-12-01)`): what's new, in plain words. Replace "(unreleased)" with the date. Merge it to `main`.
2. **Run the GPU checks on the devices** (CI can't, see below): at least `pnpm test:gpu` on a target device, and on a correctness device if anything GPU-side changed since the last release. Note the result in PROGRESS.
3. **Ask for the release,** in one of two ways.

### Option A: the Run workflow button

**Actions → Publish to npm → Run workflow**, on branch `main`, with `version` set to the exact version: `0.1.0`, or `0.2.0-beta.0` for a beta.

In one run, the workflow:

1. refuses if the version isn't a version (`patch` or `auto` won't do), if `CHANGELOG.md` has no section for it, if it isn't newer than `packages/ddgi/package.json`'s, or if the run wasn't started from `main`;
2. commits `release: vX.Y.Z [skip ci]` to `main` (only `packages/ddgi/package.json` changes), tags it `vX.Y.Z`, and pushes both together. If `main` moved while the run was going, nothing is pushed: run it again;
3. publishes to npm and creates the GitHub Release, as for a tag.

The commit and tag are pushed with the workflow's own token, and pushes made with it never start workflows, so the tag doesn't start a second run.

### Option B: a tag (including one made in the GitHub UI)

**Releases → Draft a new release → Choose a tag → type `vX.Y.Z` → Create new tag on publish**, target `main`, then **Publish release**. Pushing a `vX.Y.Z` tag on a `main` commit from anywhere works the same way.

**The tag is the version.** The tagged commit's `package.json` still has the previous version; the workflow stamps the tag's version into its own checkout and publishes that. Afterwards it brings `main` up to date with a separate `release: vX.Y.Z [skip ci]` commit: a plain fast-forward, never a force push, never a new tag. It skips that when `main` is already at or past the version, and retries up to three times if a merge lands at the same moment.

**Your Release text is kept.** "Draft a new release" creates the GitHub Release together with the tag, so the workflow finds it and leaves it as you wrote it. It creates a Release (with the CHANGELOG section as its text) only when the tag has none, and leaves a draft alone for you to publish.

### Betas

A version with a suffix, like `0.2.0-beta.0`, is a prerelease. It uses its own CHANGELOG section if there is one, otherwise the section of the version it leads up to (`## 0.2.0`), which may still say "(unreleased)". It publishes under the npm dist-tag `beta` (so `npm install @pmndrs/gi` doesn't get it; `npm install @pmndrs/gi@beta` does) and becomes a GitHub prerelease.

## What every release run checks

Before publishing anything, the run fails if:

- the tag isn't `v` followed by a version (SemVer);
- the tagged commit isn't on `main`, so a tag on a side branch never ships;
- `CHANGELOG.md` at the tagged commit has no section for the version.

Then it:

- **chooses the npm dist-tag** (below). For a stable version it reads npm's current `latest` first; if that fails for any reason but "no such package" (a first publish), it stops before publishing.
- **publishes** `packages/ddgi` with `npm publish --access public --tag <dist-tag>`. That runs the package's `prepublishOnly` gate first: lint, typecheck, unit tests, build, and the packed-consumer check. A version already on npm is skipped, not published again.
- **creates the GitHub Release** if the tag has none, from the CHANGELOG section. Preview the text with `node tools/release/release-notes.ts --version 0.1.0`.

Every step is safe to repeat. If a run failed after npm took the package (say, at the GitHub Release), use **Re-run jobs**, or run the button with the same version: when its tag exists, the run creates nothing, publishes only if npm lacks the version, creates the Release only if it's missing, and catches `main` up only if it's behind.

### Versions and dist-tags

| Version | npm dist-tag | Install | GitHub Release |
| --- | --- | --- | --- |
| `0.2.0` (the newest) | `latest` | `npm i @pmndrs/gi` | latest |
| `0.1.1` (after `0.2.0`) | `v0.1-latest` | `npm i @pmndrs/gi@v0.1-latest` | not latest |
| `0.2.0-beta.0` | `beta` | `npm i @pmndrs/gi@beta` | prerelease |
| `0.2.0-rc.0` | `rc` | `npm i @pmndrs/gi@rc` | prerelease |
| `0.2.0-0` | `next` | `npm i @pmndrs/gi@next` | prerelease |

`latest` only moves forward: a fix for an older line goes to that line's tag, so `npm install @pmndrs/gi` never goes back a version. The rules are in [`tools/release/release-version.ts`](../tools/release/release-version.ts).

The tree keeps `packages/ddgi/package.json` at `0.0.0` until the first release: the first publish is `0.1.0`, after v1's sign-off ([docs/design/m5-release.md](design/m5-release.md), the lead's defaults).

## What CI checks, and what it doesn't

[`ci.yml`](../.github/workflows/ci.yml) runs on every pull request and every push to `main`, on Ubuntu and Windows:

1. `pnpm install --frozen-lockfile`
2. `pnpm lint`
3. `pnpm typecheck` (including the docs' code examples)
4. `pnpm test` (the unit tests, including the docs-cover-exports check)
5. `pnpm test:release` (Ubuntu only): `publish.yml`'s steps, run against throwaway git repositories with a pretend npm and GitHub, so a broken release workflow shows up before a release
6. `pnpm build`
7. `pnpm verify:packed --skip-build`: the packed-consumer check (below)

**CI runs no GPU checks.** GitHub's hosted runners have no GPU, so WebGPU can't start there, and nothing in CI shows that GI renders correctly or how fast. The GPU suite (`pnpm test:gpu`, about 85 pages) runs on the target devices (the RTX 3060 Laptop, the M5 Pro) and the correctness devices (the Intel UHD, later the iPad), by hand, as each milestone note records. That is step 2 of "Cutting a release".

### The packed-consumer check

`pnpm verify:packed` ([`tools/release/verify-packed.ts`](../tools/release/verify-packed.ts), adapted from upscaler's `verify-packed-subpaths`) checks the package as a user gets it:

- it builds the package and packs it with `npm pack`, the tool that publishes;
- the tarball must hold `README.md`, `LICENSE`, `THIRD_PARTY.md`, the bundled code's licence texts (`licenses/`) and every file `exports` names, and its manifest must ask for nothing from this workspace;
- `@pmndrs/gi/core` and `/ddgi` may import no package at all, `@pmndrs/gi` and `/three` only three, `/radiance-cascades` only `@pmndrs/rc-bvh`, and `/radiance-cache` only `denoiser/ffx`;
- `@pathtracer/mlp` (the radiance cache's optional neural tail, not on npm) is never in the manifest, never imported by a `.d.ts`, and loaded only through the `import()` the build hides from bundlers (`packages/ddgi/vite.config.ts`; `scripts/verify-artifacts.mjs` checks that at build time);
- each subpath loads only its own layers' chunks (`packages/ddgi/layer-chunks.ts`): `/core` only the platform's, so it holds no technique code, `/ddgi` no three code, and `/radiance-cache` only the platform's and its own;
- in a new folder, npm installs the tarball with only `three`, `@types/three` and `@webgpu/types`; `@pmndrs/gi`, `/core`, `/ddgi`, `/three` and `/radiance-cache` must import under Node (`.` and `/three` giving the same `DDGIVolume` class), and a file using each (the radiance cache wired into three from public paths only) must typecheck against the packed types, under both `bundler` and `nodenext` module resolution;
- in another new folder, with no three at all, `/core`, `/ddgi`, `/radiance-cascades` and `/radiance-cache` must install, import and typecheck, and no name may be exported by both `/core` and `/ddgi`;
- a new vite app importing `/core`, `/ddgi`, `/three`, `/radiance-cascades` and `/radiance-cache` must build with `vite build` and run under vite's dev server, and every file its code fetches with `new URL(..., import.meta.url)` must be there. Typechecks and Node imports run no bundler, so they missed BVH workers the package build had pointed at `/assets/` (2026-11-27);
- neither folder has `@pathtracer/mlp`.

It installs from the npm registry, so it needs the network. `--keep-temp` keeps the folders for a look.

## What's in the package

`files` in `packages/ddgi/package.json` lists it: `dist/` (the six entries `index`, `core`, `ddgi`, `three`, `radiance-cascades` and `radiance-cache`, their types, one chunk per layer (`platform`, `ddgi`, `three`, `rc`, `radiance-cache`); the two BVH workers are built into the platform chunk, not shipped as files), `README.md`, `LICENSE`, `THIRD_PARTY.md` and `licenses/`. The build (`pnpm build`) copies the README and LICENSE from the repository root ([`tools/release/package-files.ts`](../tools/release/package-files.ts)):

- **README.md** is the root README down to the line `<!-- The npm package README ends here. ... -->`; everything below it ("Working on this repository") stays on GitHub. Relative links become absolute `github.com/pmndrs/gi` links, since npmjs.com can't follow them. Edit the root README, never the copy (it's git-ignored).
- **THIRD_PARTY.md** is the package's own (`packages/ddgi/THIRD_PARTY.md`, committed): only the code that ships in it: tinybvh's WebAssembly with the Emscripten runtime, the GPU tree builders ported from tinybvh and HIPRT, Split RC's raytracer code, and the radiance cache's port of AMD Capsaicin (with the platform's SSGI pass). The root `THIRD_PARTY.md` also lists test scenes and tools, which don't ship.
- **licenses/** is copied from `packages/bvh/licenses/` and each technique's own `licenses/` folder (`radiance-cascades`, `radiance-cache`). The tinybvh and HIPRT texts are also inside the bundle, in `/*! */` comments that bundlers keep.

`@pmndrs/bvh` (the tree builders) is bundled into the package until pmndrs/bvh publishes its own; then it becomes a dependency.
