# tools

Scripts that are not part of the library: the GPU test runner (T0.2), the native-harness wrappers (T0.6), the reference and compare tools (T0.7). Empty until those tasks land.

`release/` (T5.7): what packs and publishes `@pmndrs/gi`. `package-files.ts` copies the README and LICENSE into the package at build; `verify-packed.ts` is the packed-consumer check (`pnpm verify:packed`); `release-version.ts` and `release-notes.ts` are run by `.github/workflows/publish.yml`, whose steps `publish-workflow.test.ts` runs locally (`pnpm test:release`). See docs/releasing.md.
