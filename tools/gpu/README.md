# GPU test runner

```
pnpm test:gpu [--gpu nvidia|intel] [--filter name] [--json file] [--verbose] [--timeout ms] [--out dir] [--chrome path] [--no-webgpu]
```

It starts a vite server for the pages in `pages/`, launches your installed Chrome headless on the GPU you ask for, runs each page in turn, and exits non-zero if anything is wrong. It prints the adapter (`vendor=nvidia architecture=ampere`), one line per page, and a screenshot path.

It fails when:

- WebGPU is missing or `requestAdapter()` gives nothing (clear message, not a timeout);
- Chrome gives a different GPU than `--gpu` asked for, or a software fallback adapter (checked on the runner's own probe and again on the adapter each page reports);
- a page reports failure, throws, hangs past its timeout, or crashes;
- the browser logs any error, an uncaught exception, or a WebGPU/WGSL validation warning (collected from `Log.entryAdded`, `console.*` and `Runtime.exceptionThrown`), even if the page itself said "pass".

A run with no `--filter` also runs the runner's own self-tests. The `wgsl-error` page (a deliberately broken shader) must fail with the message its `test.json` names, and a Chrome started with the GPU disabled must fail with "WebGPU is not available". If either is not caught, or the wrong failure happens (a timeout, say), the run fails. `--filter wgsl-error` runs that page as an ordinary test, so it fails and shows the shader error text.

How the GPU is chosen is explained at the top of `chrome.ts`. In short, headless Chrome gives NVIDIA by default, and Intel is reached by pinning Chrome to the adapter's LUID, which the runner looks up from Windows device properties each run (about 2 s). Nothing in Windows settings or the registry is touched. `--gpu intel` needs Windows; it says so, and says which devices it found, if the lookup fails.

## Writing a test page

Make a folder `pages/<name>/` with an `index.html` and a script. The folder name is what `--filter` matches (substring). Copy `pages/smoke/`.

```html
<!doctype html>
<html><head><meta charset="utf-8" /><title>my-test</title></head>
<body><script type="module" src="./main.ts"></script></body></html>
```

```ts
import { assert, runGpuTest } from '../../client.ts';

await runGpuTest(async ({ adapter, device, log }) => {
    // ... create shaders, dispatch, read back ...
    assert(result === expected, 'what was wrong');   // or just throw
    return { milliseconds: 3.2 };                    // optional: printed by the runner
}, { optionalFeatures: ['timestamp-query'] });
```

`runGpuTest` (in `client.ts`) requests the adapter and device, records every `uncapturederror` with `addEventListener`, runs your function, drains the queue so late validation errors show up, and writes the report. You may import anything else vite can serve, including the repo's packages.

### When something else owns the device (three.js, Babylon)

Use the low-level helpers instead of `runGpuTest`. They are what `runGpuTest` is built from:

```ts
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';

const startTime = performance.now();
const renderer = new WebGPURenderer();
await renderer.init();
const device = renderer.backend.device as GPUDevice;
const deviceErrors = watchDevice(device);          // addEventListener, never onuncapturederror
try {
    // ... render, read back, compare ...
    await drainDevice(device);                     // queue idle + 50 ms for late errors
    reportGpuTest({ status: 'pass', data: { ms: 3.2 }, deviceErrors, startTime,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features) });
} catch (error) {
    reportGpuTest({ status: 'fail', message: String(error), deviceErrors, startTime });
}
```

`reportGpuTest` publishes `window.__gpuTestReport` and draws the summary. It fails the test if `deviceErrors` is not empty. Passing the adapter lets the runner check it against `--gpu`.

### The contract (`contract.ts`)

- The runner opens `/pages/<name>/index.html?token=<id>&gpu=<nvidia|intel>`. The `gpu` parameter is only present when `--gpu` was given.
- The page ends by setting `window.__gpuTestReport` to a `GpuTestReport`: `{ token, status: 'pass' | 'fail', message?, data, adapter, deviceErrors, durationMs }`. `token` must echo the query parameter, so a stale page cannot answer for the current one. `runGpuTest` and `reportGpuTest` do all of this.
- `data` is free-form JSON. The runner prints it and writes it to `--json`. The runner reads the report through DevTools `returnByValue`, which turns typed arrays into plain objects keyed by index (`{"0": 1, "1": 2}`). `reportGpuTest` converts typed arrays and bigints for you; if you write the report by hand, convert them to arrays first.
- A page with no report by its deadline is a timeout failure. Default is 20 s.
- Anything the browser logs that looks like an error fails the page regardless of the report (see the list above), so you cannot forget to check for WGSL errors.
- Opening the page by hand in Chrome works too: it draws a green or red summary.

### `test.json` (optional)

```json
{ "description": "One line shown in the output.", "timeoutMs": 60000, "expectFailure": false, "expectMessage": "Error while parsing WGSL" }
```

`expectFailure: true` marks a self-test of the runner: in a full run it passes only if the runner catches a failure. `expectMessage` (required in practice, or any failure counts) is a substring that one failure reason must contain, or a regular expression written as `/pattern/flags`. Use self-tests sparingly; `wgsl-error` is the model.

## `--json <file>`: results for other tools

Writes the whole run as JSON (`RunSummary` in `contract.ts`): `{ schema, startedAt, gpuRequested, adapter, ok, totalMs, error, pages: [{ name, outcome, acceptable, durationMs, reasons, data, adapter, screenshot }], selfTests }`. Benchmark tasks take `pages[].data` from it and turn it into CSV rows, with no copying from the console. The file is written even when the run fails.

## Notes

- Errors in a shader module surface only after the queue has drained. The page's drain window is `onSubmittedWorkDone()` plus 50 ms (`DRAIN_GRACE_MS` in `client.ts`). A validation error that arrives later than that is missed by the page's own check; the runner still sees it if it reaches the browser log before the 100 ms it waits after the report, but later than that it can be missed too. A page that creates a bad shader and exits at once can slip through; `runGpuTest` waits for you.
- Screenshots go to a temp folder (printed at the end) or `--out`.
- Chrome, its throwaway profile folder and the vite server are stopped and removed on normal exit, on a failed run, on Ctrl-C (SIGINT), and on SIGTERM, an uncaught exception or an unhandled rejection. They are registered for cleanup the moment they exist, including while Chrome is still starting and while the Intel GPU is being looked up.
  - Windows only delivers real signals for Ctrl-C, Ctrl-Break and closing the console. Another process calling `taskkill` or `process.kill(pid)` on the runner kills it outright with no cleanup, as does power loss. A leftover `ddgi-gpu-chrome-*` folder in your temp directory, or a headless `chrome.exe` using one, is such a case; delete the folder and end the process.
- Tests of the runner's pure logic (`*.test.ts` here) run in `pnpm test`; they need no GPU.
