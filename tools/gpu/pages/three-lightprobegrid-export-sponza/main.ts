/**
 * T5.3: Sponza lit by the exported `LightProbeGrid` against our live GI
 * (three-lightprobegrid-shared/export-check.ts has the steps and the bars).
 * Manual: it needs the downloaded Sponza.
 */

import { runThreePage } from '../three-shared/harness.ts';
import { CANVAS_SIZE, runExportCheck } from '../three-lightprobegrid-shared/export-check.ts';

await runThreePage(async (context) => {
    // Sponza has about 19,400 probes: every probe every 5 frames, so 300 frames are 60 passes.
    await runExportCheck(context, { sceneId: 'sponza', giFrames: 300, probesPerFrame: 4096, lightGroups: false });
}, CANVAS_SIZE);
