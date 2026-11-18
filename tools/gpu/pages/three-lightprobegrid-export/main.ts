/**
 * T5.3: the point-light Cornell box lit by the exported `LightProbeGrid`
 * against our live GI (three-lightprobegrid-shared/export-check.ts has the
 * steps and the bars). DDGI runs with one light group here, so the export
 * reads the composed atlas, and the page checks it follows the group's weight.
 */

import { runThreePage } from '../three-shared/harness.ts';
import { CANVAS_SIZE, runExportCheck } from '../three-lightprobegrid-shared/export-check.ts';

await runThreePage(async (context) => {
    // Cornell has 1,728 probes: every probe every frame, for 300 frames.
    await runExportCheck(context, { sceneId: 'cornell-box-point', giFrames: 300, probesPerFrame: 4096, lightGroups: true });
}, CANVAS_SIZE);
