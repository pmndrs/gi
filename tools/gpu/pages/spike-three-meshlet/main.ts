/**
 * T0.8, meshlet drawing pipeline: our GI light node adds the expected irradiance.
 * See ../spike-three-lib/pipeline-spike.ts for what is checked.
 */

import { runPipelineSpike } from '../spike-three-lib/pipeline-spike.ts';

await runPipelineSpike('meshlet');
