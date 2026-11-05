/** B1 profile: where the GPU LBVH's build time goes, on a real scene (../bvh-shared/build-profile.ts). */
import { runBuildProfile } from '../bvh-shared/build-profile.ts';

await runBuildProfile('lbvh');
