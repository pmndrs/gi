import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));

function entry(name, relative) {
  const checkout = process.env[name];
  if (!checkout) throw new Error(`Set ${name} to the package checkout (see README.md).`);
  const file = resolve(checkout, relative);
  if (!existsSync(file)) throw new Error(`Missing ${file}. See README.md for the tested revisions.`);
  return file;
}

const tracer = entry('RC_PATHTRACER_ROOT', 'packages/raytracer/src/index.ts');
const bvh = entry('RC_BVH_ROOT', 'dist/index.js');
// A normal pathtracer install supplies this dependency; an extracted source
// checkout can instead point at a built sibling.
const downsampler = process.env.RC_DOWNSAMPLER_ROOT && entry('RC_DOWNSAMPLER_ROOT', 'dist/webgpu/index.js');

export default {
  root,
  build: { target: 'esnext' },
  resolve: { alias: {
    '@pmndrs/raytracer': tracer,
    '@pmndrs/bvh': bvh,
    ...(downsampler && { '@pmndrs/downsampler/webgpu': downsampler }),
  } },
  server: { fs: { allow: [
    root, resolve(process.env.RC_PATHTRACER_ROOT), dirname(bvh),
    ...(downsampler ? [resolve(process.env.RC_DOWNSAMPLER_ROOT, 'dist')] : []),
  ] } },
};
