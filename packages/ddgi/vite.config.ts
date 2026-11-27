import { resolve } from 'path';
import { defineConfig, type Plugin } from 'vite';

import { layerChunk } from './layer-chunks.ts';

// Library build for @pmndrs/gi. One entry per public subpath: `/core` (the
// platform), `/ddgi` (the DDGI technique), `/radiance-cache` (the radiance
// cache), `/radiance-cascades` (Split RC), `/three` (the three.js adapter) and
// `.` (the same as `/three`). The private workspace packages (@pmndrs/ddgi-core,
// @pmndrs/ddgi-three) are bundled in.
//
// Each module lands in exactly one chunk, so every entry sees the same classes
// (an `instanceof` across subpaths must hold). The chunks are cut by layer
// (layer-chunks.ts), so `/core` loads no technique code and `/ddgi` no three code.

/** Optional packages a technique loads at run time with `import()`; none is a dependency of @pmndrs/gi. */
const OPTIONAL_RUNTIME_IMPORTS = new Set(['@pathtracer/mlp']);

/**
 * The radiance cache's neural tail imports `@pathtracer/mlp` (a private sibling,
 * not on npm) dynamically, behind `neuralCache: 'tail'`. Kept external, so no
 * private code is bundled, and its `import()` is written so a consumer's bundler
 * doesn't try to resolve it: vite, esbuild and webpack all fail a build on a
 * string-literal `import()` of a package that isn't installed. The specifier
 * goes through a function argument (with the vite and webpack ignore comments),
 * so where the package is absent the import rejects at run time and the tail
 * turns off with a warning, as it does in a checkout without the sibling.
 */
function optionalRuntimeImports(): Plugin {
    return {
        name: 'optional-runtime-imports',
        enforce: 'pre',
        resolveId(id) {
            if (OPTIONAL_RUNTIME_IMPORTS.has(id)) return { id, external: true };
        },
        renderDynamicImport({ targetModuleId }) {
            if (!targetModuleId || !OPTIONAL_RUNTIME_IMPORTS.has(targetModuleId)) return null;
            return { left: '((specifier) => import(/* @vite-ignore */ /* webpackIgnore: true */ specifier))(', right: ')' };
        },
    };
}

export default defineConfig({
    plugins: [{
        name: 'rc-traversal-bvh',
        enforce: 'pre',
        resolveId(id, importer) {
            // The pinned interval tracer uses the published BVH ABI; DDGI has its own workspace tree.
            if (id === '@pmndrs/bvh' && importer?.replaceAll('\\', '/').includes('/raytracer/')) {
                return { id: '@pmndrs/rc-bvh', external: true };
            }
        },
    }, optionalRuntimeImports()],
    build: {
        lib: {
            entry: {
                index: resolve(__dirname, 'src/index.ts'),
                core: resolve(__dirname, 'src/core.ts'),
                'radiance-cascades': resolve(__dirname, 'src/radiance-cascades.ts'),
                'radiance-cache': resolve(__dirname, 'src/radiance-cache.ts'),
                ddgi: resolve(__dirname, 'src/ddgi.ts'),
                three: resolve(__dirname, 'src/three.ts'),
            },
            formats: ['es'],
        },
        rollupOptions: {
            // three and its addons (the composite uses three's own GTAO) stay the user's copy;
            // the npm dependencies (the RC BVH, the radiance cache's denoiser) are installed with the package.
            external: ['@pmndrs/rc-bvh', /^denoiser(\/|$)/, /^three(\/|$)/],
            output: {
                manualChunks: layerChunk,
                chunkFileNames: 'chunks/[name]-[hash].js',
            },
        },
    },
});
