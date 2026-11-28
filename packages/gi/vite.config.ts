import { resolve } from 'path';
import { defineConfig, type Plugin } from 'vite';

import { layerChunk } from './layer-chunks.ts';

/** `new Worker(new URL('./x.ts', import.meta.url), { type: 'module' })`, the form the source starts its workers with. */
const MODULE_WORKER = /new\s+Worker\(\s*new\s+URL\(\s*(['"])(\.{1,2}\/[^'"]+)\1\s*,\s*import\.meta\.url\s*\)\s*,\s*\{\s*type:\s*(['"])module\3\s*\}\s*\)/g;

/**
 * Builds each worker into the chunk that starts it, as a string the browser
 * starts from a blob URL (vite's `?worker&inline`).
 *
 * Left to itself, vite emits the worker as `assets/x-<hash>.js` and points the
 * chunk at `"/assets/x-<hash>.js"`, a path from the site root. A user's
 * bundler then looks for the worker in their own `public/` folder, and
 * `vite build` of any app importing the package fails. Even a relative URL to
 * a worker file breaks under vite's dev server, which moves the package into
 * node_modules/.vite and leaves the worker behind. An inlined worker needs no
 * file, so it works in any bundler, dev server or plain browser. (The BVH
 * workers carry tinybvh's WebAssembly inline already, so they fetch nothing.)
 *
 * The source keeps the standard `new Worker(new URL(...))` form, so the
 * workspace's own apps and tests start the worker from its file as before.
 */
function inlineWorkers(): Plugin {
    return {
        name: 'gi-inline-workers',
        enforce: 'pre',
        transform(code, id) {
            if (id.includes('node_modules') || !code.includes('new Worker')) return;
            let count = 0;
            const imports: string[] = [];
            const body = code.replace(MODULE_WORKER, (_, _quote, path: string) => {
                const name = `__giInlineWorker${count++}`;
                imports.push(`import ${name} from ${JSON.stringify(`${path}?worker&inline`)};`);
                return `new ${name}()`;
            });
            if (count === 0) return;
            return { code: `${imports.join('\n')}\n${body}`, map: null };
        },
        generateBundle(_, bundle) {
            // Any worker file or worker URL left in the output is the bug this plugin is for.
            for (const [fileName, output] of Object.entries(bundle)) {
                if (output.type === 'asset' && /worker/i.test(fileName)) this.error(`the build wrote ${fileName}: a worker the package can't point at; start it as new Worker(new URL('./x.ts', import.meta.url), { type: 'module' })`);
                if (output.type === 'chunk' && /new\s+Worker\(\s*new\s+URL\(/.test(output.code)) this.error(`${fileName} starts a worker from a URL; start it as new Worker(new URL('./x.ts', import.meta.url), { type: 'module' }) so the build inlines it`);
            }
        },
    };
}

// Library build for @pmndrs/gi. One entry per public subpath: `/core` (the
// platform), `/ddgi` (the DDGI technique), `/radiance-cache` (the radiance
// cache), `/radiance-cascades` (Split RC), `/three` (the three.js adapter) and
// `.` (the same as `/three`). The private workspace packages (@pmndrs/gi-core,
// -three, -ddgi, -ddgi-three, -cache) are bundled in.
//
// Each module lands in exactly one chunk, so every entry sees the same classes
// (an `instanceof` across subpaths must hold). The chunks are cut by layer
// (layer-chunks.ts), so `/core` loads no technique code and `/ddgi` no three code.
// Workers are built into the chunk that starts them (inlineWorkers).

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
    // Module workers, as the source asks for: the inlined worker runs as an ES module.
    worker: { format: 'es' },
    plugins: [inlineWorkers(), {
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
