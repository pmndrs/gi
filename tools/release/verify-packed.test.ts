import { resolve } from 'node:path';

import { describe, expect, test } from 'vitest';

import {
    REQUIRED_FILES,
    bareImports,
    checkImportGraph,
    checkTarballContents,
    importMetaUrls,
    importSpecifiers,
    layerOf,
    missingBuiltUrls,
    parseArguments,
    parsePackJson,
    resolveRelative,
} from './verify-packed.ts';

describe('importSpecifiers', () => {
    test('static, re-exported, type-only and dynamic imports', () => {
        const source = `import { a } from './a.js';
export * from "./b.js";
import type { C } from 'three/webgpu';
export { d } from 'three';
const e = await import('./e.js');
import './side-effect.js';`;
        expect(importSpecifiers(source)).toEqual(['./a.js', './b.js', 'three/webgpu', 'three', './e.js', './side-effect.js']);
    });

    test('ignores words in strings and comments that look like imports', () => {
        const source = `const message = "measured from 'start'";
// copied from 'origin'
const pattern = /from 'x'/;
throw new Error("import('nowhere') failed");`;
        expect(importSpecifiers(source)).toEqual([]);
    });
});

describe('the import graph', () => {
    // A package cut into layer chunks as the build cuts it (packages/ddgi/layer-chunks.ts).
    const files: Record<string, string> = {
        '/pkg/dist/radiance-cascades.js': "export * from './chunks/rc-r4.js';",
        '/pkg/dist/radiance-cascades.d.ts': "export * from './chunks/rc-r4.js';",
        '/pkg/dist/chunks/rc-r4.js': "import '@pmndrs/rc-bvh'; export const rc = 1;",
        '/pkg/dist/chunks/rc-r4.d.ts': "export type { MeshGeometry } from '@pmndrs/rc-bvh';",
        '/pkg/dist/radiance-cache.js': "export * from './chunks/radiance-cache-d5.js';",
        '/pkg/dist/radiance-cache.d.ts': "export * from './chunks/radiance-cache-d5.js';",
        // The optional neural tail's package through the hidden import() the build writes (packages/ddgi/vite.config.ts).
        '/pkg/dist/chunks/radiance-cache-d5.js':
            "import { Gi1Denoiser } from 'denoiser/ffx'; import { x } from './platform-a1.js'; export const mlp = () => ((specifier) => import(/* @vite-ignore */ specifier))('@pathtracer/mlp');",
        '/pkg/dist/chunks/radiance-cache-d5.d.ts': "export { x as cache } from './platform-a1.js'; type MlpNetwork = 'default' | 'compact';",
        '/pkg/dist/core.js': "export * from './chunks/platform-a1.js';",
        '/pkg/dist/core.d.ts': "export * from './chunks/platform-a1.js';",
        '/pkg/dist/ddgi.js': "export * from './chunks/ddgi-b2.js';",
        '/pkg/dist/ddgi.d.ts': "export * from './chunks/ddgi-b2.js';",
        '/pkg/dist/three.js': "export * from './chunks/three-c3.js';",
        '/pkg/dist/three.d.ts': "export * from 'three/addons/lighting/LightProbeGrid.js'; export * from './chunks/three-c3.js';",
        '/pkg/dist/index.js': "export * from './three.js';",
        '/pkg/dist/index.d.ts': "export * from './three.js';",
        '/pkg/dist/chunks/platform-a1.js': "export const x = 1; import('./platform-a1.js');",
        '/pkg/dist/chunks/platform-a1.d.ts': 'export declare const x: number;',
        '/pkg/dist/chunks/ddgi-b2.js': "import { x } from './platform-a1.js'; export const y = x;",
        '/pkg/dist/chunks/ddgi-b2.d.ts': "export { x as y } from './platform-a1.js';",
        '/pkg/dist/chunks/three-c3.js': "import 'three/webgpu'; import './ddgi-b2.js'; export const z = 3;",
        '/pkg/dist/chunks/three-c3.d.ts': "import './ddgi-b2.js'; export declare const z: number;",
    };
    const reader = (fixture: Record<string, string>) => (path: string) => {
        const key = Object.keys(fixture).find((name) => resolve(name) === path);
        if (key === undefined) throw new Error(`no fixture file ${path}`);
        return fixture[key];
    };
    const read = reader(files);

    test('a .js import in a declaration file means the .d.ts beside it', () => {
        expect(resolveRelative(resolve('/pkg/dist/core.d.ts'), './chunks/platform-a1.js')).toBe(resolve('/pkg/dist/chunks/platform-a1.d.ts'));
        expect(resolveRelative(resolve('/pkg/dist/core.js'), './chunks/platform-a1.js')).toBe(resolve('/pkg/dist/chunks/platform-a1.js'));
    });

    test('follows relative imports to every bare one, through cycles', () => {
        expect([...bareImports(resolve('/pkg/dist/index.js'), read)]).toEqual(['three/webgpu']);
        expect([...bareImports(resolve('/pkg/dist/core.js'), read)]).toEqual([]);
        expect([...bareImports(resolve('/pkg/dist/ddgi.js'), read)]).toEqual([]);
    });

    test('a clean package passes', () => {
        expect(checkImportGraph(resolve('/pkg'), read)).toEqual([]);
    });

    test('/core reaching three fails', () => {
        const leaky = { ...files, '/pkg/dist/chunks/platform-a1.js': "import { Vector3 } from 'three';" };
        const failures = checkImportGraph(resolve('/pkg'), reader(leaky));
        expect(failures).toContain("dist/core.js reaches 'three', which /core must not need");
        expect(failures).toContain("dist/ddgi.js reaches 'three', which /ddgi must not need");
    });

    test('/radiance-cascades cannot acquire a Three dependency', () => {
        const leaky = { ...files, '/pkg/dist/chunks/rc-r4.js': "import 'three';" };
        expect(checkImportGraph(resolve('/pkg'), reader(leaky))).toContain("dist/radiance-cascades.js reaches 'three', which /radiance-cascades must not need");
    });

    test('/radiance-cache reaches its denoiser only, and loads no DDGI or three code', () => {
        expect([...bareImports(resolve('/pkg/dist/radiance-cache.js'), read)]).toEqual(['denoiser/ffx']);
        const withThree = { ...files, '/pkg/dist/chunks/radiance-cache-d5.d.ts': "import type { Node } from 'three/webgpu'; export * from './ddgi-b2.js';" };
        expect(checkImportGraph(resolve('/pkg'), reader(withThree))).toEqual([
            "dist/radiance-cache.d.ts reaches 'three/webgpu', which /radiance-cache must not need",
            'dist/radiance-cache.d.ts reaches ddgi-b2.d.ts, which holds ddgi code; /radiance-cache may load only platform, radiance-cache',
        ]);
    });

    test('/radiance-cache importing the optional neural package by name fails, in code or in types', () => {
        const literal = { ...files, '/pkg/dist/chunks/radiance-cache-d5.js': "export const mlp = () => import('@pathtracer/mlp');" };
        expect(checkImportGraph(resolve('/pkg'), reader(literal))).toEqual(["dist/radiance-cache.js reaches '@pathtracer/mlp', which /radiance-cache must not need"]);
        const typed = { ...files, '/pkg/dist/chunks/radiance-cache-d5.d.ts': "export type { MlpNetwork } from '@pathtracer/mlp';" };
        expect(checkImportGraph(resolve('/pkg'), reader(typed))).toEqual(["dist/radiance-cache.d.ts reaches '@pathtracer/mlp', which /radiance-cache must not need"]);
    });

    test('a file holds the layer its name says', () => {
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/chunks/platform-a1.js'))).toBe('platform');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/chunks/ddgi-b2.d.ts'))).toBe('ddgi');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/chunks/radiance-cache-d5.js'))).toBe('radiance-cache');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/radiance-cache.d.ts'))).toBe('radiance-cache');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/core.js'))).toBe('platform');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/index.d.ts'))).toBe('three');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/assets/worker.js'))).toBeUndefined();
    });

    test('/core loading DDGI code fails, even with no package import', () => {
        const tangled = { ...files, '/pkg/dist/chunks/platform-a1.js': "import './ddgi-b2.js'; export const x = 1;" };
        const failures = checkImportGraph(resolve('/pkg'), reader(tangled));
        expect(failures).toEqual([
            'dist/core.js reaches ddgi-b2.js, which holds ddgi code; /core may load only platform',
            // The radiance cache builds on the platform, so the leak reaches it too.
            'dist/radiance-cache.js reaches ddgi-b2.js, which holds ddgi code; /radiance-cache may load only platform, radiance-cache',
        ]);
    });

    test('/ddgi loading three code fails', () => {
        const tangled = { ...files, '/pkg/dist/chunks/ddgi-b2.d.ts': "export * from './three-c3.js';" };
        const failures = checkImportGraph(resolve('/pkg'), reader(tangled));
        expect(failures).toEqual(['dist/ddgi.d.ts reaches three-c3.d.ts, which holds three code; /ddgi may load only platform, ddgi']);
    });
});

describe('checkTarballContents', () => {
    const manifest = {
        exports: { '.': { types: './dist/index.d.ts', import: './dist/index.js' } },
        peerDependencies: { three: '~0.187.0' },
    };
    const complete = [...REQUIRED_FILES, 'dist/index.d.ts', 'dist/index.js'];

    test('a complete tarball passes', () => {
        expect(checkTarballContents(complete, manifest)).toEqual([]);
    });

    test('names each missing file', () => {
        const failures = checkTarballContents(complete.filter((path) => path !== 'THIRD_PARTY.md' && path !== 'dist/index.js'), manifest);
        expect(failures).toEqual(['the tarball lacks THIRD_PARTY.md', 'the tarball lacks dist/index.js']);
    });

    test('refuses dependencies a consumer could not install', () => {
        const failures = checkTarballContents(complete, {
            ...manifest,
            dependencies: { '@pmndrs/ddgi-core': 'workspace:*', '@pmndrs/bvh': '0.0.0' },
        });
        expect(failures).toHaveLength(2);
    });

    test('refuses the optional neural package in any dependency field: it is not on npm', () => {
        for (const field of ['dependencies', 'peerDependencies', 'optionalDependencies'] as const) {
            expect(checkTarballContents(complete, { ...manifest, [field]: { '@pathtracer/mlp': '0.1.0' } })).toEqual([`${field} names @pathtracer/mlp@0.1.0, which isn't on npm`]);
        }
    });

    test('allows workspace devDependencies, which consumers never install', () => {
        expect(checkTarballContents(complete, { ...manifest, devDependencies: { '@pmndrs/ddgi-core': 'workspace:*' } })).toEqual([]);
    });
});

describe('importMetaUrls', () => {
    test('string targets, with or without the comment vite writes', () => {
        const source = `new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const wasm = new URL("../assets/tiny.wasm", import.meta.url);
return new Worker(new URL(
  /* @vite-ignore */
  "/assets/mesh-bvh-worker-BgIMPdeh.js",
  import.meta.url
), { type: "module" });`;
        expect(importMetaUrls(source)).toEqual(['./worker.js', '../assets/tiny.wasm', '/assets/mesh-bvh-worker-BgIMPdeh.js']);
    });

    test('ignores URLs built from a variable or another base', () => {
        expect(importMetaUrls("new URL(path, import.meta.url); new URL('./x.js', location.href);")).toEqual([]);
    });

    test('skips a template made at run time, keeps a plain template', () => {
        // vite 6's client, served to every app under the dev server.
        const source = 'fetch(new URL(\n  `${base$1}__open-in-editor?file=${encodeURIComponent(file)}`,\n  import.meta.url\n)); new URL(`./plain.js`, import.meta.url);';
        expect(importMetaUrls(source)).toEqual(['./plain.js']);
    });
});

describe('missingBuiltUrls', () => {
    test('relative and root targets the build wrote pass; inline data and other origins are skipped', () => {
        const files = {
            'index.html': '',
            'assets/index-a.js': `new Worker(new URL("./worker-b.js", import.meta.url));
new URL('/assets/tiny.wasm', import.meta.url);
new Worker(new URL('data:text/javascript,0', import.meta.url));
new URL('https://example.com/x.js', import.meta.url);`,
            'assets/worker-b.js': '',
            'assets/tiny.wasm': '',
        };
        expect(missingBuiltUrls(files)).toEqual([]);
    });

    test('names a file the build points at but never wrote, as with the BVH workers at /assets/', () => {
        const files = {
            'assets/index-a.js': 'new Worker(new URL(/* @vite-ignore */ "/assets/mesh-bvh-worker-BgIMPdeh.js", import.meta.url));',
        };
        expect(missingBuiltUrls(files)).toEqual(["assets/index-a.js fetches '/assets/mesh-bvh-worker-BgIMPdeh.js', which the build didn't write"]);
    });
});

describe('parsePackJson', () => {
    const result = { filename: 'pmndrs-gi-0.1.0.tgz', size: 1, unpackedSize: 2, files: [{ path: 'package.json', size: 3 }] };

    test('npm 11 prints an array, npm 12 an object keyed by name', () => {
        expect(parsePackJson(JSON.stringify([result]))).toEqual([result]);
        expect(parsePackJson(JSON.stringify({ '@pmndrs/gi': result }))).toEqual([result]);
    });

    test('refuses anything else', () => {
        expect(() => parsePackJson('npm notice')).toThrow(/JSON/);
        expect(() => parsePackJson('{}')).toThrow(/shape/);
    });
});

test('parseArguments', () => {
    expect(parseArguments(['--skip-build', '--keep-temp'])).toEqual({ skipBuild: true, keepTemp: true, help: false });
    expect(() => parseArguments(['--publish'])).toThrow(/Unknown option/);
});
