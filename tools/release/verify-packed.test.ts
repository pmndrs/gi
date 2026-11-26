import { resolve } from 'node:path';

import { describe, expect, test } from 'vitest';

import {
    REQUIRED_FILES,
    bareImports,
    checkImportGraph,
    checkTarballContents,
    importSpecifiers,
    layerOf,
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

    test('a file holds the layer its name says', () => {
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/chunks/platform-a1.js'))).toBe('platform');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/chunks/ddgi-b2.d.ts'))).toBe('ddgi');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/core.js'))).toBe('platform');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/index.d.ts'))).toBe('three');
        expect(layerOf(resolve('/pkg'), resolve('/pkg/dist/assets/worker.js'))).toBeUndefined();
    });

    test('/core loading DDGI code fails, even with no package import', () => {
        const tangled = { ...files, '/pkg/dist/chunks/platform-a1.js': "import './ddgi-b2.js'; export const x = 1;" };
        const failures = checkImportGraph(resolve('/pkg'), reader(tangled));
        expect(failures).toEqual(['dist/core.js reaches ddgi-b2.js, which holds ddgi code; /core may load only platform']);
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

    test('allows workspace devDependencies, which consumers never install', () => {
        expect(checkTarballContents(complete, { ...manifest, devDependencies: { '@pmndrs/ddgi-core': 'workspace:*' } })).toEqual([]);
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
