import { resolve } from 'node:path';

import { describe, expect, test } from 'vitest';

import {
    REQUIRED_FILES,
    bareImports,
    checkImportGraph,
    checkTarballContents,
    importSpecifiers,
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
    const files: Record<string, string> = {
        '/pkg/dist/core.js': "export * from './chunks/shared.js';",
        '/pkg/dist/core.d.ts': "export * from './chunks/shared.js';",
        '/pkg/dist/three.js': "import 'three/webgpu'; export * from './chunks/shared.js';",
        '/pkg/dist/three.d.ts': "export * from 'three/addons/lighting/LightProbeGrid.js';",
        '/pkg/dist/index.js': "export * from './three.js';",
        '/pkg/dist/index.d.ts': "export * from './three.js';",
        '/pkg/dist/chunks/shared.js': "export const x = 1; import('./shared.js');",
        '/pkg/dist/chunks/shared.d.ts': 'export declare const x: number;',
    };
    const read = (path: string) => {
        const key = Object.keys(files).find((name) => resolve(name) === path);
        if (key === undefined) throw new Error(`no fixture file ${path}`);
        return files[key];
    };

    test('a .js import in a declaration file means the .d.ts beside it', () => {
        expect(resolveRelative(resolve('/pkg/dist/core.d.ts'), './chunks/shared.js')).toBe(resolve('/pkg/dist/chunks/shared.d.ts'));
        expect(resolveRelative(resolve('/pkg/dist/core.js'), './chunks/shared.js')).toBe(resolve('/pkg/dist/chunks/shared.js'));
    });

    test('follows relative imports to every bare one, through cycles', () => {
        expect([...bareImports(resolve('/pkg/dist/index.js'), read)]).toEqual(['three/webgpu']);
        expect([...bareImports(resolve('/pkg/dist/core.js'), read)]).toEqual([]);
    });

    test('a clean package passes', () => {
        expect(checkImportGraph(resolve('/pkg'), read)).toEqual([]);
    });

    test('/core reaching three fails', () => {
        const leaky: Record<string, string> = { ...files, '/pkg/dist/chunks/shared.js': "import { Vector3 } from 'three';" };
        const failures = checkImportGraph(resolve('/pkg'), (path) => {
            const key = Object.keys(leaky).find((name) => resolve(name) === path);
            if (key === undefined) throw new Error(`no fixture file ${path}`);
            return leaky[key];
        });
        expect(failures).toContain("dist/core.js reaches 'three', which /core must not need");
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
    const result = { filename: 'pmndrs-ddgi-0.1.0.tgz', size: 1, unpackedSize: 2, files: [{ path: 'package.json', size: 3 }] };

    test('npm 11 prints an array, npm 12 an object keyed by name', () => {
        expect(parsePackJson(JSON.stringify([result]))).toEqual([result]);
        expect(parsePackJson(JSON.stringify({ '@pmndrs/ddgi': result }))).toEqual([result]);
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
