import path from 'node:path';
import { fileURLToPath } from 'node:url';
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

/** What ddgi-core may never import: engines, the adapters, or the published package. */
const ENGINE_FREE_IMPORTS = {
    group: [
        'three',
        'three/*',
        '@babylonjs/*',
        'pixi.js',
        'pixi.js/*',
        '@pmndrs/gi',
        '@pmndrs/gi/*',
        '@pmndrs/ddgi-three',
        '@pmndrs/ddgi-three/*',
        '**/ddgi-three/**',
    ],
    message: 'ddgi-core must stay engine-free: do not import three, babylon, pixi or ddgi-three here. Put engine code in an adapter package.',
};

// The core's layers (SPEC "Engine boundary"; docs/design/radiance-cache.md):
// `platform/contract` is the API the platform promises to keep; everything else
// in `platform/` may change under a technique. So a technique may import
// `platform/contract` and its own folder, and nothing else from the repo's
// packages: not the platform's implementation (`platform/trace`, ...), not the
// core's entry point (it re-exports the whole platform), not `packages/bvh`
// (the tracer is reached through the contract's ray queries), and not another
// technique. The platform never imports a technique (or the entry point that
// assembles them). Tests are exempt: they may use internals as fixtures.
//
// A small local rule rather than path patterns: it resolves each relative
// import against the importing file and judges the file it lands on, so
// `../../platform/contract/../trace/x.ts`, any number of `../` and a
// technique's own folder named like another technique are all judged right.
const CORE_SRC = fileURLToPath(new URL('./packages/ddgi-core/src/', import.meta.url));
const PACKAGES_DIR = fileURLToPath(new URL('./packages/', import.meta.url));
const TECHNIQUE_IMPORT_MESSAGE = 'A technique may import only platform/contract and its own folder. Reach platform features through the contract (add to it if something is missing), not through platform internals, the core entry point, the BVH package or another technique.';
const PLATFORM_IMPORT_MESSAGE = 'The platform may not import a technique (or the core entry point that assembles them): techniques depend on the platform, never the other way round. Put shared code in the platform and export it through platform/contract.';
// Package names that lead into the repo's own packages from a bare specifier.
const OWN_PACKAGE_PREFIXES = ['@pmndrs/bvh', '@pmndrs/ddgi-core'];
// Technique imports of platform internals that `main` made before this rule
// reached it (the radiance-cache merge, 2026-11-19), allowed one by one so DDGI's
// code stays as `main` has it. Each is a helper the contract doesn't offer yet;
// the way out is to export it through platform/contract (the lead's call), then
// drop the line here. Paths are below ddgi-core's src/: importer -> imported file.
const KNOWN_TECHNIQUE_IMPORTS = new Set([
    'techniques/ddgi/ddgi-technique.ts -> platform/scene/scene-fingerprint.ts', // fingerprintDifferences (T5.2 saved bakes)
    'techniques/ddgi/saved-field.ts -> platform/sky/half-float.ts', // halfToFloat (T5.2 saved bakes)
]);

/** Path segments of `file` below ddgi-core's `src/`, or null when it is outside. */
function coreSegments(file) {
    const fromCore = path.relative(CORE_SRC, file);
    if (fromCore.startsWith('..') || path.isAbsolute(fromCore)) return null;
    return fromCore.split(path.sep);
}

/** Why importing `specifier` from `importer` breaks the layers, or null when it doesn't. */
function layerViolation(importer, specifier) {
    const from = coreSegments(importer);
    if (!from || /\.test\.ts$/.test(importer)) return null;
    const inTechniques = from[0] === 'techniques';
    const inPlatform = from[0] === 'platform';
    if (!inTechniques && !inPlatform) return null;

    if (!specifier.startsWith('.')) {
        const ownPackage = OWN_PACKAGE_PREFIXES.some((name) => specifier === name || specifier.startsWith(`${name}/`));
        return inTechniques && ownPackage ? TECHNIQUE_IMPORT_MESSAGE : null;
    }
    const target = path.resolve(path.dirname(importer), specifier);
    const to = coreSegments(target);
    if (inPlatform) {
        if (!to) return null;
        const isEntryPoint = to.length === 1 || to[0] === '' || to[0].startsWith('index');
        return to[0] === 'techniques' || isEntryPoint ? PLATFORM_IMPORT_MESSAGE : null;
    }
    // A technique file: `from` is ['techniques', <name>, ...] (or a file directly in techniques/, which owns no folder).
    if (!to) {
        // Outside the core: other packages of this repo are off limits (packages/bvh by path).
        const fromPackages = path.relative(PACKAGES_DIR, target);
        const insidePackages = !fromPackages.startsWith('..') && !path.isAbsolute(fromPackages);
        return insidePackages ? TECHNIQUE_IMPORT_MESSAGE : null;
    }
    if (KNOWN_TECHNIQUE_IMPORTS.has(`${from.join('/')} -> ${to.join('/')}`)) return null;
    const ownTechnique = from.length > 2 ? from[1] : null;
    const isContract = to[0] === 'platform' && to[1] !== undefined && to[1].replace(/\.ts$/, '') === 'contract';
    const isOwnFolder = ownTechnique !== null && to[0] === 'techniques' && to[1] === ownTechnique;
    return isContract || isOwnFolder ? null : TECHNIQUE_IMPORT_MESSAGE;
}

const coreLayersRule = {
    meta: { type: 'problem', schema: [] },
    create(context) {
        const check = (node, source) => {
            if (!source || source.type !== 'Literal' || typeof source.value !== 'string') return;
            const message = layerViolation(context.filename, source.value);
            if (message) context.report({ node: source, message });
        };
        return {
            ImportDeclaration: (node) => check(node, node.source),
            ExportNamedDeclaration: (node) => check(node, node.source),
            ExportAllDeclaration: (node) => check(node, node.source),
            ImportExpression: (node) => check(node, node.source),
        };
    },
};
const localPlugin = { rules: { 'core-layers': coreLayersRule } };

export default tseslint.config(
    // tools/docs/snippets/ holds the docs' examples, generated and only typechecked (tools/docs/doc-snippets.ts).
    { ignores: ['**/dist/', 'node_modules/', '.claude/', 'native/', 'research/', 'assets/downloads/', 'tools/emsdk/', 'tools/docs/snippets/'] },
    js.configs.recommended,
    ...tseslint.configs.recommended,
    {
        languageOptions: {
            globals: {
                ...globals.node,
                ...globals.browser,
            },
        },
        rules: {
            '@typescript-eslint/no-unused-vars': [
                'warn',
                { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
            ],
            '@typescript-eslint/no-explicit-any': 'error',
            '@typescript-eslint/consistent-type-imports': [
                'warn',
                { prefer: 'type-imports', fixStyle: 'inline-type-imports' },
            ],
        },
    },
    {
        // The core runs on a bare GPUDevice; engines live in their adapters.
        files: ['packages/ddgi-core/**/*.ts'],
        rules: {
            'no-restricted-imports': ['error', { patterns: [ENGINE_FREE_IMPORTS] }],
            // no-restricted-imports only sees static imports, so catch import('three') too.
            'no-restricted-syntax': ['error', {
                selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|$))|ddgi-three/]",
                message: 'ddgi-core must stay engine-free: do not import three, babylon, pixi or ddgi-three here. Put engine code in an adapter package.',
            }],
        },
    },
    {
        // `@pmndrs/gi/core` is the platform alone (DECISIONS 2026-11-10), so platform code
        // may not import a technique; techniques build on the platform, not the other way.
        // Tests may, to drive the platform with DDGI. The engine patterns are repeated:
        // a later block's options for a rule replace an earlier block's.
        files: ['packages/ddgi-core/src/platform/**/*.ts'],
        ignores: ['**/*.test.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [
                    ENGINE_FREE_IMPORTS,
                    {
                        group: ['**/techniques/**'],
                        message: 'The platform may not import a technique: @pmndrs/gi/core must hold no technique code. Move shared code into the platform.',
                    },
                ],
            }],
        },
    },
    {
        // Techniques import only platform/contract and their own folder; the
        // platform never imports a technique (see coreLayersRule above).
        files: ['packages/ddgi-core/src/**/*.ts'],
        plugins: { local: localPlugin },
        rules: {
            'local/core-layers': 'error',
        },
    },
    {
        // The BVH package is engine-free and below the core: it may not import the
        // core, the GI code, the adapters or any engine (docs/design/bvh-framework.md).
        files: ['packages/bvh/**/*.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [{
                    group: [
                        'three',
                        'three/*',
                        'three-mesh-bvh',
                        'three-mesh-bvh/*',
                        '@babylonjs/*',
                        'pixi.js',
                        'pixi.js/*',
                        '@pmndrs/gi',
                        '@pmndrs/gi/*',
                        '@pmndrs/ddgi-core',
                        '@pmndrs/ddgi-core/*',
                        '@pmndrs/ddgi-three',
                        '@pmndrs/ddgi-three/*',
                        '**/ddgi-core/**',
                        '**/ddgi-three/**',
                        '**/ddgi/src/**',
                    ],
                    message: 'packages/bvh must stay engine-free and independent of the GI code: no ddgi-core, ddgi-three, three or other engines here.',
                }],
            }],
            'no-restricted-syntax': ['error', {
                selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|$))|ddgi-(core|three)/]",
                message: 'packages/bvh must stay engine-free and independent of the GI code: no ddgi-core, ddgi-three, three or other engines here.',
            }],
        },
    },
    {
        // The three.js adapter depends on the core and three only.
        files: ['packages/ddgi-three/**/*.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [{
                    group: ['@babylonjs/*', 'pixi.js', 'pixi.js/*', '@pmndrs/gi', '@pmndrs/gi/*'],
                    message: 'ddgi-three may import only three and @pmndrs/ddgi-core.',
                }],
            }],
        },
    },
);
