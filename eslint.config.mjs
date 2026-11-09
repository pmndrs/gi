import { readdirSync } from 'node:fs';
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

// The core's engine boundary (SPEC "Engine boundary"): ddgi-core runs on a bare
// GPUDevice, so no engine and no adapter may be imported anywhere in it. Kept
// as constants because the platform and technique blocks below set the same
// rules again, and a later flat-config block replaces a rule's options rather
// than adding to them.
const ENGINE_IMPORT_PATTERN = {
    group: [
        'three',
        'three/*',
        '@babylonjs/*',
        'pixi.js',
        'pixi.js/*',
        '@pmndrs/ddgi',
        '@pmndrs/ddgi/*',
        '@pmndrs/ddgi-three',
        '@pmndrs/ddgi-three/*',
        '**/ddgi-three/**',
    ],
    message: 'ddgi-core must stay engine-free: do not import three, babylon, pixi or ddgi-three here. Put engine code in an adapter package.',
};
// no-restricted-imports only sees static imports, so catch import('three') too.
const ENGINE_DYNAMIC_IMPORT = {
    selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/ddgi)|ddgi-three/]",
    message: ENGINE_IMPORT_PATTERN.message,
};

// Techniques reach the platform only through its contract (SPEC "Engine
// boundary"; docs/design/radiance-cache.md): `platform/contract` is the API the
// platform promises to keep, everything else in `platform/` may change under a
// technique. So a technique may import `platform/contract` and its own folder,
// and nothing else from the core: not the platform's implementation
// (`platform/trace`, `platform/shade`, ...), not `@pmndrs/bvh` (the tracer is
// reached through the contract's ray queries), and not another technique.
// Tests are exempt: they may use platform internals as fixtures.
const TECHNIQUE_IMPORT_MESSAGE = 'A technique may import only platform/contract and its own folder. Reach platform features through the contract (add to it if something is missing), not through platform internals, @pmndrs/bvh or another technique.';
const TECHNIQUES_DIR = 'packages/ddgi-core/src/techniques';
const TECHNIQUE_NAMES = readdirSync(new URL(`./${TECHNIQUES_DIR}`, import.meta.url), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);

/** The import patterns forbidden in technique `name`'s production files. */
function techniqueImportPatterns(name) {
    const otherTechniques = TECHNIQUE_NAMES.filter((other) => other !== name);
    return [
        ENGINE_IMPORT_PATTERN,
        {
            // Any platform path but the contract. A regex, because gitignore-style
            // groups can't re-allow files inside a folder they exclude.
            regex: '(^|/)platform(/(?!contract(/|$)).*)?$',
            message: TECHNIQUE_IMPORT_MESSAGE,
        },
        {
            group: [
                '@pmndrs/bvh',
                '@pmndrs/bvh/*',
                '@pmndrs/ddgi-core',
                '@pmndrs/ddgi-core/*',
                ...otherTechniques.flatMap((other) => [`**/techniques/${other}`, `**/techniques/${other}/**`, `../${other}`, `../${other}/**`, `../../${other}`, `../../${other}/**`]),
            ],
            message: TECHNIQUE_IMPORT_MESSAGE,
        },
    ];
}

export default tseslint.config(
    { ignores: ['**/dist/', 'node_modules/', '.claude/', 'native/', 'research/', 'assets/downloads/', 'tools/emsdk/'] },
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
            'no-restricted-imports': ['error', { patterns: [ENGINE_IMPORT_PATTERN] }],
            'no-restricted-syntax': ['error', ENGINE_DYNAMIC_IMPORT],
        },
    },
    {
        // The platform never depends on a technique: techniques plug into it.
        // Only the core's entry point (src/index.ts, outside platform/) puts
        // them together. Tests are exempt (they borrow a technique's constants).
        files: ['packages/ddgi-core/src/platform/**/*.ts'],
        ignores: ['**/*.test.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [
                    ENGINE_IMPORT_PATTERN,
                    {
                        group: ['**/techniques', '**/techniques/**'],
                        message: 'The platform may not import a technique: techniques depend on the platform, never the other way round. Put shared code in the platform and export it through platform/contract.',
                    },
                ],
            }],
            'no-restricted-syntax': ['error', ENGINE_DYNAMIC_IMPORT, {
                selector: 'ImportExpression[source.value=/techniques/]',
                message: 'The platform may not import a technique: techniques depend on the platform, never the other way round.',
            }],
        },
    },
    // One block per technique folder, so each may import its own folder but not another technique's.
    ...TECHNIQUE_NAMES.map((name) => ({
        files: [`${TECHNIQUES_DIR}/${name}/**/*.ts`],
        ignores: ['**/*.test.ts'],
        rules: {
            'no-restricted-imports': ['error', { patterns: techniqueImportPatterns(name) }],
            'no-restricted-syntax': ['error', ENGINE_DYNAMIC_IMPORT, {
                // import('../../platform/trace/...') and the like: any platform path but the contract.
                selector: 'ImportExpression[source.value=/(^|\\/)platform(\\/(?!contract(\\/|$))|$)|^@pmndrs\\/bvh|^@pmndrs\\/ddgi-core|techniques\\//]',
                message: TECHNIQUE_IMPORT_MESSAGE,
            }],
        },
    })),
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
                        '@pmndrs/ddgi',
                        '@pmndrs/ddgi/*',
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
                selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/ddgi)|ddgi-(core|three)/]",
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
                    group: ['@babylonjs/*', 'pixi.js', 'pixi.js/*', '@pmndrs/ddgi', '@pmndrs/ddgi/*'],
                    message: 'ddgi-three may import only three and @pmndrs/ddgi-core.',
                }],
            }],
        },
    },
);
