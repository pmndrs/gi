import path from 'node:path';
import { fileURLToPath } from 'node:url';
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

/** What the engine-free packages (core and the techniques) may never import: engines, the three.js packages, or the published package. */
const ENGINE_FREE_IMPORTS = {
    group: [
        'three',
        'three/*',
        '@babylonjs/*',
        'pixi.js',
        'pixi.js/*',
        '@pmndrs/gi',
        '@pmndrs/gi/*',
        '@pmndrs/gi-three',
        '@pmndrs/gi-three/*',
        '@pmndrs/gi-ddgi-three',
        '@pmndrs/gi-ddgi-three/*',
    ],
    message: 'The platform and the techniques must stay engine-free: do not import three, babylon, pixi or the three.js packages here. Put engine code in packages/three, or in a technique\'s own <technique>-three package.',
};
const ENGINE_FREE_PACKAGES = ['packages/core/**/*.ts', 'packages/ddgi/**/*.ts', 'packages/cache/**/*.ts'];

// The workspace packages (DECISIONS 2026-11-27, "flat and technique-chained"):
// `core` (the platform), `three` (the three.js layer every technique shares),
// a package per technique (`ddgi`, `cache`; Split RC still inside `core` until
// it moves to `radiance-cascades`) and a technique's own three.js part
// (`ddgi-three`), with `gi`, the published package, re-exporting them. Three
// local rules keep that shape; each resolves a relative import against the
// importing file and judges the file it lands on, so any number of `../` is
// judged right. They see `import`, `export ... from` and `import()`, tests
// included unless the rule says otherwise.
//
// - `local/package-boundaries`: no package reaches into another's files. A
//   relative import that leaves the importing file's package, or a package
//   name followed by `/src/`, is refused: import another package by its name,
//   through its package.json `exports`.
// - `local/three-imports-no-technique`: `three` is shared by every technique,
//   so it imports no technique package (nor Split RC inside core, nor the
//   published package). Three.js code only one technique uses goes in that
//   technique's `<technique>-three` package; code a second technique would
//   import belongs in `three`.
// - `local/technique-layers` (technique packages) and `local/core-layers`
//   (Split RC and the platform inside core): see below.
const PACKAGES_DIR = fileURLToPath(new URL('./packages/', import.meta.url));
const CORE_SRC = fileURLToPath(new URL('./packages/core/src/', import.meta.url));
/** Technique packages under packages/: each builds on the platform contract alone. Split RC joins when it moves out of core. */
const TECHNIQUE_PACKAGES = new Set(['ddgi', 'cache']);
/** A bare specifier naming one of this repository's packages: `@pmndrs/gi`, `@pmndrs/gi-*`, `@pmndrs/bvh`. */
const OWN_PACKAGE = /^@pmndrs\/(gi|gi-[\w-]+|bvh)(\/|$)/;

const BOUNDARY_MESSAGE = (to) => `This reaches into packages/${to}'s files. Import another package by its name (@pmndrs/gi-…, @pmndrs/bvh), through its package.json exports.`;
const SRC_SPECIFIER_MESSAGE = 'A package name followed by /src/ reaches into that package\'s files. Import it through its package.json exports (`@pmndrs/gi-core/platform/contract/math`, not `…/src/…`).';
const THREE_MESSAGE = 'packages/three is the three.js layer every technique shares, so it may not import a technique package (or Split RC inside core, or the published package). Put three.js code only one technique uses in that technique\'s <technique>-three package (DDGI\'s is packages/ddgi-three); describe what three needs from a technique by its shape (ThreeGI\'s volume).';
const TECHNIQUE_PACKAGE_MESSAGE = 'A technique package may import only the platform contract from the workspace (@pmndrs/gi-core/platform/contract/*) and its own files. Reach platform features through the contract (add to it if something is missing), not through platform internals, the core entry point, the BVH package or another technique. Tests may use internals as fixtures.';
const TECHNIQUE_IMPORT_MESSAGE = 'A technique may import only platform/contract and its own folder. Reach platform features through the contract (add to it if something is missing), not through platform internals, the core entry point, the BVH package or another technique.';
const PLATFORM_IMPORT_MESSAGE = 'The platform may not import a technique (or the core entry point that assembles them): techniques depend on the platform, never the other way round. Put shared code in the platform and export it through platform/contract.';

/** The folder under packages/ that `file` is in, or null when it is outside packages/. */
function packageOf(file) {
    const fromPackages = path.relative(PACKAGES_DIR, file);
    if (fromPackages === '' || fromPackages.startsWith('..') || path.isAbsolute(fromPackages)) return null;
    return fromPackages.split(path.sep)[0];
}

const isTest = (file) => /\.test\.ts$/.test(file);

/** Rule 2: no package reaches into another's files. */
function boundaryViolation(importer, specifier) {
    const from = packageOf(importer);
    if (!from) return null;
    if (specifier.startsWith('.')) {
        const to = packageOf(path.resolve(path.dirname(importer), specifier));
        return to && to !== from ? BOUNDARY_MESSAGE(to) : null;
    }
    return OWN_PACKAGE.test(specifier) && /^@pmndrs\/[\w-]+\/src(\/|$)/.test(specifier) ? SRC_SPECIFIER_MESSAGE : null;
}

/** Rule 1: the shared three.js layer imports no technique package. */
function threeViolation(importer, specifier) {
    if (packageOf(importer) !== 'three' || !OWN_PACKAGE.test(specifier)) return null;
    const own = (name) => specifier === name || specifier.startsWith(`${name}/`);
    if (own('@pmndrs/gi-three')) return null;
    if (own('@pmndrs/gi-core') && !specifier.startsWith('@pmndrs/gi-core/techniques/')) return null;
    // packages/bvh is below the platform and holds no technique.
    if (own('@pmndrs/bvh')) return null;
    return THREE_MESSAGE;
}

/** A technique package's own code reaches the workspace only through the platform contract. */
function techniquePackageViolation(importer, specifier) {
    if (!TECHNIQUE_PACKAGES.has(packageOf(importer)) || isTest(importer) || !OWN_PACKAGE.test(specifier)) return null;
    return specifier.startsWith('@pmndrs/gi-core/platform/contract/') ? null : TECHNIQUE_PACKAGE_MESSAGE;
}

// Inside core, the platform and Split RC (until it moves to its own package)
// keep the layers they had (SPEC "Engine boundary"; docs/design/radiance-cache.md):
// `platform/contract` is the API the platform promises to keep; everything else
// in `platform/` may change under a technique. So a technique folder may import
// `platform/contract` and its own folder, and nothing else from the repo's
// packages: not the platform's implementation (`platform/trace`, ...), not the
// core's entry point (it re-exports the whole platform), not `packages/bvh`
// (the tracer is reached through the contract's ray queries), and not another
// technique. The platform never imports a technique (or the entry point that
// assembles them). Tests are exempt: they may use internals as fixtures.

/** Path segments of `file` below gi-core's `src/`, or null when it is outside. */
function coreSegments(file) {
    const fromCore = path.relative(CORE_SRC, file);
    if (fromCore.startsWith('..') || path.isAbsolute(fromCore)) return null;
    return fromCore.split(path.sep);
}

/** Why importing `specifier` from `importer` breaks the core's layers, or null when it doesn't. */
function layerViolation(importer, specifier) {
    const from = coreSegments(importer);
    if (!from || isTest(importer)) return null;
    const inTechniques = from[0] === 'techniques';
    const inPlatform = from[0] === 'platform';
    if (!inTechniques && !inPlatform) return null;

    if (!specifier.startsWith('.')) {
        return inTechniques && OWN_PACKAGE.test(specifier) ? TECHNIQUE_IMPORT_MESSAGE : null;
    }
    const target = path.resolve(path.dirname(importer), specifier);
    const to = coreSegments(target);
    if (inPlatform) {
        if (!to) return null;
        const isEntryPoint = to.length === 1 || to[0] === '' || to[0].startsWith('index');
        return to[0] === 'techniques' || isEntryPoint ? PLATFORM_IMPORT_MESSAGE : null;
    }
    // A technique file: `from` is ['techniques', <name>, ...] (or a file directly in techniques/, which owns no folder).
    if (!to) return packageOf(target) ? TECHNIQUE_IMPORT_MESSAGE : null;
    const ownTechnique = from.length > 2 ? from[1] : null;
    const isContract = to[0] === 'platform' && to[1] !== undefined && to[1].replace(/\.ts$/, '') === 'contract';
    const isOwnFolder = ownTechnique !== null && to[0] === 'techniques' && to[1] === ownTechnique;
    return isContract || isOwnFolder ? null : TECHNIQUE_IMPORT_MESSAGE;
}

/** A rule that reports every import, re-export and `import()` whose specifier `violation` refuses. */
function importRule(violation) {
    return {
        meta: { type: 'problem', schema: [] },
        create(context) {
            const check = (source) => {
                if (!source || source.type !== 'Literal' || typeof source.value !== 'string') return;
                const message = violation(context.filename, source.value);
                if (message) context.report({ node: source, message });
            };
            return {
                ImportDeclaration: (node) => check(node.source),
                ExportNamedDeclaration: (node) => check(node.source),
                ExportAllDeclaration: (node) => check(node.source),
                ImportExpression: (node) => check(node.source),
            };
        },
    };
}

const localPlugin = {
    rules: {
        'package-boundaries': importRule(boundaryViolation),
        'three-imports-no-technique': importRule(threeViolation),
        'technique-layers': importRule(techniquePackageViolation),
        'core-layers': importRule(layerViolation),
    },
};

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
        // The platform and the techniques run on a bare GPUDevice; engines live in the three.js packages.
        files: ENGINE_FREE_PACKAGES,
        rules: {
            'no-restricted-imports': ['error', { patterns: [ENGINE_FREE_IMPORTS] }],
            // no-restricted-imports only sees static imports, so catch import('three') too.
            'no-restricted-syntax': ['error', {
                selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|$)|@pmndrs\\/gi-(ddgi-)?three(\\/|$))/]",
                message: ENGINE_FREE_IMPORTS.message,
            }],
        },
    },
    {
        // `@pmndrs/gi/core` is the platform alone (DECISIONS 2026-11-10), so platform code
        // may not import a technique; techniques build on the platform, not the other way.
        // The engine patterns are repeated: a later block's options for a rule replace an
        // earlier block's.
        files: ['packages/core/src/platform/**/*.ts'],
        ignores: ['**/*.test.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [
                    ENGINE_FREE_IMPORTS,
                    {
                        group: ['**/techniques/**', '@pmndrs/gi-ddgi', '@pmndrs/gi-ddgi/*', '@pmndrs/gi-cache', '@pmndrs/gi-cache/*'],
                        message: 'The platform may not import a technique: @pmndrs/gi/core must hold no technique code. Move shared code into the platform.',
                    },
                ],
            }],
        },
    },
    {
        // The package shape (see the local rules above): every package keeps to its own
        // files and the others' exports; three imports no technique; a technique package
        // reaches the workspace through the platform contract; inside core, Split RC and
        // the platform keep their layers.
        files: ['packages/**/*.ts', 'packages/**/*.mjs'],
        plugins: { local: localPlugin },
        rules: {
            'local/package-boundaries': 'error',
            'local/three-imports-no-technique': 'error',
            'local/technique-layers': 'error',
            'local/core-layers': 'error',
        },
    },
    {
        // The BVH package is engine-free and below the platform: it may not import the
        // platform, the GI code, the three.js packages or any engine (docs/design/bvh-framework.md).
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
                        '@pmndrs/gi-*',
                    ],
                    message: 'packages/bvh must stay engine-free and independent of the GI code: no gi-core, the technique or three.js packages, three or other engines here.',
                }],
            }],
            'no-restricted-syntax': ['error', {
                selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/gi(\\/|-|$))/]",
                message: 'packages/bvh must stay engine-free and independent of the GI code: no gi-core, the technique or three.js packages, three or other engines here.',
            }],
        },
    },
    {
        // The three.js packages depend on three and the workspace packages only.
        files: ['packages/three/**/*.ts', 'packages/ddgi-three/**/*.ts'],
        rules: {
            'no-restricted-imports': ['error', {
                patterns: [{
                    group: ['@babylonjs/*', 'pixi.js', 'pixi.js/*', '@pmndrs/gi', '@pmndrs/gi/*'],
                    message: 'The three.js packages may import only three and the gi workspace packages.',
                }],
            }],
        },
    },
);
