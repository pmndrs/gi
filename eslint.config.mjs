import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

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
            'no-restricted-imports': ['error', {
                patterns: [{
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
                }],
            }],
            // no-restricted-imports only sees static imports, so catch import('three') too.
            'no-restricted-syntax': ['error', {
                selector: "ImportExpression[source.value=/^(three|@babylonjs\\/|pixi\\.js|@pmndrs\\/ddgi)|ddgi-three/]",
                message: 'ddgi-core must stay engine-free: do not import three, babylon, pixi or ddgi-three here. Put engine code in an adapter package.',
            }],
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
