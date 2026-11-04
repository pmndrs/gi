/**
 * Writes a viewable PNG of an EXR: exposure, then sRGB, clamped (no tone mapping).
 *
 *   node tools/compare/to-png.ts image.exr [out.png] [exposure stops]
 */

import { readFile, writeFile } from 'node:fs/promises';
import { parse } from 'node:path';
import { readExr } from './exr.ts';
import { encodePng, toDisplayRgb8 } from './png.ts';

const [inputPath, outputArgument, exposureArgument] = process.argv.slice(2);
if (!inputPath) {
    console.log('usage: node tools/compare/to-png.ts image.exr [out.png] [exposure stops]');
    process.exit(1);
}
const image = readExr(new Uint8Array(await readFile(inputPath)));
const outputPath = outputArgument ?? `${parse(inputPath).dir}/${parse(inputPath).name}.png`;
await writeFile(outputPath, encodePng(image.width, image.height, toDisplayRgb8(image.rgb, Number(exposureArgument ?? 0))));
console.log(`wrote ${outputPath}`);
