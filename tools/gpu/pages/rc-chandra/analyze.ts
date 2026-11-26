/**
 * Locate where the native WebGPU diffuse image differs from Chandra (issue #47).
 *
 * Run the comparison page with VITE_RC_CHANDRA_DUMP=1 first, then:
 *   node tools/gpu/pages/rc-chandra/analyze.ts <run.json> [chandra-dir] [out-dir]
 *
 * Prints per-channel brightness ratios, split by surface orientation, distance
 * to each emitter, position inside the native probe cell and screen tile, and
 * writes ratio maps as PNG. The red and cyan emitters light the scene through
 * separate channels (red only in R, cyan only in G and B), so each channel
 * isolates one emitter's light.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { encodePng } from '../../../compare/png.ts';

const [resultPath, chandraArg, outArg] = process.argv.slice(2);
if (!resultPath) throw new Error('Usage: analyze.ts <run.json> [chandra-dir] [out-dir]');
const chandraDir = resolve(chandraArg ?? 'bench/references/chandra-sponza');
const outDir = resolve(outArg ?? join(resultPath, '..', 'analysis'));
mkdirSync(outDir, { recursive: true });

const width = 640, height = 360, pixels = width * height;
const run = JSON.parse(readFileSync(resultPath, 'utf8'));
const page = run.pages.find((p: { name: string }) => p.name === 'rc-chandra');
if (!page?.data?.dump) throw new Error('Run the page with VITE_RC_CHANDRA_DUMP=1');
const floats = (base64: string) => {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
const fileFloats = (name: string) => {
    const bytes = readFileSync(join(chandraDir, name));
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
const ours = floats(page.data.dump.nativeDiffuse);
const position = floats(page.data.dump.position);
const normal = floats(page.data.dump.normal);
const chandra = fileFloats('diffuse.rgb32f');
const primary = new Uint8Array(readFileSync(join(chandraDir, 'primary.u8')));
for (const image of [ours, position, normal, chandra]) if (image.length !== pixels * 3) throw new Error('image size mismatch');

// The native gallery volume and its fine grid (adapter defaults unless the run overrode them).
const boundsMin = [-5, -0.4, -11.5], boundsMax = [5, 5.7, 11.5];
const grid: number[] = page.data.nativeOptions?.grid ?? [20, 12, 46];
const cellSize = boundsMin.map((v, i) => (boundsMax[i] - v) / grid[i]);
const firstInterval: number = page.data.nativeOptions?.firstInterval ?? 0.6;

// Emitter centroids in world space, from the primary mask and the dumped positions.
const emitterCentroid = (id: number) => {
    const sum = [0, 0, 0];
    let n = 0;
    for (let i = 0; i < pixels; i++) {
        if (primary[i] !== id) continue;
        for (let c = 0; c < 3; c++) sum[c] += position[3 * i + c];
        n++;
    }
    return sum.map(v => v / Math.max(n, 1));
};
const emitters = { red: emitterCentroid(2), cyan: emitterCentroid(3) };

/** Sums of ours and Chandra per channel; the ratio of sums is robust to dark pixels. */
class Tally {
    ours = [0, 0, 0];
    chandra = [0, 0, 0];
    count = 0;
    add(i: number) {
        for (let c = 0; c < 3; c++) { this.ours[c] += ours[3 * i + c]; this.chandra[c] += chandra[3 * i + c]; }
        this.count++;
    }
    summary() {
        const ratio = (c: number) => this.chandra[c] > 0 ? +(this.ours[c] / this.chandra[c]).toFixed(4) : null;
        return { pixels: this.count, red: ratio(0), cyan: ratio(1),
            chandraMean: this.chandra.map(v => +(v / Math.max(this.count, 1)).toFixed(5)),
            oursMean: this.ours.map(v => +(v / Math.max(this.count, 1)).toFixed(5)) };
    }
}
const tallies = new Map<string, Tally>();
const tally = (key: string, i: number) => {
    let t = tallies.get(key);
    if (!t) tallies.set(key, t = new Tally());
    t.add(i);
};
const distanceBin = (d: number) => {
    // Bins follow the native interval endpoints r, 4r and their halves.
    const edges = [0, 0.5, 1, 2, 4, 6, 8].map(k => k * firstInterval);
    let k = edges.length - 1;
    while (k > 0 && d < edges[k]) k--;
    return k === edges.length - 1 ? '>=' + edges[k].toFixed(2) : edges[k].toFixed(2) + '-' + edges[k + 1].toFixed(2);
};
const axisNames = ['+x', '-x', '+y', '-y', '+z', '-z'];
const tilesX = 8, tilesY = 6;
const classNames = ['miss', 'surface', 'red-emitter', 'cyan-emitter'];
const fifths = (f: number) => (Math.min(4, Math.floor(f * 5)) / 5).toFixed(1);
for (let i = 0; i < pixels; i++) {
    if (primary[i] === 0) continue;
    tally('class ' + classNames[primary[i]], i);
    if (primary[i] !== 1) continue;
    const n = [normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]];
    const abs = n.map(Math.abs);
    const axis = abs[0] >= abs[1] && abs[0] >= abs[2] ? 0 : abs[1] >= abs[2] ? 1 : 2;
    tally('normal ' + axisNames[2 * axis + (n[axis] < 0 ? 1 : 0)], i);
    tally('normal ' + (Math.max(...abs) > 0.99 ? 'axis-aligned' : 'tilted'), i);
    const p = [position[3 * i], position[3 * i + 1], position[3 * i + 2]];
    for (const [name, centre] of Object.entries(emitters)) {
        tally(name + ' distance ' + distanceBin(Math.hypot(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2])), i);
    }
    const cellFraction = p.map((v, c) => {
        const f = (v - boundsMin[c]) / cellSize[c];
        return f - Math.floor(f);
    });
    // Along the dominant normal axis: where the surface sits inside its cell, measured
    // toward the side it faces (0 = the cell lies entirely in front, 1 = entirely behind).
    const along = n[axis] >= 0 ? cellFraction[axis] : 1 - cellFraction[axis];
    tally('cell depth behind surface ' + fifths(along), i);
    // Across the surface: distance to the nearest cell face on the two tangent axes (0 face, 0.5 centre).
    const across = Math.min(...cellFraction.filter((_, c) => c !== axis).map(f => Math.min(f, 1 - f)));
    tally('cell tangent edge ' + fifths(2 * across), i);
    const x = i % width, y = Math.floor(i / width);
    tally('tile ' + Math.floor(x * tilesX / width) + ',' + Math.floor(y * tilesY / height), i);
}
// Each pixel class's share of the image mean, in Rec. 709 luminance (red=R, cyan=G=B here).
const luminance = (rgb: number[]) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
const shareOfMean = Object.fromEntries(['surface', 'red-emitter', 'cyan-emitter'].map(name => {
    const t = tallies.get('class ' + name)!;
    return [name, { ours: +(luminance(t.ours) / pixels).toFixed(6), chandra: +(luminance(t.chandra) / pixels).toFixed(6) }];
}));
/**
 * Relative RMS difference after averaging square blocks of pixels. Noise shrinks
 * as blocks grow; a difference that survives large blocks is low-frequency bias.
 */
function blockRelativeRms(block: number, surfacesOnly: boolean) {
    let squared = 0, signal = 0;
    for (let by = 0; by < height / block; by++) for (let bx = 0; bx < width / block; bx++) {
        const a = [0, 0, 0], b = [0, 0, 0];
        let n = 0;
        for (let y = by * block; y < (by + 1) * block; y++) for (let x = bx * block; x < (bx + 1) * block; x++) {
            const i = y * width + x;
            if (surfacesOnly ? primary[i] !== 1 : primary[i] === 0) continue;
            for (let c = 0; c < 3; c++) { a[c] += ours[3 * i + c]; b[c] += chandra[3 * i + c]; }
            n++;
        }
        for (let c = 0; c < 3; c++) { squared += (a[c] - b[c]) ** 2 / n || 0; signal += b[c] ** 2 / n || 0; }
    }
    return +Math.sqrt(squared / signal).toFixed(4);
}
const relativeRms = Object.fromEntries([1, 8, 40].map(block => ['block' + block,
    { all: blockRelativeRms(block, false), surfaces: blockRelativeRms(block, true) }]));
const report: Record<string, unknown> = {
    run: resultPath, nativeOptions: page.data.nativeOptions, cellSize, emitters, shareOfMean, relativeRms,
    groups: Object.fromEntries([...tallies].sort(([a], [b]) => a.localeCompare(b)).map(([k, t]) => [k, t.summary()])),
};

// Ratio maps: blue where ours is dimmer, red where brighter, white at equal. Grey marks pixels too dark to judge.
function ratioMap(channel: number, file: string) {
    const rgb = new Uint8Array(pixels * 3);
    for (let i = 0; i < pixels; i++) {
        const a = ours[3 * i + channel], b = chandra[3 * i + channel];
        if (primary[i] !== 1 || b < 0.004) { rgb.fill(primary[i] >= 2 ? 0 : 60, 3 * i, 3 * i + 3); continue; }
        const t = Math.max(-1, Math.min(1, Math.log2(a / b))); // ±1 = half / double
        const cool = t < 0 ? -t : 0, warm = t > 0 ? t : 0;
        rgb[3 * i] = 255 * (1 - cool);
        rgb[3 * i + 1] = 255 * (1 - cool - warm);
        rgb[3 * i + 2] = 255 * (1 - warm);
    }
    writeFileSync(join(outDir, file), encodePng(width, height, rgb));
}
ratioMap(0, 'ratio-red.png');
ratioMap(1, 'ratio-cyan.png');
// Fine-cell faces drawn over a grey version of Chandra, to compare edges in the ratio maps with the probe grid.
{
    const rgb = new Uint8Array(pixels * 3);
    for (let i = 0; i < pixels; i++) {
        const l = chandra[3 * i] + chandra[3 * i + 1];
        let v = 255 * Math.min(1, Math.sqrt(l / (1 + l)) * 2);
        const p = [position[3 * i], position[3 * i + 1], position[3 * i + 2]];
        const onEdge = primary[i] === 1 && p.some((value, c) => {
            const f = (value - boundsMin[c]) / cellSize[c];
            return Math.abs(f - Math.round(f)) < 0.04;
        });
        if (onEdge) v = 0;
        rgb.fill(v, 3 * i, 3 * i + 3);
        if (onEdge) rgb[3 * i + 1] = 200;
    }
    writeFileSync(join(outDir, 'cells.png'), encodePng(width, height, rgb));
}
writeFileSync(join(outDir, 'report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
