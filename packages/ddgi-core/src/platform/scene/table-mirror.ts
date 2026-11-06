/**
 * The CPU copy of the table buffer: header, records and material lists.
 *
 * Every record write goes here first and marks its vec4s dirty; `encode`
 * uploads the dirty ranges. The tables are small (a 10,000-instance scene is
 * about 1.3 MB), so keeping a full copy on the CPU is cheap, and it means a
 * reallocated GPU buffer is refilled with one write instead of a GPU copy.
 *
 * Sections have fixed capacities. When one fills up the whole layout is
 * rebuilt with that section doubled; record indices and list offsets are
 * relative to their section, so nothing inside a record has to change.
 */

import { HEADER_VEC4S, INSTANCE_RECORD_VEC4S, LIGHT_RECORD_VEC4S, MATERIAL_RECORD_VEC4S, MESH_RECORD_VEC4S, tableLayout, type TableLayout } from './layout.ts';
import { RangeAllocator } from './range-allocator.ts';

export type TableSection = 'instances' | 'meshes' | 'materials' | 'lights';

const RECORD_VEC4S: Record<TableSection, number> = {
    instances: INSTANCE_RECORD_VEC4S,
    meshes: MESH_RECORD_VEC4S,
    materials: MATERIAL_RECORD_VEC4S,
    lights: LIGHT_RECORD_VEC4S,
};

const CAPACITY_KEY: Record<TableSection, Exclude<keyof TableCapacities, 'materialListCapacity'>> = {
    instances: 'instanceCapacity',
    meshes: 'meshCapacity',
    materials: 'materialCapacity',
    lights: 'lightCapacity',
};

/** More dirty ranges than this in one frame are uploaded as one span. */
const MAX_DIRTY_RANGES = 64;

export interface TableCapacities {
    instanceCapacity: number;
    meshCapacity: number;
    materialCapacity: number;
    lightCapacity: number;
    materialListCapacity: number;
}

export class TableMirror {
    layout: TableLayout;
    words: Uint32Array<ArrayBuffer>;
    floats: Float32Array<ArrayBuffer>;
    /** Bumps whenever the layout (and so the buffer size) changes. */
    layoutVersion = 0;
    private readonly lists: RangeAllocator;
    /** Dirty `[start, end)` ranges in vec4s, unsorted, possibly overlapping. */
    private dirty: [number, number][] = [];

    constructor(capacities: TableCapacities) {
        this.layout = tableLayout(capacities);
        this.words = new Uint32Array(this.layout.totalVec4s * 4);
        this.floats = new Float32Array(this.words.buffer);
        this.lists = new RangeAllocator(this.layout.materialListCapacity);
        this.markDirty(0, this.layout.totalVec4s);
    }

    get byteLength(): number {
        return this.words.byteLength;
    }

    /** First word of record `index` in `section`. */
    recordWord(section: TableSection, index: number): number {
        const base = {
            instances: this.layout.instancesBase,
            meshes: this.layout.meshesBase,
            materials: this.layout.materialsBase,
            lights: this.layout.lightsBase,
        }[section];
        return (base + index * RECORD_VEC4S[section]) * 4;
    }

    /** Marks record `index` of `section` for upload. */
    markRecord(section: TableSection, index: number): void {
        const start = this.recordWord(section, index) / 4;
        this.markDirty(start, start + RECORD_VEC4S[section]);
    }

    clearRecord(section: TableSection, index: number): void {
        const start = this.recordWord(section, index);
        this.words.fill(0, start, start + RECORD_VEC4S[section] * 4);
        this.markRecord(section, index);
    }

    markHeader(): void {
        this.markDirty(0, HEADER_VEC4S);
    }

    /** Copies record `from` over record `to` in the same section (packing lights). */
    moveRecord(section: TableSection, from: number, to: number): void {
        const source = this.recordWord(section, from);
        this.words.copyWithin(this.recordWord(section, to), source, source + RECORD_VEC4S[section] * 4);
        this.markRecord(section, to);
    }

    /** Makes sure `section` can hold record `index`, doubling it (and relaying out) if not. */
    ensureRecord(section: TableSection, index: number): void {
        const key = CAPACITY_KEY[section];
        let capacity = this.layout[key];
        if (index < capacity) return;
        while (capacity <= index) capacity *= 2;
        this.relayout({ ...this.capacities(), [key]: capacity });
    }

    /** Allocates `count` words of material list; returns the word offset inside the list section. */
    allocateList(count: number): number {
        let offset = this.lists.allocate(count);
        if (offset === null) {
            const capacity = Math.max(this.layout.materialListCapacity * 2, this.lists.capacityToFit(count));
            this.relayout({ ...this.capacities(), materialListCapacity: capacity });
            offset = this.lists.allocate(count);
            if (offset === null) throw new Error('material list allocation failed after growing');
        }
        return offset;
    }

    releaseList(offset: number, count: number): void {
        this.lists.release(offset, count);
    }

    /** Words holding material list entries, for memory reports. */
    get listWordsUsed(): number {
        return this.lists.used;
    }

    /** Writes material list entries and marks them dirty. */
    writeList(offset: number, values: readonly number[]): void {
        const start = this.layout.materialListsBase * 4 + offset;
        for (let index = 0; index < values.length; index++) this.words[start + index] = values[index];
        this.markDirty(Math.floor(start / 4), Math.ceil((start + values.length) / 4));
    }

    /**
     * The ranges to upload since the last call, merged and sorted, in vec4s.
     * Clears the dirty list.
     */
    takeDirtyRanges(): [number, number][] {
        if (this.dirty.length === 0) return [];
        const sorted = this.dirty.sort((a, b) => a[0] - b[0]);
        this.dirty = [];
        const merged: [number, number][] = [];
        for (const range of sorted) {
            const last = merged[merged.length - 1];
            if (last && range[0] <= last[1]) {
                last[1] = Math.max(last[1], range[1]);
            } else {
                merged.push([range[0], range[1]]);
            }
        }
        if (merged.length > MAX_DIRTY_RANGES) return [[merged[0][0], merged[merged.length - 1][1]]];
        return merged;
    }

    private markDirty(start: number, end: number): void {
        this.dirty.push([start, end]);
        // Keep the list bounded between uploads; merging is cheap at this size.
        if (this.dirty.length > MAX_DIRTY_RANGES * 4) this.dirty = this.takeDirtyRanges();
    }

    private capacities(): TableCapacities {
        const { instanceCapacity, meshCapacity, materialCapacity, lightCapacity, materialListCapacity } = this.layout;
        return { instanceCapacity, meshCapacity, materialCapacity, lightCapacity, materialListCapacity };
    }

    private relayout(capacities: TableCapacities): void {
        const previous = this.layout;
        const previousWords = this.words;
        const next = tableLayout(capacities);
        const words = new Uint32Array(next.totalVec4s * 4);
        // Copy each section to its new place. The header is rewritten by the owner.
        const copySection = (fromBase: number, toBase: number, vec4s: number) =>
            words.set(previousWords.subarray(fromBase * 4, (fromBase + vec4s) * 4), toBase * 4);
        copySection(0, 0, HEADER_VEC4S);
        copySection(previous.instancesBase, next.instancesBase, previous.instanceCapacity * INSTANCE_RECORD_VEC4S);
        copySection(previous.meshesBase, next.meshesBase, previous.meshCapacity * MESH_RECORD_VEC4S);
        copySection(previous.materialsBase, next.materialsBase, previous.materialCapacity * MATERIAL_RECORD_VEC4S);
        copySection(previous.lightsBase, next.lightsBase, previous.lightCapacity * LIGHT_RECORD_VEC4S);
        copySection(previous.materialListsBase, next.materialListsBase, previous.materialListCapacity / 4);
        this.layout = next;
        this.words = words;
        this.floats = new Float32Array(words.buffer);
        this.lists.grow(next.materialListCapacity);
        this.layoutVersion++;
        this.dirty = [[0, next.totalVec4s]];
    }
}
