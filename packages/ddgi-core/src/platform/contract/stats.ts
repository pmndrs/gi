/**
 * Stats and debug data. Every technique reports the same shapes so the
 * viewer, tests and comparisons work for all of them unchanged.
 */

/** One timed GPU pass. */
export interface PassTiming {
    /** e.g. `ddgi/trace`, `trace/tlas-build`. */
    readonly name: string;
    /** GPU milliseconds, or `undefined` when timestamps aren't available. */
    readonly gpuMs: number | undefined;
    /** The frame these timings came from (they arrive a few frames late). */
    readonly frame: number;
}

/** GPU memory owned by the core, by what it's for. */
export interface MemoryReport {
    readonly sceneCopy: number;
    readonly acceleration: number;
    /**
     * The texture store (M4): 0.33 MB per texture (a 256² layer with mips,
     * the array grown in doubles), plus build buffers and staging copies in
     * the frames that make layers.
     */
    readonly textures: number;
    /** Per technique id. */
    readonly techniques: Readonly<Record<string, number>>;
    readonly total: number;
}

export interface GIStats {
    readonly frame: number;
    /** Empty unless `gpuTiming` is on and `timestamp-query` is available. */
    readonly passes: readonly PassTiming[];
    readonly memory: MemoryReport;
    /**
     * Technique counters for the latest frame, by technique id. Plain-language
     * names, e.g. `{ ddgi: { 'probes updated': 512, 'rays traced': 131072 } }`.
     */
    readonly counters: Readonly<Record<string, Readonly<Record<string, number>>>>;
    /** CPU milliseconds spent in the latest `encode` call, including scene uploads. */
    readonly encodeCpuMs: number;
    /**
     * Compute pipelines the platform and its techniques created while the
     * platform was ready (after `prepare`, and not inside a later `prepare`):
     * shader compiles that frames or scene edits caused. It should stay 0.
     * Scene tables grow by reallocating buffers, never by changing shader
     * code, so adding and removing objects compiles nothing. Counted since
     * the platform was created.
     */
    readonly recompiles: number;
}

/**
 * Debug data a technique exposes. The core only provides data; each adapter
 * draws it in its own engine.
 */
export type DebugView =
    | {
          /**
           * Colors the shaded image through the shading module's
           * `giDebugColor` (GI only, update freshness, coverage...).
           */
          readonly kind: 'screen';
          readonly id: number;
          readonly label: string;
          /** One sentence for the viewer's hover help. */
          readonly description: string;
      }
    | {
          /**
           * Points to draw as small spheres (probes, surfels). Stored in an
           * rgba32float texture, one point per texel in row order: xyz
           * position and a state value in w, so adapters that can't bind
           * buffers can still draw them.
           */
          readonly kind: 'points';
          readonly id: number;
          readonly label: string;
          readonly description: string;
          readonly texture: () => GPUTexture;
          readonly count: () => number;
          /** Meaning of each state value in w, for the legend: `{ 0: 'active', 1: 'inside geometry' }`. */
          readonly states: Readonly<Record<number, string>>;
      };
