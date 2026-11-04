/**
 * Probe volumes the DDGI pages use besides the manifests' own. Plain data, so
 * Node scripts (ddgi-test-view/render.ts) can import it too.
 */

/**
 * The point-light Cornell box's volume with every probe in the air, 2–5 cm
 * from the surfaces: T1.5a's measurement, before relocation existed. The
 * manifest's volume (0.5 m from y = −0.01) buries its bottom layer 1 cm in
 * the floor slab and its top layer 2 mm in the ceiling slab. Here the y
 * spacing is 0.49 m, so the layers run from 5 cm above the floor to 5 cm below
 * the ceiling; x and z are the manifest's. Since T1.5b the gate is the
 * manifest's volume (DECISIONS 2026-11-05); this one is an extra report.
 */
export const CORNELL_POINT_IN_AIR_VOLUME = { center: [2.78, 2.745, 2.8], size: [5.8, 5.39, 5.8], spacing: [0.5, 0.49, 0.5] } as const;
