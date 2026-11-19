/**
 * Small pure helpers a technique may need, public through the contract so a
 * technique never reaches into platform internals (Dennis, 2026-11-19).
 * Explained in docs/api/platform.md.
 */

/** The parts of two scene fingerprints that differ (saved bakes use it to say what changed). */
export { fingerprintDifferences } from '../scene/scene-fingerprint.ts';
/** binary16 bits to a number (the reading half of the half-float files and saved bakes). */
export { halfToFloat } from '../sky/half-float.ts';
