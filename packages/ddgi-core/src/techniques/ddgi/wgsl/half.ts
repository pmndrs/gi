/**
 * Stochastic rounding to half floats, for writing DDGI's half-float atlases
 * (DECISIONS 2026-11-05). Its own module so a GPU test can run exactly this
 * WGSL (tools/gpu/pages/ddgi-wgsl-units).
 */

export const DDGI_STOCHASTIC_HALF_WGSL = /* wgsl */ `
// Smallest normal half float, 2^-14. Below it half floats are evenly spaced.
const DDGI_SMALLEST_NORMAL_HALF: f32 = 6.103515625e-5;

// Rounds a value to a neighbouring half float at random, in proportion to how
// close it is to each (stochastic rounding), so it is exact when packed.
// Why: with hysteresis 0.97 an update moves a texel by 3% of the gap to its
// estimate. Rounded to the nearest half, a small gap moves it by less than
// half a step, so it never moves, and the field stalls short of its answer:
// the furnace room settled 4.6% low that way (T1.5a), against 0.6% with
// this. Rounded at random, the expected value moves by exactly the blended
// amount. 'random' is uniform in [0, 1).
fn ddgiStochasticHalf(value: f32, random: f32) -> f32 {
    // Half floats have 10 fraction bits, so neighbours are 2^(e - 10) apart for
    // a value in [2^e, 2^(e+1)); below the smallest normal the step is fixed.
    // frexp gives e exactly (magnitude = fraction * 2^exponent, fraction in [0.5, 1)).
    let magnitude = max(abs(value), DDGI_SMALLEST_NORMAL_HALF);
    let step = exp2(f32(frexp(magnitude).exp - 1) - 10.0);
    let steps = value / step;
    let below = floor(steps);
    return (below + select(0.0, 1.0, random < steps - below)) * step;
}
`;
