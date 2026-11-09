/**
 * CPU references for the platform-features page (RC1.4): what each GPU check
 * should read back. Kept free of WebGPU so `pnpm test` covers them.
 */

/** Workgroup counts an indirect dispatch's arguments hold. */
export interface DispatchArguments {
    x: number;
    y: number;
    z: number;
}

/**
 * The split the page's argument-writing kernel does on the GPU (and RC1.3's
 * writer will do): `groups` workgroups as `[x, y, 1]`, with x at most
 * `maxPerDimension`, so counts past `maxComputeWorkgroupsPerDimension` still
 * dispatch. y rounds up, so x * y can exceed `groups` (the kernel's own
 * bounds check skips the extra workgroups). Zero groups gives `[0, 1, 1]`.
 */
export function splitDispatch(groups: number, maxPerDimension: number): DispatchArguments {
    if (groups === 0) return { x: 0, y: 1, z: 1 };
    return { x: Math.min(groups, maxPerDimension), y: Math.ceil(groups / maxPerDimension), z: 1 };
}

/**
 * How many workgroups an indirect dispatch with these arguments runs. The
 * WebGPU spec makes an indirect dispatch with any dimension over
 * `maxComputeWorkgroupsPerDimension` a no-op (it can't be validated on the
 * CPU, so it is skipped rather than refused).
 */
export function workgroupsRun(args: DispatchArguments, maxComputeWorkgroupsPerDimension: number): number {
    const overLimit = [args.x, args.y, args.z].some((count) => count > maxComputeWorkgroupsPerDimension);
    return overLimit ? 0 : args.x * args.y * args.z;
}

/** IEEE 754 binary16 bits to a number: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits. */
export function halfToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) return sign * mantissa * 2 ** -24; // subnormal: no implicit leading 1
    if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
    return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/**
 * The f16 kernel's cases: each is computed in f16 on the GPU from f32
 * inputs, and every input and expected result is exactly representable in
 * f16, so the readback must match bit for bit. `onePlusTiny` is the case that
 * tells real f16 arithmetic from f32 promoted: 1 + 0.0001 is 1.0001 in f32,
 * but f16's step at 1.0 is 2^-10 (0.000977), so in f16 it is exactly 1.0
 * whichever way the implementation rounds.
 */
export const F16_INPUTS = {
    one: 1,
    tiny: 0.0001,
    multiplicandA: 1.5,
    multiplicandB: 2.25,
    squareOf: 6.25,
    largestFinite: 65504,
    dividend: -7,
    divisor: 2,
} as const;

/** Results the kernel stores to `array<f16>`, in this order. */
export const F16_EXPECTED = [
    { name: 'product', value: 3.375 }, // 1.5 * 2.25
    { name: 'onePlusTiny', value: 1 }, // see F16_INPUTS
    { name: 'squareRoot', value: 2.5 }, // sqrt(6.25)
    { name: 'largestFinite', value: 65504 }, // f16's largest finite value survives the conversion
    { name: 'dot', value: 4 }, // dot((0.5, 1.5, -2, 4), (2, 2, 0.25, 0.125)) = 1 + 3 - 0.5 + 0.5
    { name: 'quotient', value: -3.5 }, // -7 / 2
] as const;

/** The value one invocation of the subgroup check adds: small, so 8,192 of them sum well inside u32. */
export function subgroupInput(index: number): number {
    return (Math.imul(index, 2654435761) >>> 20) & 0xfff;
}

/**
 * Checks each invocation's `subgroupAdd` against the sum over its own
 * subgroup, which the kernel identifies by the global index of the
 * subgroup's first active lane (`subgroupBroadcastFirst`). The CPU doesn't
 * assume a lane order or size (NOTES: lane order isn't the local index).
 * Returns how many invocations read a wrong sum.
 */
export function wrongSubgroupSums(inputs: ArrayLike<number>, sums: ArrayLike<number>, leaders: ArrayLike<number>): number {
    const totals = new Map<number, number>();
    for (let index = 0; index < inputs.length; index++) {
        totals.set(leaders[index], (totals.get(leaders[index]) ?? 0) + inputs[index]);
    }
    let wrong = 0;
    for (let index = 0; index < inputs.length; index++) {
        if (sums[index] !== totals.get(leaders[index])) wrong++;
    }
    return wrong;
}
