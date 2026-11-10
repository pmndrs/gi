/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/math/spherical_harmonics.hlsl,
 * src/core/src/math/pack.hlsl, and the SH helpers of
 * src/core/src/render_techniques/gi1/screen_probes.hlsl.
 *
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Capsaicin's L2 spherical harmonics (nine coefficients) and the screen
 * probes' SH storage, in WGSL with an `rc` prefix. Code only, nothing bound:
 * the compute kernels (`screen-irradiance.ts`) and the shading module
 * (`shading.ts`, for the SH debug view) both include it. The CPU twins are in
 * `capsaicin-math.ts`.
 *
 * **Where a probe's SH lives** (register D26). Capsaicin keeps the nine
 * coefficients of probe tile `p` at `g_ScreenProbes_ProbeSHBuffer[9 * index +
 * i]`, `index = p.x + p.y * probe_count.x`, a structured buffer of `uint2`
 * (each coefficient's rgb and a weight as four halves, `packHalf4`). We keep
 * the same packed values in an `rg32uint` texture, the nine coefficients of
 * probe tile `p` in the 3 × 3 block at `3 p`, because the shading module's
 * debug view reads them in fragment shaders and our rule is that
 * fragment-readable data lives in textures (DECISIONS, "The three.js
 * integration"; behind it, three r187 has no supported way to bind an
 * outside buffer, and WebGPU's compatibility mode may allow no storage
 * buffers in the fragment stage: NOTES). Core WebGPU itself would allow it:
 * a read-only storage buffer can be bound in a fragment shader.
 */

/** Coefficients per probe: SH bands 0 to 2. */
export const SH_COEFFICIENTS = 9;
/** Texels across a probe's block of coefficients in the SH texture (3 × 3 = 9). */
export const SH_BLOCK_SIZE = 3;

export const SPHERICAL_HARMONICS_WGSL = /* wgsl */ `
// ---- math/spherical_harmonics.hlsl -----------------------------------------------------------

const RC_SH_COEFFICIENTS: u32 = ${SH_COEFFICIENTS}u;
const RC_SH_BLOCK_SIZE: u32 = ${SH_BLOCK_SIZE}u;

// SH_GetCoefficients (spherical_harmonics.hlsl:28-48): the real SH basis Y_i at a unit
// direction, Capsaicin's order and signs (band 1 is -y, z, -x).
fn rcSHGetCoefficients(direction: vec3f) -> array<f32, 9> {
    var coefficients: array<f32, 9>;
    let pz2 = direction.z * direction.z;
    coefficients[0] = 0.2820947917738781;
    coefficients[2] = 0.4886025119029199 * direction.z;
    coefficients[6] = 0.9461746957575601 * pz2 + -0.3153915652525201;
    let fC0 = direction.x;
    let fS0 = direction.y;
    let fTmpA = -0.48860251190292;
    coefficients[3] = fTmpA * fC0;
    coefficients[1] = fTmpA * fS0;
    let fTmpB = -1.092548430592079 * direction.z;
    coefficients[7] = fTmpB * fC0;
    coefficients[5] = fTmpB * fS0;
    let fC1 = direction.x * fC0 - direction.y * fS0;
    let fS1 = direction.x * fS0 + direction.y * fC0;
    let fTmpC = 0.5462742152960395;
    coefficients[8] = fTmpC * fC1;
    coefficients[4] = fTmpC * fS1;
    return coefficients;
}

// SH_GetCoefficients_ClampedCosine_Cone (spherical_harmonics.hlsl:78-104): the clamped
// cosine lobe about cosineLobeDirection, cut to a cone of half-angle coneThetaMax, as SH,
// already convolved (band l carries its A_l), so sum_i coefficient_i * c_i is the irradiance
// of a radiance whose projection is c. At coneThetaMax = pi/2 (no occlusion) it is the plain
// clamped cosine: band factors pi, 2 pi / 3 and pi / 4 times the SH basis.
fn rcSHGetCoefficientsClampedCosineCone(cosineLobeDirection: vec3f, coneThetaMax: f32) -> array<f32, 9> {
    var coefficients: array<f32, 9>;
    let sinThetaMax = sin(coneThetaMax);
    let cosThetaMax = cos(coneThetaMax);
    let sinThetaMax2 = sinThetaMax * sinThetaMax;
    let sinThetaMax3 = sinThetaMax2 * sinThetaMax;
    let cosThetaMax2 = cosThetaMax * cosThetaMax;
    let cosThetaMax3 = cosThetaMax2 * cosThetaMax;

    let band1Factor = 1.023326707946489 * (1.0 - cosThetaMax3);
    let band2Factor = (4.0 - 3.0 * sinThetaMax3) * sinThetaMax2;

    let d = cosineLobeDirection;
    coefficients[0] = 0.886226925452758 * sinThetaMax2;

    coefficients[1] = -band1Factor * d.y;
    coefficients[2] = band1Factor * d.z;
    coefficients[3] = -band1Factor * d.x;

    coefficients[4] = 0.8580855308097834 * band2Factor * d.x * d.y;
    coefficients[5] = -0.8580855308097834 * band2Factor * d.y * d.z;
    coefficients[6] = 0.2477079561003757 * band2Factor * (3.0 * d.z * d.z - 1.0);
    coefficients[7] = -0.8580855308097834 * band2Factor * d.x * d.z;
    coefficients[8] = 0.4290427654048917 * band2Factor * (d.x * d.x - d.y * d.y);
    return coefficients;
}

// ---- math/pack.hlsl, screen_probes.hlsl ------------------------------------------------------

// The largest finite half: SH coefficients are clamped to it before packing (register D28).
const RC_SH_LARGEST_HALF: f32 = 65504.0;

// ScreenProbes_PackSHColor (screen_probes.hlsl:169-172): packHalf4 (pack.hlsl:423-428).
// f32tof16 turns a value past the half range into infinity; WGSL's pack2x16float gives an
// indeterminate value there (WGSL 17.9.9; 16.9 in older drafts), so the coefficient is clamped first (D28).
fn rcPackSHColor(shColor: vec4f) -> vec2u {
    let clamped = clamp(shColor, vec4f(-RC_SH_LARGEST_HALF), vec4f(RC_SH_LARGEST_HALF));
    return vec2u(pack2x16float(clamped.xy), pack2x16float(clamped.zw));
}

// ScreenProbes_UnpackSHColor (screen_probes.hlsl:175-178): unpackHalf4 (pack.hlsl:456-460).
fn rcUnpackSHColor(packedSHColor: vec2u) -> vec4f {
    return vec4f(unpack2x16float(packedSHColor.x), unpack2x16float(packedSHColor.y));
}

// Where coefficient 'coefficient' of probe tile 'probe' lives in the SH texture: the 3 x 3
// block at 3 * probe, in row order (register D26; Capsaicin's index is 9 * index + i).
fn rcProbeSHTexel(probe: vec2u, coefficient: u32) -> vec2u {
    return probe * RC_SH_BLOCK_SIZE + vec2u(coefficient % RC_SH_BLOCK_SIZE, coefficient / RC_SH_BLOCK_SIZE);
}
`;

/**
 * `ScreenProbes_CalculateSHIrradiance_BentCone` (screen_probes.hlsl:200-216)
 * over an SH texture given by name: the irradiance at a surface facing
 * `normal` from probe tile `probe`'s SH, through a cone of half-angle
 * `acos(sqrt(1 - ao))`, never negative. A function of the texture so the
 * compute kernel (a global) and the shading module (a parameter) share it.
 */
export function shIrradianceWgsl(functionName: string, textureParameter: string): string {
    return /* wgsl */ `
fn ${functionName}(normal: vec3f, ao: f32, probe: vec2u, ${textureParameter}: texture_2d<u32>) -> vec3f {
    let clampedCosineSH = rcSHGetCoefficientsClampedCosineCone(normal, acos(sqrt(saturate(1.0 - ao))));
    var irradiance = vec3f(0.0);
    for (var i = 0u; i < RC_SH_COEFFICIENTS; i++) {
        irradiance += clampedCosineSH[i] * rcUnpackSHColor(textureLoad(${textureParameter}, rcProbeSHTexel(probe, i), 0).xy).xyz;
    }
    return max(irradiance, vec3f(0.0));
}
`;
}
