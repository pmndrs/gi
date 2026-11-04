"""
Unit conversions from the manifest (three.js physical units) to Blender / Cycles.

This is the part of a reference render that usually goes wrong, so every
conversion is here, in one place, with the reason. `calibrate.py` renders a
known case for each one and compares it with the analytic answer; run it
whenever Blender is upgraded.

Everything is in a linear scene space. Pixel values are radiance L, in the
same units the manifest uses for emissive surfaces.

How the numbers were pinned down (measured in Blender 4.5.0, CPU, see
calibrate.py): a white (albedo 1) diffuse plane seen straight on has pixel
value L = E / pi, where E is the irradiance on it. Then

  * a Cycles sun of strength S gives E = S              (S is W/m^2, i.e. lux here)
  * a Cycles point light of P watts, at distance d, gives E = P / (4 pi d^2)
    (that is a radiant intensity I = P / (4 pi) per steradian)
  * an emission shader of strength S and colour c gives radiance S * c
"""

import math

# ---- Directional light -----------------------------------------------------

def directional_to_sun_strength(irradiance_lux):
    """
    Manifest directional `intensity` is the irradiance on a surface facing the
    light (three.js DirectionalLight, physical units). A Cycles sun's
    `strength` is exactly that irradiance, so the value passes through.
    """
    return irradiance_lux


# ---- Point and spot lights -------------------------------------------------

def radiant_intensity_to_blender_watts(candela):
    """
    Manifest point/spot `intensity` is radiant intensity I (three.js: irradiance
    at distance d is I / d^2). Cycles' point light of P "watts" has
    I = P / (4 pi) (it is the power of an isotropic emitter, spread over
    4 pi steradians), so P = 4 pi * I.

    A Blender spot light uses the same P with the same I along its axis: the
    cone only cuts light off, it does not concentrate it. three.js is the same
    (SpotLight intensity is the candela along the axis), so spots use this too.
    """
    return 4.0 * math.pi * candela


def spot_cone_to_blender(outer_cone_rad, inner_cone_rad):
    """
    Returns (spot_size, spot_blend) for a Blender spot light.

    Manifest cones are half-angles; Blender's `spot_size` is the FULL angle.

    three.js attenuates a spot by smoothstep(cos(outer), cos(inner), cos(theta)).
    Cycles uses smoothstep(0, blend * (1 - cos(outer)), cos(theta) - cos(outer)).
    The two are the same function when
        blend * (1 - cos(outer)) = cos(inner) - cos(outer),
    so the blend is solved for exactly rather than approximated.
    """
    cos_outer = math.cos(outer_cone_rad)
    cos_inner = math.cos(inner_cone_rad)
    blend = (cos_inner - cos_outer) / (1.0 - cos_outer)
    return 2.0 * outer_cone_rad, min(max(blend, 0.0), 1.0)


# ---- Emissive surfaces and the environment ---------------------------------

def emissive_radiance_to_strength(radiance):
    """
    An emissive surface in the manifest geometry has radiance L (the glTF
    emissive colour times `KHR_materials_emissive_strength`). A Cycles emission
    shader of strength S and colour c gives radiance S * c, with no pi in
    between, so the Blender glTF importer's values are already right and
    nothing is converted. This function exists so the claim is written down
    next to the others and is checked by calibrate.py.
    """
    return radiance


def environment_to_background_strength(intensity):
    """
    Manifest `environment.intensity` multiplies a radiance. Cycles' Background
    shader emits `colour * strength` as radiance, so it passes through.
    """
    return intensity


# ---- Coordinates -----------------------------------------------------------

def to_blender_space(vector):
    """
    Manifest/three.js space is right-handed, +Y up, the camera looks down -Z.
    Blender is right-handed, +Z up. Mapping (x, y, z) -> (x, -z, y) is a
    proper rotation (determinant +1), the same one Blender's glTF importer
    applies to the geometry, so lights and cameras land exactly on the meshes.
    """
    x, y, z = vector
    return (x, -z, y)
