"""sh_model.py [bent-normal.exr [crop%]]: offline replica of GI-1.2's frame-1 screen-probe SH path for a constant sky
(gi1.comp ProjectScreenProbes: every hemi-octahedral cell of an 8x8 probe holds L=1, weight 1/probe_size at the cell
centre; screen_probes.hlsl BentCone evaluation with ao=1). Prints E/(pi L) for the probe normal and for tilted normals;
with an OcclusionAndBentNormal dump it averages over the SSGI bent normals of the centre crop (D25, 2026-11-19)."""
import math, subprocess, re, sys
def hemi_oct(u, v):
    sx, sy = 2*u-1, 2*v-1
    st = ((sx+sy)*0.5, (sx-sy)*0.5)
    ax, ay = abs(st[0]), abs(st[1])
    dist = 1 - (ax+ay); rad = 1 - abs(dist)
    phi = 0 if rad == 0 else math.pi/4*((ay-ax)/rad + 1)
    r2 = rad*rad; s = rad*math.sqrt(2-r2)
    sgn = lambda a: (a > 0) - (a < 0)
    return (s*sgn(st[0])*math.cos(phi), s*sgn(st[1])*math.sin(phi), sgn(dist)*(1-r2))
def Y(d):
    x, y, z = d
    return [0.2820947917738781, -0.48860251190292*y, 0.4886025119029199*z, -0.48860251190292*x,
            0.5462742152960395*(x*y+y*x), -1.092548430592079*z*y, 0.9461746957575601*z*z-0.3153915652525201,
            -1.092548430592079*z*x, 0.5462742152960395*(x*x-y*y)]
def cc(n, theta=math.pi/2):
    x, y, z = n; s, c = math.sin(theta), math.cos(theta)
    b1 = 1.023326707946489*(1-c**3); b2 = (4-3*s**3)*s*s
    return [0.886226925452758*s*s, -b1*y, b1*z, -b1*x, 0.8580855308097834*b2*x*y, -0.8580855308097834*b2*y*z,
            0.2477079561003757*b2*(3*z*z-1), -0.8580855308097834*b2*x*z, 0.4290427654048917*b2*(x*x-y*y)]
P = 8
sh = [0.0]*9
for j in range(P):
    for i in range(P):
        d = hemi_oct((i+0.5)/P, (j+0.5)/P)   # local frame, z = probe normal
        for k, yk in enumerate(Y(d)):
            sh[k] += yk * 1.0 / P            # radiance / probe_size, as gi1.comp ProjectScreenProbes
E = lambda n: max(sum(a*b for a, b in zip(cc(n), sh)), 0.0)
print('frame-1 model, normal = probe normal: E/(pi L) = %.4f' % (E((0, 0, 1))/math.pi))
for deg in (5, 10, 17, 25, 30):
    t = math.radians(deg); print('  tilt %2d deg: %.4f' % (deg, E((math.sin(t), 0, math.cos(t)))/math.pi))
crop = sys.argv[2] if len(sys.argv) > 2 else '50'
if len(sys.argv) > 1:   # bent-normal EXR: average the model over its per-pixel normals (world +Y = probe normal)
    txt = subprocess.run(['magick', sys.argv[1], '-gravity', 'center', '-crop', '%s%%x%s%%+0+0' % (crop, crop), '+repage', '-sample', '128x72!', 'txt:-'],
                         capture_output=True, text=True).stdout.splitlines()[1:]
    acc = cnt = cos = 0.0
    for l in txt:
        m = re.search(r'\(\s*([\d.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+)', l)
        r, g, b = (float(v)/65535 for v in m.groups())
        n = (2*r-1, 2*b-1, 2*g-1); L = math.sqrt(sum(c*c for c in n)); n = tuple(c/L for c in n)  # world Y -> local z
        acc += E(n)/math.pi; cos += n[2]; cnt += 1
    print('averaged over %d SSGI bent normals (mean cos tilt %.4f): E/(pi L) = %.4f' % (cnt, cos/cnt, acc/cnt))
