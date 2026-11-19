"""preview.py <exr> <out.png> [target]: linear EXR -> sRGB PNG with exposure scaled so the 90th-percentile luminance is `target` (default 0.5).
Viewing aid only; prints the unscaled median/mean luminance, which is the number to compare between runs."""
import subprocess, sys, re
exr, out = sys.argv[1], sys.argv[2]
target = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
txt = subprocess.run(['magick', exr, '-resize', '256x144!', 'txt:-'], capture_output=True, text=True).stdout
lum = []
for l in txt.splitlines()[1:]:
    m = re.search(r'\(\s*([\d.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+)', l)
    if m:
        r, g, b = (float(x) for x in m.groups()); lum.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
lum = [v / 65535.0 for v in lum]  # this magick build is Q16: txt: values are 0-65535
lum.sort(); med = lum[len(lum) // 2]; p90 = lum[int(len(lum) * 0.9)]; mean = sum(lum) / len(lum)
print('linear luminance median=%.5g p90=%.5g mean=%.5g (256x144 box-downsample)' % (med, p90, mean))
subprocess.run(['magick', exr, '-evaluate', 'Multiply', str(target / max(p90, 1e-9)), '-colorspace', 'sRGB', out])
