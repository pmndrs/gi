"""region_mean.py <exr>... [--crop PCT] [--ref VALUE]: linear mean / sd / min / max of the red channel over the centre
PCT% of each EXR (default 50%, i.e. the middle quarter of the image), via ImageMagick's HDRI build. With --ref, also
prints mean/VALUE. Used for the D25 constant-sky test (research/gi-1.0/native-capsaicin.md, 2026-11-19)."""
import subprocess, sys

args = sys.argv[1:]
crop, ref = 50, None
if '--crop' in args:
    i = args.index('--crop'); crop = float(args[i + 1]); del args[i:i + 2]
if '--ref' in args:
    i = args.index('--ref'); ref = float(args[i + 1]); del args[i:i + 2]
fmt = '%[fx:mean.r] %[fx:standard_deviation.r] %[fx:minima.r] %[fx:maxima.r] %[fx:mean.g] %[fx:mean.b]'
for f in args:
    out = subprocess.run(['magick', f, '-gravity', 'center', '-crop', '%g%%x%g%%+0+0' % (crop, crop), '+repage',
                          '-format', fmt, 'info:'], capture_output=True, text=True, check=True).stdout.split()
    m, sd, lo, hi, g, b = (float(x) for x in out)
    line = '%s  mean=%.5f sd=%.5f min=%.5f max=%.5f (g=%.5f b=%.5f)' % (f.replace('\\', '/').split('/')[-1], m, sd, lo, hi, g, b)
    if ref:
        line += '  ratio=%.4f' % (m / ref)
    print(line)
