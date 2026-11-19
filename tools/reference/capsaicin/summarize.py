"""Group a Capsaicin GI-1.2 timing CSV (benchmark-first-frame averages) into the pass groups used in the doc.
usage: python summarize.py <run dir> [...]   (prints one markdown row per run plus the unassigned passes, if any)"""
import csv, glob, os, sys

GROUPS = [
    ('probe tracing / cell population', ['SampleScreenProbes', 'PopulateScreenProbesCells', 'PopulateMultibounceCells',
                                         'PopulateRadianceCache', 'GenerateReservoirs', 'GenerateMultibounceReservoirs',
                                         'CalculateLightSamplerBounds', 'CalculateLightSamplerConfiguration', 'BuildLightSampler']),
    ('cache update / resolve', ['PurgeRadianceCache', 'UpdateRadianceCache', 'UpdateMultibounceCells', 'ResolveCells']),
    ('screen probes', ['ReprojectScreenProbes', 'LookupScreenProbes', 'SpawnScreenProbes', 'PatchScreenProbes', 'BlendScreenProbes',
                       'ReorderScreenProbes', 'FilterProbeMask', 'FilterScreenProbes', 'ProjectScreenProbes', 'InterpolateScreenProbes']),
    ('denoise / resolve', ['ReprojectGI', 'FilterGI', 'ResolveGI1', 'MarkFireflies', 'CleanupFireflies']),
    ('reflections', ['TraceReflections', 'ResolveReflections Atrous First', 'ResolveReflections Atrous 1',
                     'ResolveReflections Atrous 2', 'ResolveReflections Atrous Last', 'ReprojectReflections']),
]


def summarize(run):
    csv_path = [p for p in glob.glob(os.path.join(run, '*.csv')) if 'GI-1.2' in p][0]
    rows = {r[0]: float(r[1]) for r in list(csv.reader(open(csv_path)))[1:]}
    out = {}
    seen = set()
    for g, passes in GROUPS:
        out[g] = sum(rows['GI-1.2 / ' + p] for p in passes)
        seen.update('GI-1.2 / ' + p for p in passes)
    out['GI-1.2 block'] = rows['GI-1.2']
    out['diffuse subtotal'] = sum(out[g] for g, _ in GROUPS if g != 'reflections')
    out['unassigned'] = [k for k in rows if k.startswith('GI-1.2 / ') and k not in seen]
    out['visibility buffer'] = rows['Visibility buffer']
    out['SSGI'] = rows['SSGI']
    out['FSR'] = rows['FSR']
    out['whole frame (top-level sum)'] = sum(v for k, v in rows.items() if ' / ' not in k and k not in ('GI-1.2',)) + rows['GI-1.2']
    return out


for run in sys.argv[1:]:
    r = summarize(run)
    print(os.path.basename(run))
    for k, v in r.items():
        print('  %-34s %s' % (k, v if isinstance(v, list) else '%.3f' % v))
