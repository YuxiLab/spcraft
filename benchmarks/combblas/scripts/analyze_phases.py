#!/usr/bin/env python3
"""Summarize wall phases, sampled worker shares, and controlled diagnostics."""
import argparse
import collections
import csv
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    data = collections.defaultdict(list)
    calls = collections.defaultdict(dict)
    for path in sorted(args.directory.glob('*.csv')):
        if path.name == 'phase-summary.csv':
            continue
        reader = csv.DictReader(path.open())
        if not {'backend', 'phase', 'kind', 'seconds'}.issubset(reader.fieldnames or []):
            continue
        for row in reader:
            key = (row['dataset'], int(row['threads']), row['backend'], row['kind'], row['phase'])
            seconds = float(row['seconds'])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f'invalid phase time in {path}')
            data[key].append(seconds)
            if row['kind'] == 'wall':
                calls[(row['dataset'], row['threads'], row['iteration'], row['backend'])][row['phase']] = seconds
    if not data:
        raise ValueError('no phase samples')
    for key, phases in calls.items():
        total = phases['total']
        remainder = total - sum(value for phase, value in phases.items() if phase != 'total')
        if abs(remainder) > max(1e-5, total * .001):
            raise ValueError(f'phase times do not reconcile for {key}: {remainder}')
    summary = []
    for key, values in sorted(data.items()):
        summary.append(dict(zip(('dataset', 'threads', 'backend', 'kind', 'phase'), key),
                            samples=len(values), mean_ms=1000*statistics.mean(values),
                            median_ms=1000*statistics.median(values),
                            cv_pct=100*statistics.pstdev(values)/statistics.mean(values) if sum(values) else 0))
    with (args.directory/'phase-summary.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    def mean(dataset, threads, backend, phase):
        values=data.get((dataset, threads, backend, 'wall', phase), [0.0])
        return 1000*statistics.mean(values)
    def median(dataset, threads, backend):
        return 1000*statistics.median(data[(dataset,threads,backend,'wall','total')])
    cases=sorted({(k[0],k[1]) for k in data})
    report=['# SpGEMM phase measurements', '',
            'Wall phases are measured outside the OpenMP loops. Tables show means so phase '
            'contributions add; total-call and diagnostic tables show medians. The sampled '
            'worker table is a separate experiment and shows shares of sampled active elapsed '
            'time across workers, not fractions of wall time. It excludes scheduling/wait time '
            'and may perturb very short columns.', '',
            '## Whole-call controls and diagnostics', '',
            '| Input | Threads | CSC control ms | DCSC control ms | Comb native ms | DCSC coarse/control | Comb coarse/control | Dense lookup ms | Static ms | Dense+static ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,t in cases:
        get=lambda b: median(ds,t,b)
        report.append(f'| {ds} | {t} | {get("CSC-control"):.3f} | {get("DCSC-control"):.3f} '
                      f'| {get("CombBLAS-control"):.3f} | {get("DCSC-profile")/get("DCSC-control"):.3f} '
                      f'| {get("CombBLAS-profile")/get("CombBLAS-control"):.3f} '
                      f'| {get("DCSC-dense-lookup"):.3f} | {get("DCSC-static"):.3f} '
                      f'| {get("DCSC-dense-static"):.3f} |')
    report += ['', '## Wall-clock phases (mean milliseconds)', '',
               'CombBLAS work and symbolic routines each build additional auxiliary lookup '
               'structures internally; their time stays in those routines. The outer lookup '
               'row is not the full cost of all CombBLAS lookup operations.', '']
    stages=[('Lookup build', ['lookup_build'], ['lookup_build']),
            ('Scratch/thread setup', ['validate','scratch_setup'], ['thread_setup']),
            ('Work bound/capacity', ['work_bound','symbolic_capacity'], ['work_bound','work_prefix']),
            ('Symbolic', ['symbolic'], ['symbolic']),
            ('Prefix/numeric capacity', ['prefix_numeric_capacity'], ['output_prefix']),
            ('Output allocation/setup', ['output_allocate'], ['output_allocate','numeric_scratch_setup']),
            ('Numeric incl. sort/write', ['numeric'], ['numeric']),
            ('Cleanup/free', ['return_cleanup','output_free'], ['internal_cleanup','return_cleanup','output_free'])]
    for ds,t in cases:
        report += [f'### {ds}, {t} threads', '', '| Step | CSC ms | DCSC ms | CombBLAS native ms |', '|---|---:|---:|---:|']
        for name,sp,cb in stages:
            x=sum(mean(ds,t,'CSC-profile',p) for p in sp)
            y=sum(mean(ds,t,'DCSC-profile',p) for p in sp)
            z=sum(mean(ds,t,'CombBLAS-profile',p) for p in cb)
            report.append(f'| {name} | {x:.3f} | {y:.3f} | {z:.3f} |')
        report.append(f'| Total | {mean(ds,t,"CSC-profile","total"):.3f} '
                      f'| {mean(ds,t,"DCSC-profile","total"):.3f} | {mean(ds,t,"CombBLAS-profile","total"):.3f} |')
    if any(k[2] == 'Tuples-profile' for k in data):
        report += ['', '## Matched native tuple output', '',
                   'Both implementations use DCSC input and sorted std::tuple arrays. '
                   'Allocation, initialization, symbolic/numeric work, sorting and destruction '
                   'are included. Means reconcile by phase; controls quantify clock perturbation.', '',
                   '| Input | Threads | SpCraft control ms | Comb control ms | SpCraft coarse/control |',
                   '|---|---:|---:|---:|---:|']
        for ds,t in cases:
            report.append(f'| {ds} | {t} | {median(ds,t,"Tuples-control"):.3f} '
                          f'| {median(ds,t,"CombBLAS-control"):.3f} '
                          f'| {median(ds,t,"Tuples-profile")/median(ds,t,"Tuples-control"):.3f} |')
        for ds,t in cases:
            report += [f'### Native tuples: {ds}, {t} threads', '',
                       '| Step | SpCraft ms | CombBLAS ms |', '|---|---:|---:|']
            for name,sp,cb in stages:
                x=sum(mean(ds,t,'Tuples-profile',p) for p in sp)
                y=sum(mean(ds,t,'CombBLAS-profile',p) for p in cb)
                report.append(f'| {name} | {x:.3f} | {y:.3f} |')
            report.append(f'| Total | {mean(ds,t,"Tuples-profile","total"):.3f} '
                          f'| {mean(ds,t,"CombBLAS-profile","total"):.3f} |')
    report += ['', '## Sampled numeric worker time', '',
               'DCSC hashing includes repeated left-column lookup; CombBLAS lookup is timed '
               'separately before hashing. Percentages normalize the summed sampled durations. '
               'Sampling selects approximately 8,192 evenly spaced candidate columns per call '
               '(all columns for smaller inputs); these shares are descriptive, not estimates '
               'of a critical thread or an unbiased distribution of irregular work.', '',
               '| Input | Threads | Backend | Lookup % | Alloc/init % | Hash/expand % | Compact % | Sort % | Write % | Free % | Sampled/coarse total |',
               '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    parts=['lookup','alloc_init','lookup_multiply_hash','compact','sort','write','scratch_free']
    for ds,t in cases:
        for backend in ['DCSC','CombBLAS']:
            values=[statistics.mean(data.get((ds,t,backend+'-sampled','sampled_worker',p),[0])) for p in parts]
            total=sum(values) or 1
            ratio=median(ds,t,backend+'-sampled')/median(ds,t,backend+'-profile')
            report.append(f'| {ds} | {t} | {backend} | '+' | '.join(f'{100*v/total:.1f}' for v in values)+f' | {ratio:.3f} |')
    report += ['', '## Tuple-to-DCSC conversion (mean milliseconds)', '',
               '| Input | Threads | Native call | Compression | DCSC free | Tuple free |', '|---|---:|---:|---:|---:|---:|']
    for ds,t in cases:
        report.append(f'| {ds} | {t} | '+' | '.join(f'{mean(ds,t,"CombBLAS-conversion",p):.3f}' for p in ['native_call','tuple_to_dcsc','dcsc_free','output_free'])+' |')
    report += ['', '## Interpretation limits', '',
               '- Dense lookup is valid only when sorted unique DCSC column IDs cover every logical column (nzc==n). Otherwise it falls back to the original lookup.',
               '- Static and dense+static change scheduling only, or scheduling plus that lookup shortcut. These diagnostic variants are not production changes.',
               '- Controls retain the exact production kernels. Coarse and sampled copies are regenerated from source with checked insertion anchors.',
               '- Every implementation and diagnostic is verified against Eigen before timing at each thread count. Phase sums are checked against elapsed total; raw CSV samples are retained.',
               '- Neither sampling nor equal numerical results repairs the pre-existing upstream numThreads data races. No hardware-counter claim is made from these clocks.']
    (args.directory/'PhaseReport.md').write_text('\n'.join(report)+'\n')
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        datasets=sorted({ds for ds,t in cases if t==128})
        fig,axes=plt.subplots(1,len(datasets),figsize=(4*len(datasets),5),layout='constrained',squeeze=False)
        colors=plt.get_cmap('tab10').colors
        for ax,ds in zip(axes[0],datasets):
            base=[0.,0.,0.]
            for i,(name,sp,cb) in enumerate(stages):
                heights=[sum(mean(ds,128,b,p) for p in phases) for b,phases in
                         [('CSC-profile',sp),('DCSC-profile',sp),('CombBLAS-profile',cb)]]
                ax.bar(['CSC','DCSC','Comb native'],heights,bottom=base,label=name,color=colors[i])
                base=[a+b for a,b in zip(base,heights)]
            ax.set(title=ds,ylabel='Mean phase wall time (ms), 128 threads')
            ax.tick_params(axis='x',labelrotation=20)
        axes[0][-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1))
        fig.savefig(args.directory/'phases.png',dpi=180)
        fig.savefig(args.directory/'phases.pdf')
        plt.close(fig)
    print(args.directory/'PhaseReport.md')


if __name__=='__main__':
    main()
