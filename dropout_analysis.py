"""Independent, matched-seed segmentation feature-dropout experiment on fixed splits."""
import copy
import json
from pathlib import Path
import re
import statistics

from annotation_cycle import atomic_text, digest, load_prepared, validate_contract
from cross_validation import summarize_runs, train_and_evaluate


def add_feature_dropout(model, probability):
    """Use standard PyTorch modules so saved models need no custom inference hooks/classes."""
    import torch.nn as nn
    from ultralytics.nn.modules import SPPF
    if not 0 <= probability < 1:
        raise ValueError('Feature dropout probability must be in [0, 1)')
    block = model.model[len(model.yaml['backbone'])-1]
    if not isinstance(block, SPPF):
        raise ValueError('Feature dropout requires a YOLOv8 SPPF backbone output')
    if any(isinstance(module, nn.Dropout2d) for module in block.modules()):
        raise ValueError('Feature dropout already installed; initialize from a fresh base checkpoint')
    block.cv2.act = nn.Sequential(block.cv2.act, nn.Dropout2d(probability))


def summarize_sweep(rate_reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    baseline = next(reports for rate, reports in rate_reports if rate == 0)
    rows, summaries = [], []
    for rate, reports in rate_reports:
        summary = summarize_runs(reports, output/f'rate-{rate}',
                                 method=f'Fixed-split dropout analysis, p={rate:g}; matched seeds across rates')
        summaries.append({'dropout_rate':rate, 'summary':summary})
        for row in summary['metrics']:
            base_by_seed = {report['seed']:report for report in baseline}
            if {report['seed'] for report in reports} != set(base_by_seed):
                raise ValueError('Dropout comparison requires matching seeds at every rate')
            deltas = [report[row['split']]['metrics'][row['metric']] -
                      base_by_seed[report['seed']][row['split']]['metrics'][row['metric']] for report in reports]
            rows.append(dict(row, dropout_rate=rate, paired_delta_mean=statistics.mean(deltas),
                             paired_delta_std=statistics.stdev(deltas)))
    figure, axes = plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for row_index, kind in enumerate(('B','M')):
        for col, metric in enumerate(('precision','recall','mAP50','mAP50-95')):
            ax = axes[row_index,col]
            for split in ('val','test'):
                selected = sorted((row for row in rows if row['split']==split and
                                   row['metric']==f'metrics/{metric}({kind})'), key=lambda row:row['dropout_rate'])
                ax.errorbar([row['dropout_rate'] for row in selected], [row['mean'] for row in selected],
                            yerr=[row['std'] for row in selected], fmt='o-',capsize=4,label=split)
            ax.set_title(('Box ' if kind=='B' else 'Mask ')+metric)
            ax.set_xlabel('Dropout probability'); ax.set_ylabel('Mean ± sample SD'); ax.legend()
    figure.savefig(output/'dropout_effect.png',dpi=160); plt.close(figure)
    result = {'analysis':'fixed-split matched-seed dropout sweep, separate from CV',
              'location':'Dropout2d after backbone SPPF, disabled for validation/test/inference',
              'metrics':rows, 'rates':summaries,
              'all_converged':all(item['summary']['all_converged'] for item in summaries),
              'limitations':'Use validation to select a rate, not these repeated test measurements. SD is not a confidence interval.'}
    atomic_text(output/'summary.json',json.dumps(result,indent=2)+'\n')
    import csv
    with (output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = '# Dropout analysis\n\n'+result['analysis']+'.\n\n'+result['location']+'.\n\n'+result['limitations']+'\n\n'
    text += f"All runs converged: {result['all_converged']}. Inspect each run's convergence history; capped runs are not converged.\n\n"
    text += '| Dropout | Split | Metric | Mean | SD | Paired Δ vs 0 | Δ SD |\n| ---: | --- | --- | ---: | ---: | ---: | ---: |\n'
    text += ''.join(f"| {r['dropout_rate']:g} | {r['split']} | {r['metric']} | {r['mean']:.6f} | {r['std']:.6f} | {r['paired_delta_mean']:+.6f} | {r['paired_delta_std']:.6f} |\n" for r in rows)
    text += '\n![Dropout effect](dropout_effect.png)\n'
    atomic_text(output/'report.md',text)
    return result


def dropout_sweep(args):
    if not re.fullmatch(r'yolov8[nslmx]-seg\.pt',Path(args.model).name):
        raise ValueError('Dropout sweep requires a fresh stock yolov8*-seg.pt base checkpoint')
    info, root, _ = load_prepared(args.dataset)
    validate_contract(info,root,require_version=True)
    for split in ('train','val','test'):
        if not any((root/split/'labels'/Path(row['image']).with_suffix('.txt').name).read_text().strip()
                   for row in info['prepared_images'] if row['role']==split):
            raise ValueError(f'Dropout sweep requires nonempty labelled {split} data')
    output = Path(args.sweep_output or ('data/derived/dropout_analysis/'+Path(args.output).stem)).resolve()
    if output == root or root in output.parents:
        raise ValueError('Sweep output must be outside the source dataset')
    output.mkdir(parents=True,exist_ok=False)
    manifest = {'source_dataset':str(Path(args.dataset).resolve()), 'source_sha256':digest(args.dataset),
                'dropout_rates':args.dropout_rates, 'seeds':list(range(args.seed,args.seed+args.dropout_runs)),
                'split_inventory':info['prepared_images']}
    atomic_text(output/'plan.json',json.dumps(manifest,indent=2)+'\n')
    rate_reports = []
    for rate in args.dropout_rates:
        rate_dir = output/f'rate-{rate}'
        rate_dir.mkdir()
        reports = []
        for index, seed in enumerate(manifest['seeds'],1):
            if digest(args.dataset) != manifest['source_sha256']:
                raise ValueError('Dataset manifest changed during dropout sweep')
            validate_contract(info,root,require_version=True)
            run_dir = rate_dir/f'run-{index:02d}'
            run_dir.mkdir()
            run = copy.copy(args)
            run.feature_dropout = rate
            report = train_and_evaluate(run,run_dir,args.dataset,seed)
            report['run'], report['dropout_rate'] = index, rate
            reports.append(report)
            atomic_text(rate_dir/'completed_runs.json',json.dumps(reports,indent=2)+'\n')
        rate_reports.append((rate,reports))
    summarize_sweep(rate_reports,output)
    print(f'Dropout report: {output / "report.md"}')
