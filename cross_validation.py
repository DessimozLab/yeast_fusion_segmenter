"""Repeated grouped holdout CV without promoting frozen acquisitions or model candidates."""
import copy
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics

from annotation_cycle import (CLASSES, atomic_text, digest, identity, image_entries,
                              load_prepared, read_registry, staged_output,
                              validate_contract, write_info)


def plan_runs(info, root, runs, seed, fraction):
    validate_contract(info, root, require_version=True)
    if runs < 2 or not 0 < fraction < 1:
        raise ValueError('CV needs at least two runs and 0 < validation fraction < 1')
    entries = image_entries(info, root)
    groups = sorted({identity(name) for name, (split, image, _) in entries.items()
                     if split == 'train' and (root/split/'labels'/image.with_suffix('.txt').name).read_text().strip()})
    if len(groups) < 2:
        raise ValueError('CV requires at least two annotated training acquisitions')
    if not any(row['role'] == 'test' and
               (root/'test'/'labels'/Path(row['image']).with_suffix('.txt').name).read_text().strip()
               for row in info['prepared_images']):
        raise ValueError('CV requires a nonempty labelled frozen test set')
    count = min(len(groups)-1, max(1, math.ceil(len(groups)*fraction)))
    plans = []
    for index in range(runs):
        shuffled = list(groups)
        random.Random(seed+index).shuffle(shuffled)
        plans.append({'run': index+1, 'seed': seed+index,
                      'train_acquisitions': [list(key) for key in sorted(shuffled[count:])],
                      'val_acquisitions': [list(key) for key in sorted(shuffled[:count])]})
    return plans


def build_run_dataset(source, root, config, plan, output):
    """Copy derived data only, retaining approval hashes and the source raw contract."""
    output = Path(output).resolve()
    info_path = output.parent/(output.name+'.json')
    if output.exists() or info_path.exists():
        raise ValueError('CV dataset output already exists')
    validate_contract(source, root, require_version=True)
    train = {tuple(key) for key in plan['train_acquisitions']}
    val = {tuple(key) for key in plan['val_acquisitions']}
    allowed = {tuple(row['raw_identity']) for row in source['prepared_images'] if row['role']=='train'}
    if not train or not val or train & val or not (train | val) <= allowed:
        raise ValueError('CV acquisition leakage or invalid training/validation groups')
    inventory = {row['image']:row for row in source['prepared_images']}
    approvals, new_approvals = read_registry(root), {}
    with staged_output(output, info_path) as stage:
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                (stage/split/kind).mkdir(parents=True)
        for name, (split, image, _) in image_entries(source, root).items():
            key = identity(name)
            if split == 'train' and key in train | val:
                target = 'val' if key in val else 'train'
            elif split == 'test':
                target = 'test'
            else:
                continue  # Frozen original validation acquisitions never enter CV training or selection.
            shutil.copy2(image, stage/target/'images'/name)
            shutil.copy2(root/split/'labels'/image.with_suffix('.txt').name,
                         stage/target/'labels'/image.with_suffix('.txt').name)
            if (digest(stage/target/'images'/name) != inventory[name]['image_sha256'] or
                    digest(stage/target/'labels'/image.with_suffix('.txt').name) != inventory[name]['label_sha256']):
                raise ValueError('Copied CV data differs from the source contract')
            if f'{split}/{name}' in approvals:
                new_approvals[f'{target}/{name}'] = approvals[f'{split}/{name}']
        atomic_text(stage/'review_approvals.json', json.dumps(new_approvals, indent=2)+'\n')
        payload = copy.deepcopy(source)
        payload['cross_validation'] = plan
        payload['purpose'] = 'training'
        written = write_info(payload, config, stage, stage/'.publication.json', output.name,
                             published_root=output, published_info=info_path)
        validate_contract(written, stage)
        # Detect a changed source before publishing freshly hashed copies.
        validate_contract(source, root, require_version=True)
    return info_path


def summarize_runs(reports, output, method='repeated acquisition-grouped holdout within original training split'):
    """Plot sample SD across repeated runs, not biological confidence intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    rows = []
    for split in ('val', 'test'):
        keys = sorted({key for report in reports for key in report[split]['metrics'] if key.startswith('metrics/')})
        for key in keys:
            values = [float(report[split]['metrics'][key]) for report in reports]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'Nonfinite CV metric: {split}/{key}')
            rows.append({'split': split, 'metric': key, 'n': len(values),
                         'mean': statistics.mean(values), 'std': statistics.stdev(values)})
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split','metric','n','mean','std'])
        writer.writeheader(); writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, suffix, title in zip(axes, ('(B)', '(M)'), ('Boxes', 'Masks')):
        keys = [f'metrics/{metric}{suffix}' for metric in ('precision','recall','mAP50','mAP50-95')]
        for split, shift in (('val', -0.08), ('test', 0.08)):
            selected = [next(row for row in rows if row['split']==split and row['metric']==key) for key in keys]
            ax.errorbar([i+shift for i in range(4)], [row['mean'] for row in selected],
                        yerr=[row['std'] for row in selected], fmt='o', capsize=4, label=split)
        ax.set_xticks(range(4), ['Precision','Recall','AP50','AP50–95'])
        ax.set_title(title); ax.set_ylabel('Mean ± sample SD'); ax.legend()
    figure.suptitle(f'{len(reports)} runs: mean ± SD (not confidence intervals)')
    figure.savefig(output/'metrics_error_bars.png', dpi=160)
    plt.close(figure)
    per_class = []
    figure, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for split, shift in (('val', -0.08), ('test', 0.08)):
        for class_id, name in enumerate(CLASSES):
            values = [row['mask']['mAP50_95'] for report in reports
                      for row in report[split]['per_class_metrics'] if row['class_id']==class_id and row['mask'] is not None]
            row = {'split':split, 'class_id':class_id, 'class':name, 'n':len(values),
                   'mean':statistics.mean(values) if values else None,
                   'std':statistics.stdev(values) if len(values)>1 else None}
            per_class.append(row)
            if values:
                ax.errorbar(class_id+shift, row['mean'], yerr=row['std'], fmt='o', capsize=4,
                            color='C0' if split=='val' else 'C1')
    ax.set_xticks(range(7), CLASSES); ax.set_ylabel('Mask AP50–95 mean ± sample SD')
    ax.set_title('Validation (blue), test (orange); unsupported classes omitted')
    figure.savefig(output/'per_class_error_bars.png', dpi=160); plt.close(figure)
    result = {'method':method,
              'error_bars':'sample standard deviation; correlated repeats, not confidence intervals',
              'metrics':rows, 'per_class_mask_ap':per_class, 'runs':reports,
              'all_converged':all(report['training']['convergence']['converged'] for report in reports)}
    atomic_text(output/'summary.json', json.dumps(result, indent=2)+'\n')
    text = '# Repeated training report\n\n'+result['method']+'.\n\n'
    text += 'Error bars are sample SD, not confidence intervals or independent biological replicates.\n\n'
    text += f"All runs converged: {result['all_converged']}. Epoch-capped runs are not evidence of convergence.\n\n"
    text += '| Split | Metric | Mean | SD | Runs |\n| --- | --- | ---: | ---: | ---: |\n'
    text += ''.join(f"| {r['split']} | {r['metric']} | {r['mean']:.6f} | {r['std']:.6f} | {r['n']} |\n" for r in rows)
    text += '\n![Aggregate metrics](metrics_error_bars.png)\n\n![Phenotype metrics](per_class_error_bars.png)\n'
    atomic_text(output/'report.md', text)
    return result


def train_and_evaluate(args, run_dir, dataset, seed):
    """Shared protocol: fresh training, then independent labelled validation/test evaluation."""
    from train_yolo import train_model, evaluate_model
    run = copy.copy(args)
    run.dataset, run.data, run.seed = str(dataset), None, seed
    run.annotated_only, run.cross_validate, run.evaluate = True, False, False
    run.output, run.run_dir = str(run_dir/'checkpoint.pt'), str(run_dir/'training')
    train_model(run)
    report = {'seed':seed, 'dataset':str(dataset),
              'training':json.loads((run_dir/'training/checkpoint/protocol_metrics.json').read_text())}
    for split in ('val', 'test'):
        evaluation = copy.copy(run)
        evaluation.evaluate, evaluation.eval_split = True, split
        evaluation.model, evaluation.run_dir = run.output, str(run_dir/'evaluation'/split)
        evaluate_model(evaluation)
        report[split] = json.loads((run_dir/'evaluation'/split/'checkpoint/protocol_metrics.json').read_text())
    return report


def cross_validate(args):
    if not re.fullmatch(r'yolov8[nslmx]-seg\.pt', Path(args.model).name):
        raise ValueError('CV requires fresh initialization from a stock yolov8*-seg.pt base, not an experiment checkpoint')
    info, root, config = load_prepared(args.dataset)
    plans = plan_runs(info, root, args.cv_runs, args.seed, args.cv_val_fraction)
    output = Path(args.cv_output or ('data/derived/cross_validation/'+Path(args.output).stem)).resolve()
    if output == root or root in output.parents:
        raise ValueError('CV outputs must be outside the source dataset')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'source_dataset': str(Path(args.dataset).resolve()), 'source_sha256':digest(args.dataset),
                'base_seed':args.seed, 'runs':plans, 'frozen_holdouts':[
                    row for row in info['prepared_images'] if row['role'] in ('val', 'test')]}
    atomic_text(output/'plan.json', json.dumps(manifest, indent=2)+'\n')
    reports = []
    for plan in plans:
        if digest(args.dataset) != manifest['source_sha256']:
            raise ValueError('Source dataset manifest changed during CV')
        run_dir = output/f"run-{plan['run']:02d}"
        run_dir.mkdir()
        dataset = build_run_dataset(info, root, config, plan, run_dir/'dataset')
        report = train_and_evaluate(args, run_dir, dataset, plan['seed'])
        report['run'] = plan['run']
        reports.append(report)
        atomic_text(output/'completed_runs.json', json.dumps(reports, indent=2)+'\n')
    summarize_runs(reports, output, method='Repeated acquisition-grouped holdout within original training split; '
                   'original validation acquisitions excluded and frozen test reused')
    print(f'Cross-validation report: {output / "report.md"}')
