#!/usr/bin/env python3
"""Compare matched fresh runs only after verifying identical labelled holdouts."""
import argparse
import json
from pathlib import Path

from annotation_cycle import atomic_text, load_prepared, validate_contract


def compare(args):
    baseline,baseline_root,_=load_prepared(args.baseline_dataset)
    reviewed,reviewed_root,_=load_prepared(args.reviewed_dataset)
    validate_contract(baseline,baseline_root,require_version=True)
    validate_contract(reviewed,reviewed_root,require_version=True)
    protected={tuple(row['raw_identity']) for row in baseline['prepared_images'] if row['role'] in ('val','test')}
    def heldout(info):
        return {row['image']:(row['split'],row['role'],row['image_sha256'],row['label_sha256'])
                for row in info['prepared_images'] if tuple(row['raw_identity']) in protected}
    if heldout(baseline)!=heldout(reviewed):raise ValueError('Held-out acquisitions changed; comparison is invalid')
    evaluations=[json.loads(Path(p).read_text()) for p in (args.baseline_report,args.reviewed_report)]
    training=[json.loads(Path(p).read_text()) for p in (args.baseline_training_report,args.reviewed_training_report)]
    from annotation_cycle import digest
    for data_path,info,evaluation,train in zip((args.baseline_dataset,args.reviewed_dataset),(baseline,reviewed),evaluations,training):
        if evaluation['mode']!='evaluation' or evaluation['eval_split']!='test':raise ValueError('Require labelled test evaluation reports')
        if evaluation['dataset_manifest_sha256']!=digest(data_path) or train['dataset_manifest_sha256']!=digest(data_path):
            raise ValueError('Report belongs to a different dataset version')
        if evaluation['checkpoint_sha256']!=train['checkpoint_sha256']:raise ValueError('Evaluation does not use the reported trained checkpoint')
        if evaluation['split_inventory']!=info['prepared_images']:raise ValueError('Evaluation inventory differs')
        if train['effective_arguments'].get('resume'):raise ValueError('Pilot requires fresh training, not resumed runs')
        if Path(train['effective_arguments']['model']).name!='yolov8s-seg.pt':raise ValueError('Pilot must initialize independently from yolov8s-seg.pt')
    keys=('epochs','imgsz','batch','seed','nbs','optimizer','lr0','lrf','momentum','weight_decay',
          'hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','flipud','fliplr','mosaic','mixup','copy_paste')
    mismatches=[key for key in keys if training[0]['effective_arguments'].get(key)!=training[1]['effective_arguments'].get(key)]
    mismatches += ['evaluation '+key for key in ('imgsz','batch','conf','iou','max_det','split')
                   if evaluations[0]['effective_arguments'].get(key)!=evaluations[1]['effective_arguments'].get(key)]
    if training[0]['versions']!=training[1]['versions']:mismatches.append('software versions')
    if training[0].get('initial_checkpoint_sha256')!=training[1].get('initial_checkpoint_sha256'):
        mismatches.append('pretrained initialization hash')
    if mismatches:raise ValueError('Protocol differs: '+', '.join(mismatches))
    metric_keys=('metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)',
                 'metrics/precision(M)','metrics/recall(M)','metrics/mAP50(M)','metrics/mAP50-95(M)')
    rows=[]
    for key in metric_keys:
        first=float(evaluations[0]['metrics'][key]);second=float(evaluations[1]['metrics'][key])
        rows.append({'metric':key,'baseline':first,'reviewed':second,'delta':second-first})
    result={'holdouts_identical':True,'baseline_dataset':str(Path(args.baseline_dataset).resolve()),
            'reviewed_dataset':str(Path(args.reviewed_dataset).resolve()),'metrics':rows,
            'per_class':{'baseline':evaluations[0].get('per_class_metrics'), 'reviewed':evaluations[1].get('per_class_metrics')},
            'protocol':{key:training[0]['effective_arguments'].get(key) for key in keys},
            'versions':{'baseline':training[0]['versions'],'reviewed':training[1]['versions']},
            'review_history':reviewed.get('review_history',[]),
            'limitations':['Small, single-field CZI test is not a generalization benchmark.',
                           'Audit identities do not establish biological reviewer expertise.']}
    output=Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Choose a fresh comparison output')
    text='# Matched annotation-cycle model comparison\n\nLabelled validation/test acquisitions are byte-identical.\n\n'
    text+='| Metric | Baseline | Reviewed | Delta |\n| --- | ---: | ---: | ---: |\n'
    text+=''.join(f"| {r['metric']} | {r['baseline']:.6f} | {r['reviewed']:.6f} | {r['delta']:+.6f} |\n" for r in rows)
    text+='\nThis is a bounded pilot, not evidence of broad generalization. Biological review and class support must be assessed separately.\n'
    text+='\n## Datasets, protocol, class support and review lineage\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n'
    atomic_text(output,text)
    atomic_text(output.with_suffix('.json'),json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('baseline-dataset','reviewed-dataset','baseline-report','reviewed-report',
                 'baseline-training-report','reviewed-training-report','output'):
        p.add_argument('--'+flag,required=True)
    args=p.parse_args()
    compare(args)
    print(f'Comparison: {args.output}')


if __name__=='__main__':main()
