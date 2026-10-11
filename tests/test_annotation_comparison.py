"""Comparison must reject changed holdouts and unmatched protocols."""
import argparse
import json
from pathlib import Path
import unittest

from annotation_cycle import compile_reviews,digest,load_prepared,write_info
from compare_annotation_models import compare
from tests import test_annotation_cycle as fixtures


class TestComparison(unittest.TestCase):
    setUp = fixtures.TestReviewCompilation.setUp
    tearDown = fixtures.TestReviewCompilation.tearDown
    make_dataset = fixtures.TestReviewCompilation.make_dataset
    approve = fixtures.TestReviewCompilation.approve

    def comparison_args(self):
        self.approve();compile_reviews(self.args)
        flags={}
        for role,path in (('baseline',self.base),('reviewed',Path(self.args.dataset_info))):
            info,_,_=load_prepared(path)
            settings={'model':'yolov8s-seg.pt','seed':42,'epochs':100,'imgsz':1024,'batch':4}
            metrics={f'metrics/{key}({suffix})':0.1 for key in ('precision','recall','mAP50','mAP50-95') for suffix in ('B','M')}
            common={'dataset_manifest_sha256':digest(path),'split_inventory':info['prepared_images'],
                    'checkpoint_sha256':role,'versions':{'torch':'fixture'},'effective_arguments':settings}
            train=self.root/(role+'-train.json');evaluation=self.root/(role+'-eval.json')
            train.write_text(json.dumps(dict(common,mode='training')))
            evaluation.write_text(json.dumps(dict(common,mode='evaluation',eval_split='test',metrics=metrics)))
            flags[role+'_dataset']=str(path)
            flags[role+'_report']=str(evaluation)
            flags[role+'_training_report']=str(train)
        return argparse.Namespace(**flags,output=str(self.root/'comparison.md'))

    def test_matched_comparison_records_zero_delta(self):
        args=self.comparison_args()
        report=compare(args)
        self.assertTrue(report['holdouts_identical'])
        self.assertTrue(all(row['delta']==0 for row in report['metrics']))
        self.assertTrue(Path(args.output).exists())

    def test_changed_test_ground_truth_rejects_comparison(self):
        args=self.comparison_args()
        info,root,config=load_prepared(args.reviewed_dataset)
        (root/'test/labels/other_heldout.txt').write_text('1 0 0 1 0 1 1\n')
        write_info(info,config,root,Path(args.reviewed_dataset),'next')
        with self.assertRaisesRegex(ValueError,'Held-out acquisitions changed'):compare(args)

    def test_unmatched_training_budget_rejected(self):
        args=self.comparison_args()
        p=Path(args.reviewed_training_report);report=json.loads(p.read_text())
        report['effective_arguments']['epochs']=1;p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'Protocol differs: epochs'):compare(args)

    def test_different_checkpoint_report_rejected(self):
        args=self.comparison_args()
        p=Path(args.reviewed_report);report=json.loads(p.read_text())
        report['checkpoint_sha256']='different';p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'reported trained checkpoint'):compare(args)


if __name__=='__main__':unittest.main()
