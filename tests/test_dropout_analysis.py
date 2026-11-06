"""Dropout must affect segmentation training, not evaluation or serialized inference."""
import copy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import torch
from ultralytics import YOLO

from dropout_analysis import add_feature_dropout, summarize_sweep
from train_yolo import parse_args


class TestFeatureDropout(unittest.TestCase):
    def test_real_backbone_training_eval_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        model = YOLO('yolov8n-seg.yaml').model
        block = model.model[len(model.yaml['backbone'])-1]
        original = copy.deepcopy(block).eval()
        add_feature_dropout(model, .5)
        features = torch.randn(2, block.cv1.conv.in_channels, 4, 4)
        block.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(block(features),original(features)))
        # Test the actual inserted output activation without BatchNorm running-stat changes.
        activation = block.cv2.act.train()
        values = torch.ones(4,128,4,4)
        torch.manual_seed(42); first = activation(values)
        torch.manual_seed(43); second = activation(values)
        self.assertFalse(torch.equal(first,second))
        self.assertTrue((first==0).any())
        activation.eval()
        self.assertTrue(torch.equal(activation(values),activation(values)))
        buffer = BytesIO()
        torch.save(model.eval(),buffer); buffer.seek(0)
        loaded = torch.load(buffer,weights_only=False)
        loaded_block = loaded.model[len(loaded.yaml['backbone'])-1]
        with torch.no_grad():
            self.assertTrue(torch.equal(loaded_block(features),original(features)))
        with self.assertRaisesRegex(ValueError,'already installed'):
            add_feature_dropout(model,.2)

    def test_zero_dropout_is_identity_in_training(self):
        model = YOLO('yolov8n-seg.yaml').model
        block = model.model[len(model.yaml['backbone'])-1]
        activation = copy.deepcopy(block.cv2.act)
        add_feature_dropout(model,0)
        values = torch.randn(2,3,4,4)
        self.assertTrue(torch.equal(activation(values),block.cv2.act(values)))

    def test_cli_keeps_sweep_separate_and_checks_probabilities(self):
        args = parse_args(['--dataset','source.json','--dropout-sweep'])
        self.assertEqual(args.dropout_runs,3)
        self.assertIn(0,args.dropout_rates)
        for flags in (['--cross-validate'],['--evaluate'],['--dropout-rates','.2','.5'],
                      ['--dropout-rates','0','1'],['--dropout-runs','1'],['--feature-dropout','nan']):
            with self.assertRaises(SystemExit):
                parse_args(['--dataset','source.json','--dropout-sweep']+flags)

    def test_sweep_reports_paired_deltas_against_matching_seeds(self):
        def reports(rate):
            rows=[]
            for seed, value in ((42,.2),(43,.4)):
                metrics = {f'metrics/{key}({kind})':value+rate for kind in ('B','M')
                           for key in ('precision','recall','mAP50','mAP50-95')}
                evaluation = {'metrics':metrics,'per_class_metrics':[
                    {'class_id':i,'mask':{'mAP50_95':value+rate} if i==0 else None} for i in range(7)]}
                rows.append({'seed':seed,'training':{'convergence':{'converged':True}},
                             'val':evaluation,'test':evaluation})
            return rows
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rate in (0.0,.2):
                (root/f'rate-{rate}').mkdir()
            result=summarize_sweep([(0.0,reports(0)),(.2,list(reversed(reports(.2))))],root)
            changed=next(row for row in result['metrics'] if row['dropout_rate']==.2)
            self.assertAlmostEqual(changed['paired_delta_mean'],.2)
            self.assertAlmostEqual(changed['paired_delta_std'],0)
            self.assertTrue((root/'dropout_effect.png').is_file())
            self.assertTrue(json.loads((root/'summary.json').read_text())['rates'])


if __name__ == '__main__':
    unittest.main()
