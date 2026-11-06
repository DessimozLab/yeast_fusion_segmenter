"""Opt-in real CPU train/eval orchestration on a synthetic software fixture, not biology."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from tests import test_cross_validation as fixtures


@unittest.skipUnless(os.environ.get('YFS_RUN_TRAINING_SMOKE')=='1', 'opt-in CPU training smoke test')
class TestTrainingAnalysisSmoke(unittest.TestCase):
    def test_cv_and_dropout_cli_end_to_end(self):
        fixture=fixtures.TestCrossValidation()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root=fixture.parent
        hyp=root/'hyp.yaml'
        hyp.write_text('lr0: 0.001\nnbs: 2\nwarmup_epochs: 0\nmosaic: 0\nclose_mosaic: 0\n')
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        common=[sys.executable,'train_yolo.py','--dataset',str(fixture.manifest),
                '--model','yolov8n-seg.pt','--epochs','2','--batch-size','2','--img-size','64',
                '--device','cpu','--workers','0','--seed','42','--hyp',str(hyp),
                '--lr-patience','0','--min-lr','0.001','--convergence-patience','1']
        for name,flags in (('cv',['--cross-validate','--cv-runs','2','--cv-output',str(root/'cv')]),
                           ('sweep',['--dropout-sweep','--dropout-rates','0','0.5','--dropout-runs','2',
                                     '--sweep-output',str(root/'sweep')])):
            with (root/(name+'.log')).open('w') as log:
                process=subprocess.run(common+flags,cwd=str(Path(__file__).resolve().parents[1]),
                                       env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if process.returncode:
                self.fail((root/(name+'.log')).read_text()[-18000:])
            summary=json.loads((root/name/'summary.json').read_text())
            self.assertTrue(summary['metrics'])
            self.assertTrue((root/name/'report.md').is_file())
        cv=json.loads((root/'cv/summary.json').read_text())
        for report in cv['runs']:
            self.assertEqual(len(report['training']['convergence']['history']),2)
            self.assertIn('per_class_metrics',report['test'])
        sweep=json.loads((root/'sweep/summary.json').read_text())
        for rate in sweep['rates']:
            for report in rate['summary']['runs']:
                self.assertEqual(report['training']['feature_dropout']['rate'],rate['dropout_rate'])


if __name__=='__main__':
    unittest.main()
