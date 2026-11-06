"""CV split/lineage safety, loss-controlled scheduling and report regressions."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from annotation_cycle import CLASSES, digest, load_prepared, validate_contract, write_info
from cross_validation import build_run_dataset, plan_runs, summarize_runs
from train_yolo import ConvergingSegmentationTrainer, LossPlateau, parse_args


class TestLossPlateau(unittest.TestCase):
    def controller(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
        return LossPlateau(optimizer, patience=0, factor=0.5, min_lr=0.025, stop_patience=2)

    def test_reduction_then_convergence_only_after_waiting_at_floor(self):
        controller = self.controller()
        for epoch in range(1, 6):
            self.assertEqual(controller.step(epoch, {'val/seg_loss':1.0}), epoch == 5)
        self.assertEqual([row['learning_rates'][0] for row in controller.history], [.1,.05,.025,.025,.025])

    def test_improvement_resets_convergence_wait(self):
        controller = self.controller()
        for epoch in range(1, 5):
            controller.step(epoch, {'val/seg_loss':1.0})
        self.assertFalse(controller.step(5, {'val/seg_loss':0.5}))
        self.assertEqual(controller.floor_epochs, 0)
        self.assertEqual(controller.best, 0.5)

    def test_uses_all_validation_losses_not_training_or_map(self):
        controller = self.controller()
        controller.step(1, {'val/box_loss':2, 'val/seg_loss':3, 'val/cls_loss':4,
                            'val/dfl_loss':5, 'train/seg_loss':100, 'metrics/mAP50(M)':.9})
        self.assertEqual(controller.history[0]['validation_loss'], 14)

    def test_missing_and_nonfinite_validation_fail(self):
        for metrics in ({}, {'val/seg_loss':float('nan')}, {'val/seg_loss':float('inf')}):
            with self.assertRaisesRegex(ValueError, 'finite validation'):
                self.controller().step(1, metrics)

    def test_trainer_does_not_step_during_minimum_100_batch_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = object.__new__(ConvergingSegmentationTrainer)
            trainer.train_loader = range(10)
            trainer.args = SimpleNamespace(warmup_epochs=3)
            trainer.epoch, trainer.epochs, trainer.stop = 9, 1000, False
            trainer.save_dir = tmp
            trainer.convergence_settings = {}
            trainer.loss_plateau = self.controller()
            trainer.optimizer = trainer.loss_plateau.optimizer
            trainer.scheduler = SimpleNamespace(base_lrs=[.1])
            with patch('ultralytics.models.yolo.segment.SegmentationTrainer.validate', return_value=({'val/seg_loss':1}, .5)):
                trainer.validate()
                self.assertEqual(trainer.loss_plateau.history, [])
                trainer.epoch = 10
                trainer.validate()
                self.assertEqual(len(trainer.loss_plateau.history), 1)

    def test_cli_defaults_and_invalid_combinations(self):
        args = parse_args(['--dataset','fixture.json','--cross-validate'])
        self.assertEqual((args.cv_runs, args.seed), (10,42))
        for flags in (['--cv-runs','1'], ['--lr-factor','1'], ['--min-lr','0'],
                      ['--cross-validate','--evaluate']):
            with self.assertRaises(SystemExit):
                parse_args(['--dataset','fixture.json']+flags)

    def test_next_epoch_does_not_undo_plateau_reductions(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=object.__new__(ConvergingSegmentationTrainer)
            trainer.optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.1)
            trainer.convergence_settings={'patience':0,'factor':.5,'min_lr':.025,'stop_patience':2}
            trainer.args=SimpleNamespace(warmup_epochs=0)
            trainer.train_loader,trainer.save_dir,trainer.epochs,trainer.stop=range(1),tmp,1000,False
            trainer._setup_scheduler()
            rates=[]
            with patch('ultralytics.models.yolo.segment.SegmentationTrainer.validate',return_value=({'val/seg_loss':1},.5)):
                for epoch in range(4):
                    trainer.epoch=epoch
                    trainer.optimizer.step(); trainer.scheduler.step()
                    rates.append(trainer.optimizer.param_groups[0]['lr'])
                    trainer.validate()
            self.assertEqual(rates,[.1,.1,.05,.025])


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent/'source'
        splits = {'train':['other_a__f0000.png','other_a__f0001.png','other_b.png','other_c.png'],
                  'val':['other_frozenval.png'], 'test':['other_frozentest.png','other_pool.png']}
        records = {}
        for split, names in splits.items():
            for kind in ('images','labels'):
                (self.root/split/kind).mkdir(parents=True)
            for name in names:
                Image.new('RGB', (64,64), color=(100,110,120)).save(self.root/split/'images'/name)
                label = '' if name in ('other_a__f0001.png','other_pool.png') else '0 0.2 0.2 0.8 0.2 0.8 0.8\n'
                (self.root/split/'labels'/Path(name).with_suffix('.txt').name).write_text(label)
                sample = name.split('__')[0][6:].removesuffix('.png')
                records[sample] = {'sample_id':sample, 'magnification':'other', 'source_format':'tiff',
                                   'sources':{}, 'annotation_path': None if sample=='pool' else '/fixture/'+sample+'.h5'}
        self.config = {'path':str(self.root), 'train':'train/images', 'val':'val/images',
                       'test':'test/images', 'names':dict(enumerate(CLASSES))}
        self.manifest = self.parent/'source.json'
        self.info = write_info({'records':list(records.values()), 'magnifications':['other']},
                               self.config, self.root, self.manifest, 'source')

    def plans(self):
        return plan_runs(self.info, self.root, 10, 42, .2)

    def test_repeatable_grouped_splits_and_seed_schedule(self):
        plans = self.plans()
        self.assertEqual(plans, self.plans())
        self.assertEqual([plan['seed'] for plan in plans], list(range(42,52)))
        for plan in plans:
            train = set(map(tuple, plan['train_acquisitions']))
            val = set(map(tuple, plan['val_acquisitions']))
            self.assertFalse(train & val)
            self.assertEqual(train | val, {('other','a'),('other','b'),('other','c')})
        self.assertNotEqual(plans, plan_runs(self.info,self.root,10,99,.2))

    def test_copy_contract_preserves_test_pool_and_groups_empty_frames(self):
        before = {str(path.relative_to(self.root)):digest(path) for path in self.root.rglob('*') if path.is_file()}
        plan = {'run':1, 'seed':42, 'train_acquisitions':[['other','b'],['other','c']],
                'val_acquisitions':[['other','a']]}
        path = build_run_dataset(self.info,self.root,self.config,plan,self.parent/'fold')
        info, root, _ = load_prepared(path)
        validate_contract(info,root,require_version=True)
        rows = {row['image']:row for row in info['prepared_images']}
        self.assertEqual(rows['other_a__f0001.png']['split'], 'val')
        self.assertEqual(rows['other_a__f0000.png']['split'], 'val')
        self.assertNotIn('other_frozenval.png', rows)
        self.assertEqual(rows['other_pool.png']['role'], 'pool')
        original = {row['image']:row for row in self.info['prepared_images']}
        self.assertEqual(rows['other_frozentest.png'], original['other_frozentest.png'])
        self.assertEqual(before, {str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()})

    def test_review_approval_survives_split_change(self):
        name = 'other_a__f0000.png'
        approval = {'status':'approved','reviewer':'fixture-only',
                    'image_sha256':digest(self.root/'train/images'/name),
                    'label_sha256':digest(self.root/'train/labels'/Path(name).with_suffix('.txt').name)}
        (self.root/'review_approvals.json').write_text(json.dumps({'train/'+name:approval}))
        self.info = write_info(self.info,self.config,self.root,self.manifest,'source')
        plan = {'run':1,'seed':42,'train_acquisitions':[['other','b'],['other','c']], 'val_acquisitions':[['other','a']]}
        path = build_run_dataset(self.info,self.root,self.config,plan,self.parent/'fold')
        info, root, _ = load_prepared(path)
        validate_contract(info,root,require_version=True)
        row = next(row for row in info['prepared_images'] if row['image']==name)
        self.assertEqual(row['annotation_origin'], 'human_reviewed')
        self.assertEqual(row['approval'], approval)

    def test_rejects_legacy_candidates_stale_labels_and_missing_test(self):
        invalid = copy.deepcopy(self.info); invalid['purpose'] = 'candidate_review'
        with self.assertRaisesRegex(ValueError, 'not ground truth'):
            plan_runs(invalid,self.root,10,42,.2)
        invalid = copy.deepcopy(self.info); invalid['schema_version'] = 1
        with self.assertRaises(ValueError):
            plan_runs(invalid,self.root,10,42,.2)
        (self.root/'test/labels/other_frozentest.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'Label inventory changed'):
            self.plans()
        self.info = write_info(self.info,self.config,self.root,self.manifest,'source')
        with self.assertRaisesRegex(ValueError, 'labelled frozen test'):
            self.plans()

    def test_partial_copy_is_not_published(self):
        with patch('cross_validation.shutil.copy2', side_effect=OSError('copy failed')):
            with self.assertRaises(OSError):
                build_run_dataset(self.info,self.root,self.config,self.plans()[0],self.parent/'fold')
        self.assertFalse((self.parent/'fold').exists())
        self.assertFalse((self.parent/'fold.json').exists())

    def test_frozen_acquisition_cannot_be_requested_as_training(self):
        plan=self.plans()[0]
        plan['train_acquisitions']=[['other','frozentest']]
        with self.assertRaisesRegex(ValueError,'acquisition leakage'):
            build_run_dataset(self.info,self.root,self.config,plan,self.parent/'fold')
        self.assertFalse((self.parent/'fold').exists())

    def test_summary_has_correct_sd_and_missing_classes_and_plots(self):
        reports = []
        for number, value in enumerate((.2,.4), 1):
            metrics = {f'metrics/{key}({kind})':value for kind in ('B','M')
                       for key in ('precision','recall','mAP50','mAP50-95')}
            evaluation = {'metrics':metrics, 'per_class_metrics':[
                {'class_id':i, 'mask':{'mAP50_95':value} if i==0 else None} for i in range(7)]}
            reports.append({'run':number,'training':{'convergence':{'converged':number==1}},
                            'val':evaluation,'test':evaluation})
        summary = summarize_runs(reports,self.parent)
        self.assertAlmostEqual(summary['metrics'][0]['mean'], .3)
        self.assertAlmostEqual(summary['metrics'][0]['std'], .2/(2**.5))
        self.assertIsNone(summary['per_class_mask_ap'][1]['mean'])
        self.assertFalse(summary['all_converged'])
        for name in ('summary.json','metrics.csv','metrics_error_bars.png','per_class_error_bars.png','report.md'):
            self.assertTrue((self.parent/name).is_file())


if __name__ == '__main__':
    unittest.main()
