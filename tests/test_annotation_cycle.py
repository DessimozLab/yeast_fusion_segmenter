"""Regression tests for approval, frozen holdouts and review promotion."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import random
from types import SimpleNamespace
import numpy as np

from PIL import Image
import yaml

from annotation_cycle import CLASSES, compile_reviews, digest, load_prepared, write_info


class TestReviewCompilation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.make_dataset('base', {'train': ['other_old.png'],
                                             'val': ['other_val.png'],
                                             'test': ['other_pool.png', 'other_heldout.png']})
        self.review = self.make_dataset('review', {'train': [], 'val': [], 'test': ['other_pool.png']})
        self.args = argparse.Namespace(dataset=str(self.base), reviewed_dataset=str(self.review),
            output_dir=str(self.root/'next'), dataset_info=str(self.root/'next.json'), name='next')

    def tearDown(self):
        self.temp.cleanup()

    def make_dataset(self, name, splits):
        root = self.root/name
        records = []
        for split, names in splits.items():
            for kind in ('images','labels'):
                (root/split/kind).mkdir(parents=True)
            for filename in names:
                Image.new('RGB',(16,16)).save(root/split/'images'/filename)
                label = '' if filename == 'other_pool.png' and name == 'base' else '0 0.1 0.1 0.5 0.1 0.5 0.5\n'
                (root/split/'labels'/Path(filename).with_suffix('.txt').name).write_text(label)
                sample = filename[6:-4]
                records.append({'sample_id':sample, 'magnification':'other', 'source_format':'tiff',
                    'sources':{'bf':f'/raw/{sample}__bf.tiff'},
                    'annotation_path': None if sample=='pool' else f'/raw/{sample}__mask.h5'})
        (root/'dataset.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'train/images',
            'val':'val/images','test':'test/images','names':dict(enumerate(CLASSES))}))
        info = self.root/f'{name}.json'
        info.write_text(json.dumps({'schema_version':1,'name':name,'magnifications':['other'],
            'records':records,'yolo_data':str(root/'dataset.yaml')}))
        payload=json.loads(info.read_text())
        if name.startswith('review'):
            payload['purpose']='candidate_review'
        write_info(payload,yaml.safe_load((root/'dataset.yaml').read_text()),root,info,name)
        return info

    def approve(self, name='other_pool.png'):
        review_root = self.root/'review'
        image = review_root/'test'/'images'/name
        label = review_root/'test'/'labels'/Path(name).with_suffix('.txt').name
        (review_root/'review_approvals.json').write_text(json.dumps({f'test/{name}':
            {'reviewer':'test-only','image_sha256':digest(image),'label_sha256':digest(label)}}))

    def test_unapproved_candidates_rejected(self):
        with self.assertRaisesRegex(ValueError,'No approved'):
            compile_reviews(self.args)
        self.assertFalse((self.root/'next').exists())

    def test_review_dataset_cannot_be_used_for_training_or_evaluation(self):
        from train_yolo import resolve_data_path
        info=json.loads(self.review.read_text())
        info['purpose']='candidate_review'
        self.review.write_text(json.dumps(info))
        args=argparse.Namespace(data=None,dataset=str(self.review),annotated_only=True)
        with self.assertRaisesRegex(ValueError,'not ground truth'):
            resolve_data_path(args)

    def test_empty_review_cannot_create_biological_negatives(self):
        (self.root/'review/test/labels/other_pool.txt').write_text('')
        self.approve()
        with self.assertRaisesRegex(ValueError,'Empty reviewed mask'):
            compile_reviews(self.args)

    def test_promotes_reviewed_pool_preserves_holdouts_and_raw_provenance(self):
        self.approve()
        compile_reviews(self.args)
        info, root, _ = load_prepared(self.args.dataset_info)
        self.assertTrue((root/'train/images/other_pool.png').exists())
        self.assertFalse((root/'test/images/other_pool.png').exists())
        for split, name in [('val','other_val'),('test','other_heldout')]:
            self.assertEqual(digest(root/split/'labels'/f'{name}.txt'),
                             digest(self.root/'base'/split/'labels'/f'{name}.txt'))
        self.assertEqual(info['records'], json.loads(self.base.read_text())['records'])
        approvals=json.loads((root/'review_approvals.json').read_text())
        self.assertIn('train/other_pool.png',approvals)
        self.assertEqual(len(info['prepared_images']),4)

    def test_changed_label_invalidates_approval(self):
        self.approve()
        (self.root/'review/test/labels/other_pool.txt').write_text('1 0 0 1 0 1 1\n')
        with self.assertRaisesRegex(ValueError,'Stale approval'):
            compile_reviews(self.args)

    def test_heldout_prediction_cannot_be_promoted(self):
        self.review = self.make_dataset('review-heldout', {'train':[], 'val':[], 'test':['other_heldout.png']})
        root = self.root/'review-heldout'
        (root/'review_approvals.json').write_text(json.dumps({'test/other_heldout.png':
            {'reviewer':'test-only','image_sha256':digest(root/'test/images/other_heldout.png'),
             'label_sha256':digest(root/'test/labels/other_heldout.txt')}}))
        self.args.reviewed_dataset = str(self.review)
        with self.assertRaisesRegex(ValueError,'Held-out'):
            compile_reviews(self.args)

    def test_fractional_splits_keep_sibling_stack_frames_together(self):
        from prepare_yolo_data import split_dataset
        root = self.root/'stack'
        for split in ('train','val','test'):
            for kind in ('images','labels'):
                (root/split/kind).mkdir(parents=True)
        for sample in ('alpha','beta','gamma','delta'):
            for frame in range(3):
                stem=f'other_{sample}__f{frame:04d}'
                Image.new('RGB',(16,16)).save(root/'train/images'/f'{stem}.png')
                (root/'train/labels'/f'{stem}.txt').write_text('0 0 0 1 0 1 1\n')
        random.seed(42)
        split_dataset(str(root),0.25,0.25)
        locations={}
        for split in ('train','val','test'):
            for image in (root/split/'images').glob('*.png'):
                locations.setdefault(image.stem.split('__f')[0],set()).add(split)
        self.assertTrue(all(len(splits)==1 for splits in locations.values()))

    def test_csv_statistics_use_mask_interior_and_bf_gfp_rfp_order(self):
        from batch_predict import predict_and_collect
        image=np.zeros((5,5,3),dtype=np.uint8)
        image[2,2]=[10,20,30]
        path=self.root/'statistics.png'
        Image.fromarray(image).save(path)
        class Tensor:
            def __init__(self,array): self.array=np.asarray(array)
            def cpu(self): return self
            def numpy(self): return self.array
        class Boxes:
            cls=[0]
            conf=Tensor([.9])
            xyxy=Tensor([[0,0,4,4]])
            def __len__(self): return 1
        result=SimpleNamespace(names=dict(enumerate(CLASSES)),boxes=Boxes(),
            masks=[SimpleNamespace(xy=[np.array([[0,0],[4,0],[4,4],[0,4]])])])
        frame=predict_and_collect(lambda *a,**kw:[result],str(path),str(self.root/'statistics.csv'))
        self.assertAlmostEqual(frame.iloc[0]['bf_mean'],.4)
        self.assertAlmostEqual(frame.iloc[0]['gfp_mean'],.8)
        self.assertAlmostEqual(frame.iloc[0]['rfp_mean'],1.2)


if __name__ == '__main__':
    unittest.main()
