"""Dataset publication and real HTTP review safety (no biological approval)."""
import argparse
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from annotation_cycle import (compile_reviews, digest, load_prepared, migrate_dataset,
                              validate_contract, write_info)
from manual_yolo_annotation_app import make_handler
from tests import test_annotation_cycle as fixtures


class TestAnnotationSafety(unittest.TestCase):
    setUp = fixtures.TestReviewCompilation.setUp
    tearDown = fixtures.TestReviewCompilation.tearDown
    make_dataset = fixtures.TestReviewCompilation.make_dataset
    approve = fixtures.TestReviewCompilation.approve

    def test_candidate_yaml_cannot_bypass_training_guard(self):
        from train_yolo import resolve_data_path
        args=argparse.Namespace(data=str(self.root/'review/dataset.yaml'),dataset=None,annotated_only=False)
        with self.assertRaisesRegex(ValueError,'not ground truth'):
            resolve_data_path(args)

    def test_compiled_label_changes_rejected_by_training(self):
        self.approve(); compile_reviews(self.args)
        (self.root/'next/train/labels/other_pool.txt').write_text('1 0 0 1 0 1 1\n')
        info, root, _=load_prepared(self.args.dataset_info)
        with self.assertRaisesRegex(ValueError,'Label inventory changed'):
            validate_contract(info,root)

    def test_compiled_approval_changes_rejected(self):
        self.approve(); compile_reviews(self.args)
        p=self.root/'next/review_approvals.json'
        p.write_text('{}')
        info, root, _=load_prepared(self.args.dataset_info)
        with self.assertRaisesRegex(ValueError,'Stale approval'):
            validate_contract(info,root)

    def test_failed_compilation_leaves_no_published_output(self):
        self.approve()
        with patch('annotation_cycle.shutil.copy2',side_effect=OSError('injected copy failure')):
            with self.assertRaisesRegex(OSError,'injected'):
                compile_reviews(self.args)
        self.assertFalse((self.root/'next').exists())
        self.assertFalse((self.root/'next.json').exists())
        self.assertFalse(list(self.root.glob('.next-*')))

    def test_successive_cycles_retain_review_and_teacher_lineage(self):
        info, root, config=load_prepared(self.review)
        info['candidate_source']={'model_sha256':'teacher-fixture'}
        write_info(info,config,root,self.review,'review')
        self.approve(); compile_reviews(self.args)
        second=argparse.Namespace(dataset=str(self.base),reviewed_dataset=self.args.dataset_info,
             output_dir=str(self.root/'third'),dataset_info=str(self.root/'third.json'),name='third')
        compile_reviews(second)
        info, root, _=load_prepared(second.dataset_info)
        row=next(row for row in info['prepared_images'] if row['image']=='other_pool.png')
        self.assertEqual(row['candidate_source']['model_sha256'],'teacher-fixture')
        self.assertEqual(row['annotation_origin'],'human_reviewed')
        self.assertTrue(info['inherited_review_history'])
        validate_contract(info,root)

    def test_legacy_migration_is_explicit_and_cannot_refresh_v2_hashes(self):
        payload=json.loads(self.base.read_text());payload['schema_version']=1
        self.base.write_text(json.dumps(payload))
        (self.root/'base/dataset-contract.json').unlink()
        args=argparse.Namespace(dataset=str(self.base),acknowledge_legacy=False)
        with self.assertRaisesRegex(ValueError,'acknowledge'):
            migrate_dataset(args)
        args.acknowledge_legacy=True
        migrate_dataset(args)
        with self.assertRaisesRegex(ValueError,'already version 2'):
            migrate_dataset(args)

    def test_empty_sibling_frame_leakage_is_rejected(self):
        info, root, config=load_prepared(self.base)
        import shutil
        shutil.copy2(root/'train/images/other_old.png',root/'test/images/other_old__f0001.png')
        (root/'test/labels/other_old__f0001.txt').write_text('')
        info=write_info(info,config,root,self.base,'base')
        with self.assertRaisesRegex(ValueError,'Acquisition leakage'):
            validate_contract(info,root)

    def test_deferred_review_cannot_be_compiled(self):
        self.approve()
        p=self.root/'review/review_approvals.json'
        states=json.loads(p.read_text());states['test/other_pool.png']['status']='deferred'
        p.write_text(json.dumps(states))
        with self.assertRaisesRegex(ValueError,'No approved'):
            compile_reviews(self.args)


class TestReviewHTTP(unittest.TestCase):
    make_dataset = fixtures.TestReviewCompilation.make_dataset
    def setUp(self):
        fixtures.TestReviewCompilation.setUp(self)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.root/'review',
                                       list(load_prepared(self.review)[2]['names'].values()),'test'))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        fixtures.TestReviewCompilation.tearDown(self)

    def request(self,route,value=None):
        req=Request(self.url+route,data=json.dumps(value).encode() if value is not None else None,
                    headers={'Content-Type':'application/json'})
        try:
            with urlopen(req) as response:return response.status,json.load(response)
        except HTTPError as error:return error.code,error.read().decode()

    def snapshot(self):
        return self.request('/api/labels/other_pool.png')[1]

    def test_stale_browser_tab_cannot_overwrite_labels(self):
        revision=self.snapshot()
        newer=dict(revision,labels=[{'class_id':1,'points':[[0,0],[1,0],[1,1]]}])
        self.assertEqual(self.request('/api/labels/other_pool.png',newer)[0],200)
        self.assertEqual(self.request('/api/labels/other_pool.png',revision)[0],409)
        self.assertEqual(self.snapshot()['labels'][0]['class_id'],1)

    def test_save_invalidates_approval_and_defer_is_visible(self):
        revision=self.snapshot()
        self.assertEqual(self.request('/api/approve/other_pool.png',dict(revision,reviewer='automated-test'))[0],200)
        self.assertTrue(self.snapshot()['review']['valid'])
        self.request('/api/labels/other_pool.png',self.snapshot())
        self.assertEqual(self.snapshot()['review']['status'],'changed_since_approval')
        self.assertFalse(self.snapshot()['review']['valid'])
        self.request('/api/defer/other_pool.png',dict(self.snapshot(),reviewer='automated-test',notes='fixture ambiguity'))
        self.assertEqual(self.snapshot()['review']['status'],'deferred')

    def test_compiled_dataset_is_read_only(self):
        handler=make_handler(self.root/'base',list(load_prepared(self.base)[2]['names'].values()),'test')
        self.server.RequestHandlerClass=handler
        revision=self.snapshot()
        self.assertFalse(revision['editable'])
        self.assertEqual(self.request('/api/labels/other_pool.png',revision)[0],403)

    def test_heldout_is_read_only_even_in_candidate_review(self):
        info, root, config=load_prepared(self.review)
        info['prepared_images'][0]['role']='test'
        write_info(info,config,root,self.review,'review')
        self.server.RequestHandlerClass=make_handler(root,list(config['names'].values()),'test')
        self.assertFalse(self.snapshot()['editable'])
        self.assertEqual(self.request('/api/approve/other_pool.png',dict(self.snapshot(),reviewer='automated-test'))[0],403)

    def test_approval_requires_current_revision(self):
        revision=self.snapshot();revision['label_sha256']='stale'
        self.assertEqual(self.request('/api/approve/other_pool.png',dict(revision,reviewer='automated-test'))[0],409)


if __name__=='__main__':unittest.main()
