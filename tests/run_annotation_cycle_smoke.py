#!/usr/bin/env python3
"""Repeat the real TIFF/CZI→CUDA→browser→retraining loop; never human approval.

Run from the project environment. Requires a fresh artifact directory and an
installed Playwright CLI/wrapper. All source data are copied into that directory.
"""
import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from urllib.request import Request, urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from annotation_cycle import atomic_text, digest, load_prepared
from manual_yolo_annotation_app import make_handler, parse_label


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--browser-cli',required=True,help='Installed playwright-cli or wrapper executable')
    parser.add_argument('--browser-config',help='Optional Playwright CLI launch configuration')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=Path(args.output_dir).resolve()
    if root.exists():parser.error('Choose a new output directory; smoke artifacts are never overwritten')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MPLCONFIGDIR='/tmp/yeast-mpl',PYTHONPATH=str(repo))
    commands=[]
    def run(command,name):
        command=[str(x) for x in command]
        print(shlex.join(command),flush=True)
        commands.append(command)
        log=root/(name+'.log')
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=str(repo),env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see {log}')
    # Fixture setup owns creation of the exact fresh output directory.
    subprocess.run([sys.executable,str(repo/'tests/setup_annotation_cycle_smoke.py'),str(root)],
                   cwd=str(repo),env=env,check=True)
    raw_hashes={str(p.relative_to(root)):digest(p) for p in (root/'raw').rglob('*') if p.is_file()}
    py=[sys.executable]
    run(py+['build_dataset.py','--name','loop-v1','--raw-root',root/'raw',
        '--output-dir',root/'v1','--dataset-info',root/'v1.json',
        '--val-samples','p1-1g2-09','--test-samples','p1-1e3-13','--val-split','0','--test-split','0'],'build')
    for version in ('v1',):
        run(py+['annotation_cycle.py','qc','--dataset',root/(version+'.json'),'--output',root/('qc-'+version+'.json')],'qc-'+version)
    def train(version):
        run(py+['train_yolo.py','--dataset',root/(version+'.json'),'--annotated-only',
            '--model','yolov8s-seg.pt','--notebook-protocol','--epochs','1','--img-size','320',
            '--batch-size','2','--workers','0','--seed','42','--device','0',
            '--run-dir',root/'training','--output',root/('model-'+version+'.pt')],'train-'+version)
    def export(version,name):
        run(py+['annotation_cycle.py','export','--dataset',root/(version+'.json'),
            '--model',root/('model-'+version+'.pt'),'--split','all','--confidence','0.001',
            '--imgsz','320','--device','0','--name',name,'--output-dir',root/name,
            '--dataset-info',root/(name+'.json')],name)
    train('v1');export('v1','review')
    info,review_root,config=load_prepared(root/'review.json')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(review_root,list(config['names'].values()),'train'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    name='40x_p1-3b9-04.png'
    def snapshot():
        with urlopen(url+'/api/labels/'+name) as response:return json.load(response)
    def save(segments):
        payload=dict(snapshot(),labels=segments)
        req=Request(url+'/api/labels/'+name,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req) as response:json.load(response)
    cli=[args.browser_cli,'--session','loop-'+str(os.getpid())]
    try:
        save([])
        launch=cli+['open',url]
        if args.browser_config:launch+=['--config',args.browser_config]
        run(launch,'browser-open');run(cli+['snapshot'],'browser-snapshot')
        # Scripted canvas interactions, not direct calls to internal app functions.
        code="""async page => {
          await page.locator('#images').selectOption('40x_p1-3b9-04.png');
          await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('40x_p1-3b9-04.png —'));
          await page.locator('#cls').selectOption('6');
          const c=page.locator('#c');
          await c.click({position:{x:30,y:30}});await c.click({position:{x:90,y:30}});await c.click({position:{x:90,y:90}});
          await page.getByRole('button',{name:'Finish polygon',exact:true}).click();
          await page.getByRole('button',{name:'Save labels',exact:true}).click();
          await page.waitForFunction(() => document.querySelector('#status').textContent.includes('saved 1 instances'));
          await page.locator('#cls').selectOption('1');
          await page.getByRole('button',{name:'Change class by click',exact:true}).click();
          await c.click({position:{x:75,y:45}});
          await page.getByRole('button',{name:'Save labels',exact:true}).click();
          await page.waitForFunction(() => !busy);
        }"""
        run(cli+['run-code',code],'browser-draw-reclassify')
        assert snapshot()['labels'][0]['class_id']==1
        code="""async page => {
          await page.getByRole('button',{name:'Remove by click',exact:true}).click();
          await page.locator('#c').click({position:{x:75,y:45}});
          await page.getByRole('button',{name:'Save labels',exact:true}).click();
          await page.waitForFunction(() => document.querySelector('#status').textContent.includes('saved 0 instances'));
        }"""
        run(cli+['run-code',code],'browser-remove');assert snapshot()['labels']==[]
        code="""async page => {
          await page.route('**/api/labels/other_mk3-f012.png',async route => {
            await new Promise(resolve => setTimeout(resolve,600));await route.continue();
          });
          await page.locator('#images').selectOption('other_mk3-f012.png');
          await page.locator('#images').selectOption('40x_p1-3b9-04.png');
          await page.waitForFunction(() => !loading && current==='40x_p1-3b9-04.png');
          await page.waitForTimeout(1000);
          if(!await page.evaluate(() => current==='40x_p1-3b9-04.png' && labels.length===0)) throw Error('Late image response corrupted selection');
          await page.unroute('**/api/labels/other_mk3-f012.png');
        }"""
        run(cli+['run-code',code],'browser-rapid-switch')
        # Synthetic geometry is removed. Restore real existing labels before training.
        save(parse_label(root/'v1/train/labels'/Path(name).with_suffix('.txt').name))
        run(cli+['reload'],'browser-reload');run(cli+['snapshot'],'browser-restored-snapshot')
        code="""async page => {
          await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('40x_p1-3b9-04.png —'));
          await page.locator('#reviewer').fill('automated-smoke-test-original-hdf5');
          await page.locator('#notes').fill('Mechanical test only; original HDF5 polygons restored. Not biological approval.');
          await page.getByRole('button',{name:'Save and approve reviewed image',exact:true}).click();
          await page.waitForFunction(() => document.querySelector('#status').textContent.includes('reviewed and approved'));
          await page.locator('#zoom').selectOption('2');
        }"""
        run(cli+['run-code',code],'browser-approve')
        assert snapshot()['review']['valid']
        screenshot=repo/'output/playwright'/('annotation-loop-'+root.name+'.png')
        screenshot.parent.mkdir(parents=True,exist_ok=True)
        run(cli+['run-code',f"async page => {{ await page.screenshot({{path:{json.dumps(str(screenshot))},fullPage:true}}); }}"],'browser-screenshot')
    finally:
        subprocess.run(cli+['close'],cwd=str(repo),env=env,stdout=subprocess.DEVNULL)
        server.shutdown();server.server_close();thread.join()
    run(py+['annotation_cycle.py','compile','--dataset',root/'v1.json','--reviewed-dataset',root/'review.json',
            '--name','loop-v2','--output-dir',root/'v2','--dataset-info',root/'v2.json'],'compile')
    for split in ('val','test'):
        for source in (root/'v1'/split).rglob('*'):
            if source.suffix in ('.png','.txt'):assert digest(source)==digest(root/'v2'/split/source.relative_to(root/'v1'/split))
    train('v2')
    for version in ('v1','v2'):
        run(py+['train_yolo.py','--dataset',root/(version+'.json'),'--annotated-only','--evaluate',
            '--eval-split','test','--model',root/('model-'+version+'.pt'),'--img-size','320',
            '--batch-size','2','--workers','0','--seed','42','--device','0','--run-dir',root/'evaluation'],'eval-'+version)
    export('v2','reannotation')
    run(py+['annotation_cycle.py','qc','--dataset',root/'v2.json','--output',root/'qc-v2.json'],'qc-v2')
    for version in ('v1','v2','reannotation'):
        run(py+['create_label_overlays.py','--dataset',root/(version+'.json'),'--output-dir',root/('overlays-'+version)],'overlays-'+version)
    run(py+['compare_annotation_models.py','--baseline-dataset',root/'v1.json','--reviewed-dataset',root/'v2.json',
            '--baseline-report',root/'evaluation/model-v1/protocol_metrics.json',
            '--reviewed-report',root/'evaluation/model-v2/protocol_metrics.json',
            '--baseline-training-report',root/'training/model-v1/protocol_metrics.json',
            '--reviewed-training-report',root/'training/model-v2/protocol_metrics.json',
            '--output',root/'mechanical-comparison.md'],'comparison')
    assert all(digest(root/p)==value for p,value in raw_hashes.items())
    atomic_text(root/'verification.json',json.dumps({'status':'passed','biological_approval':False,
        'commands':commands,'raw_unchanged':True,'holdouts_unchanged':True,
        'test_instances':len((root/'v1/test/labels/40x_p1-1e3-13.txt').read_text().splitlines())},indent=2)+'\n')
    print(f'Mechanical loop passed: {root}. No scientific improvement claim.',flush=True)


if __name__=='__main__':main()
