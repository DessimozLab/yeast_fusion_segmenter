#!/usr/bin/env python3
"""Local browser UI for manually approving annotated CZI mask orientations.

Raw CZI and HDF5 files are never modified.  The selected transform is saved
to a JSON file consumable by ``prepare_yolo_data.py --orientation-overrides``.
"""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image

from image_dataset import MicroscopyImageDataset
from prepare_yolo_data import (
    ORIENTATION_TRANSFORMS, apply_mask_orientation, center_crop_or_pad,
    load_annotation_mask, split_mask,
)

PALETTE = ((0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255),
           (255, 0, 0), (0, 255, 255), (0, 0, 255))

PAGE = """<!doctype html><title>Annotation orientation review</title>
<style>body{font:15px sans-serif;margin:1rem}#records{width:22rem;height:80vh;float:left;margin-right:1rem}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:1rem}button{border:0;background:none;padding:0;cursor:pointer}img{width:100%;display:block;border:4px solid #bbb}button.selected img{border-color:#2b8a3e}#detail{overflow:hidden}small{display:block;margin:.4rem}</style>
<h1>Manual annotation orientation review</h1><p>Select a record, then click the overlay whose contours align with the image. Saving changes only the override JSON; raw CZI/HDF5 files remain untouched.</p>
<select id=records size=30 onchange=show(this.value)></select><main id=detail><p>Loading records…</p></main>
<script>
let data=[]; const transforms=['orig','flip_ud','flip_lr','flip_udlr'];
async function init(){data=await (await fetch('/api/records')).json();let s=document.querySelector('#records');data.forEach(r=>s.add(new Option(`${r.key}${r.selected?'  — '+r.selected:''}`,r.key)));if(data.length){s.value=data[0].key;show(data[0].key)}}
function show(key){let r=data.find(x=>x.key===key),d=document.querySelector('#detail');d.innerHTML=`<h2>${key}</h2><p>Current choice: <b>${r.selected||'automatic'}</b></p><div class=grid>${transforms.map(t=>`<button class="${r.selected===t?'selected':''}" onclick="save('${key}','${t}')"><img src="/api/candidate/${encodeURIComponent(key)}/${t}.png"><small>${t}</small></button>`).join('')}</div>`}
async function save(key,transform){let res=await fetch('/api/overrides',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key,transform})});if(!res.ok){alert(await res.text());return}data.find(x=>x.key===key).selected=transform;show(key);document.querySelector(`#records option[value="${key}"]`).text=`${key}  — ${transform}`}
init();</script>"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--overrides', default='data/manual_orientation_overrides.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    return parser.parse_args()


def load_overrides(path):
    if not path.exists():
        return {'schema_version': 1, 'mask_transforms': {}}
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != 1 or not isinstance(payload.get('mask_transforms'), dict):
        raise ValueError(f'Invalid override file: {path}')
    return payload


def make_handler(dataset, override_path):
    records = {f'{r.magnification}/{r.sample_id}': r for r in dataset.annotated_records()}
    def candidate(key, transform):
        record = records[key]
        png = dataset.png_root / record.magnification / f'{record.sample_id}.png'
        if not png.exists():
            raise FileNotFoundError(f'Materialize PNG first: {png}')
        with Image.open(png) as source:
            rgb = center_crop_or_pad(np.asarray(source.convert('RGB')), 1024)
        mask = center_crop_or_pad(load_annotation_mask(record.annotation_path, record.sample_id), 1024)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for class_id, class_mask in enumerate(split_mask(apply_mask_orientation(mask, transform), crop=1024)):
            contours, _ = cv2.findContours((class_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, PALETTE[class_id], 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode('.png', canvas)
        if not ok: raise RuntimeError('PNG encoding failed')
        return encoded.tobytes()
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, kind, body):
            self.send_response(status); self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/': return self.send(200, 'text/html; charset=utf-8', PAGE.encode())
                if path == '/api/records':
                    selected = load_overrides(override_path)['mask_transforms']
                    body = json.dumps([{'key': key, 'selected': selected.get(key)} for key in sorted(records)]).encode()
                    return self.send(200, 'application/json', body)
                parts = path.split('/')
                if len(parts) == 5 and parts[1:3] == ['api', 'candidate'] and parts[4].endswith('.png'):
                    return self.send(200, 'image/png', candidate(unquote(parts[3]), parts[4][:-4]))
                self.send(404, 'text/plain', b'Not found')
            except Exception as error: self.send(400, 'text/plain', str(error).encode())
        def do_POST(self):
            if urlparse(self.path).path != '/api/overrides': return self.send(404, 'text/plain', b'Not found')
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                key, transform = request['key'], request['transform']
                if key not in records or transform not in ORIENTATION_TRANSFORMS: raise ValueError('Invalid record or transform')
                payload = load_overrides(override_path); payload['mask_transforms'][key] = transform
                override_path.parent.mkdir(parents=True, exist_ok=True); override_path.write_text(json.dumps(payload, indent=2) + '\n')
                self.send(200, 'application/json', b'{}')
            except Exception as error: self.send(400, 'text/plain', str(error).encode())
        def log_message(self, *_): pass
    return Handler


def main():
    args = parse_args(); dataset = MicroscopyImageDataset(args.raw_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dataset, Path(args.overrides)))
    print(f'Open http://{args.host}:{args.port}  (Ctrl-C to stop)')
    server.serve_forever()

if __name__ == '__main__': main()
