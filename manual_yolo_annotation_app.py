#!/usr/bin/env python3
"""Local browser editor for hand-correcting prepared YOLO segmentation labels."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

PAGE = r'''<!doctype html><title>YOLO annotation editor</title>
<style>body{font:14px sans-serif;margin:1rem}select{width:19rem;height:80vh;float:left;margin-right:1rem}main{overflow:hidden}canvas{max-width:75vw;max-height:75vh;border:1px solid #888;display:block}button,select{margin:.25rem}.active{background:#5a5;color:#fff}.warn{color:#a30}</style>
<h1>YOLO annotation editor</h1><p class=warn>Edits prepared YOLO labels only; raw CZI/TIFF/HDF5 files are never changed. Each save creates a one-time <code>.manual-backup</code> and appends an audit record.</p>
<select id=images size=30 onchange="loadImage(this.value)"></select><main><p id=status>Loading…</p><label>Class <select id=cls></select></label><button onclick="mode='draw'">Draw polygon</button><button onclick="finish()">Finish polygon</button><button onclick="mode='remove'">Remove by click</button><button onclick="save()">Save labels</button><button onclick="loadImage(current)">Discard unsaved</button><p>Draw: click at least three vertices, then Finish. Remove: click inside an existing polygon.</p><canvas id=c></canvas></main>
<script>
let current, labels=[], draft=[], mode='draw', image=new Image(), classes=[], canvas=document.querySelector('#c'), ctx=canvas.getContext('2d'); const colors=['#00ff00','#00ffff','#ff9900','#ff00ff','#0066ff','#ffff00','#ff0000'];
function inside(p,poly){let hit=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++)if((poly[i][1]>p[1])!=(poly[j][1]>p[1])&&p[0]<(poly[j][0]-poly[i][0])*(p[1]-poly[i][1])/(poly[j][1]-poly[i][1])+poly[i][0])hit=!hit;return hit}
function draw(){ctx.drawImage(image,0,0); for(let x of labels){ctx.strokeStyle=colors[x.class_id%colors.length];ctx.lineWidth=2;ctx.beginPath();x.points.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));ctx.closePath();ctx.stroke()}if(draft.length){ctx.strokeStyle='#fff';ctx.beginPath();draft.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));ctx.stroke()}}
async function loadImage(name){current=name; labels=await (await fetch('/api/labels/'+encodeURIComponent(name))).json();draft=[];image.onload=()=>{canvas.width=image.width;canvas.height=image.height;draw()};image.src='/api/image/'+encodeURIComponent(name);document.querySelector('#status').textContent=name+' — '+labels.length+' instances'}
function finish(){if(draft.length<3)return alert('A polygon needs at least three vertices');labels.push({class_id:+document.querySelector('#cls').value,points:draft});draft=[];draw()}
canvas.onclick=e=>{if(!current)return;let r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)*canvas.width/r.width/canvas.width,(e.clientY-r.top)*canvas.height/r.height/canvas.height];if(mode==='remove'){let i=labels.findIndex(x=>inside(p,x.points));if(i>=0)labels.splice(i,1);else alert('No polygon at that location')}else draft.push(p);draw()}
async function save(){if(draft.length)if(!confirm('Discard unfinished polygon?'))return;let r=await fetch('/api/labels/'+encodeURIComponent(current),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(labels)});if(!r.ok)return alert(await r.text());document.querySelector('#status').textContent=current+' — saved '+labels.length+' instances'}
async function init(){let d=await (await fetch('/api/records')).json();classes=d.classes;let c=document.querySelector('#cls');classes.forEach((x,i)=>c.add(new Option(i+': '+x,i)));let s=document.querySelector('#images');d.images.forEach(x=>s.add(new Option(x,x)));if(d.images.length){s.value=d.images[0];loadImage(s.value)}}init();
</script>'''

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='dataset-info JSON or YOLO dataset.yaml')
    p.add_argument('--split', choices=('train','val','test'), default='train')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8767)
    return p.parse_args()

def resolve_dataset(path):
    source=Path(path); config=json.loads(source.read_text()) if source.suffix=='.json' else None
    yaml_path=Path(config['yolo_data']) if config else source
    data=yaml.safe_load(yaml_path.read_text()); root=Path(data.get('path',yaml_path.parent))
    if not root.is_absolute(): root=(yaml_path.parent/root).resolve()
    names=data['names']
    if isinstance(names,dict):
        # PyYAML may retain numeric keys as integers, while JSON-derived YAML
        # sometimes exposes them as strings. Both are valid YOLO YAML forms.
        names=[names.get(i, names.get(str(i))) for i in range(len(names))]
    return root, names

def parse_label(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_text().splitlines():
        fields=line.split(); coords=[float(v) for v in fields[1:]]
        if len(coords)<6 or len(coords)%2: raise ValueError(f'Malformed YOLO label: {path}')
        result.append({'class_id':int(fields[0]),'points':list(zip(coords[::2],coords[1::2]))})
    return result

def validate_labels(value, class_count):
    if not isinstance(value,list): raise ValueError('Expected a label list')
    for segment in value:
        if not isinstance(segment,dict) or not isinstance(segment.get('class_id'),int) or not 0<=segment['class_id']<class_count: raise ValueError('Invalid class')
        points=segment.get('points');
        if not isinstance(points,list) or len(points)<3: raise ValueError('Every polygon needs three points')
        if any(not isinstance(p,list) or len(p)!=2 or not all(isinstance(v,(int,float)) and 0<=v<=1 for v in p) for p in points): raise ValueError('Invalid normalized polygon')

def make_handler(root,names,split):
    images=root/split/'images'; labels=root/split/'labels'; audit=root/'manual_annotation_edits.jsonl'
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,kind,body): self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_HEAD(self):
            p=urlparse(self.path).path
            if p.startswith('/api/image/'):
                name=unquote(p.rsplit('/',1)[-1]); source=images/name
                if source.exists():
                    self.send_response(200); self.send_header('Content-Type','image/png'); self.send_header('Content-Length',str(source.stat().st_size)); self.end_headers(); return
            self.send_error(404)
        def do_GET(self):
            p=urlparse(self.path).path
            try:
                if p=='/': return self.reply(200,'text/html',PAGE.encode())
                if p=='/api/records': return self.reply(200,'application/json',json.dumps({'images':[x.name for x in sorted(images.glob('*.png'))],'classes':names}).encode())
                name=unquote(p.rsplit('/',1)[-1]); source=images/name if p.startswith('/api/image/') else labels/(Path(name).stem+'.txt')
                if p.startswith('/api/image/'): return self.reply(200,'image/png',source.read_bytes())
                if p.startswith('/api/labels/'): return self.reply(200,'application/json',json.dumps(parse_label(source)).encode())
                self.reply(404,'text/plain',b'Not found')
            except Exception as e:self.reply(400,'text/plain',str(e).encode())
        def do_POST(self):
            if not urlparse(self.path).path.startswith('/api/labels/'): return self.reply(404,'text/plain',b'Not found')
            try:
                name=unquote(urlparse(self.path).path.rsplit('/',1)[-1])
                if Path(name).name != name or not name.endswith('.png'):
                    raise ValueError('Invalid image name')
                image=images/name
                if not image.exists(): raise ValueError('Unknown image')
                value=json.loads(self.rfile.read(int(self.headers['Content-Length']))); validate_labels(value,len(names))
                target=labels/(image.stem+'.txt'); backup=target.with_suffix('.txt.manual-backup')
                if target.exists() and not backup.exists(): shutil.copy2(target,backup)
                text=''.join(f"{x['class_id']} "+' '.join(f'{v:.6f}' for p in x['points'] for v in p)+'\n' for x in value)
                tmp=target.with_suffix('.tmp'); tmp.write_text(text);tmp.replace(target)
                with audit.open('a') as f:
                    f.write(json.dumps({
                        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                        'split': split, 'image': name, 'instances': len(value),
                        'label_path': str(target.relative_to(root)),
                    })+'\n')
                self.reply(200,'application/json',b'{}')
            except Exception as e:self.reply(400,'text/plain',str(e).encode())
        def log_message(self,*_):pass
    return Handler

def main():
    args=parse_args();root,names=resolve_dataset(args.dataset)
    server=ThreadingHTTPServer((args.host,args.port),make_handler(root,names,args.split))
    print(f'Open http://{args.host}:{args.port}');server.serve_forever()
if __name__=='__main__':main()
