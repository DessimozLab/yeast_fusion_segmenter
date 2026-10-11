#!/usr/bin/env python3
"""Local browser editor for hand-correcting prepared YOLO segmentation labels."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlparse

import yaml
from annotation_cycle import CLASSES, digest, now, atomic_text, read_registry

PAGE = r'''<!doctype html><meta charset="utf-8"><title>YOLO annotation editor</title>
<style>body{font:14px sans-serif;margin:1rem}#images{width:19rem;height:80vh;float:left;margin-right:1rem}main{overflow:hidden}.viewport{overflow:auto;max-height:75vh}canvas{max-width:100%;max-height:75vh;border:1px solid #888;display:block}button,select{margin:.25rem}.active{background:#5a5;color:#fff}.warn{color:#a30}</style>
<h1>YOLO annotation editor</h1><p class=warn>Edits prepared YOLO labels only; raw CZI/TIFF/HDF5 files are never changed. Each save creates a one-time <code>.manual-backup</code> and appends an audit record.</p>
<select id=images size=30 onchange="loadImage(this.value)"></select><main><p id=status>Loading…</p><label>Class <select id=cls></select></label><label>Zoom <select id=zoom onchange="zoomCanvas()"><option value=0>Fit</option><option value=1>100%</option><option value=2>200%</option><option value=4>400%</option></select></label><button onclick="mode='draw'">Draw polygon</button><button onclick="finish()">Finish polygon</button><button onclick="mode='remove'">Remove by click</button><button onclick="mode='class'">Change class by click</button><button onclick="save()">Save labels</button><button onclick="dirty=false;loadImage(current)">Discard unsaved</button><p><label>Reviewer <input id=reviewer></label><label>Review notes <input id=notes></label><button onclick="approve()">Save and approve reviewed image</button><button onclick="defer()">Defer unresolved image</button></p><p>Draw: click at least three vertices, then Finish. Remove/change class: click inside an existing polygon. Approval confirms you checked the entire image, its orientation, missing cells and all phenotype classes. Frozen holdouts and compiled datasets are read-only.</p><div class=viewport><canvas id=c></canvas></div></main>
<script>
let current, dirty=false, labels=[], draft=[], mode='draw', image=new Image(), classes=[], canvas=document.querySelector('#c'), ctx=canvas.getContext('2d'), revision={}, editable=false, loading=false, busy=false, generation=0; const colors=['#00ff00','#00ffff','#ff9900','#ff00ff','#0066ff','#ffff00','#ff0000'];
function controls(){document.querySelectorAll('main button,#cls,#reviewer,#notes').forEach(x=>x.disabled=loading||busy||!editable);document.querySelector('#images').disabled=busy}
function inside(p,poly){let hit=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++)if((poly[i][1]>p[1])!=(poly[j][1]>p[1])&&p[0]<(poly[j][0]-poly[i][0])*(p[1]-poly[i][1])/(poly[j][1]-poly[i][1])+poly[i][0])hit=!hit;return hit}
function zoomCanvas(){let z=+document.querySelector('#zoom').value;canvas.style.maxWidth=z?'none':'100%';canvas.style.maxHeight=z?'none':'75vh';canvas.style.width=z?(canvas.width*z)+'px':'';canvas.style.height=z?(canvas.height*z)+'px':''}
function draw(){ctx.drawImage(image,0,0); for(let x of labels){ctx.strokeStyle=colors[x.class_id%colors.length];ctx.lineWidth=2;ctx.beginPath();x.points.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));ctx.closePath();ctx.stroke()}if(draft.length){ctx.strokeStyle='#fff';ctx.beginPath();draft.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));ctx.stroke()}}
async function loadImage(name){if(busy)return;if(dirty&&!confirm('Discard unsaved changes?')){document.querySelector('#images').value=current;return}let token=++generation;loading=true;controls();document.querySelector('#status').textContent='Loading '+name;try{let r=await fetch('/api/labels/'+encodeURIComponent(name));if(!r.ok)throw Error(await r.text());let d=await r.json(),next=new Image();next.src='/api/image/'+encodeURIComponent(name);await next.decode();if(token!==generation)return;current=name;labels=d.labels;revision=d;editable=d.editable;image=next;draft=[];dirty=false;canvas.width=image.width;canvas.height=image.height;zoomCanvas();draw();document.querySelector('#notes').value=d.review.notes||'';document.querySelector('#status').textContent=name+' — '+labels.length+' instances — '+(editable?'':'READ ONLY — ')+(d.review.valid?'approved by '+d.review.reviewer:d.review.status||'pending')}catch(e){if(token===generation){editable=false;document.querySelector('#status').textContent=e.message}}finally{if(token===generation){loading=false;controls()}}}
function finish(){if(!editable||loading||busy)return;if(draft.length<3)return alert('A polygon needs at least three vertices');labels.push({class_id:+document.querySelector('#cls').value,points:draft});draft=[];dirty=true;draw()}
canvas.onclick=e=>{if(!current||!editable||loading||busy)return;let r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];if(mode!=='draw'){let i=labels.findIndex(x=>inside(p,x.points));if(i>=0){if(mode==='remove')labels.splice(i,1);else labels[i].class_id=+document.querySelector('#cls').value;dirty=true}else alert('No polygon at that location')}else{draft.push(p);dirty=true}draw()}
async function post(route,extra){let r=await fetch('/api/'+route+'/'+encodeURIComponent(current),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...revision,...extra})});if(!r.ok)throw Error(await r.text());let d=await r.json();revision={...revision,...d};return d}
async function saveDraft(){if(draft.length)throw Error('Finish the polygon before saving');await post('labels',{labels});dirty=false;document.querySelector('#status').textContent=current+' — saved '+labels.length+' instances (not approved)'}
async function action(fn){if(!editable||loading||busy)return;busy=true;controls();try{await fn()}catch(e){alert(e.message)}finally{busy=false;controls()}}
async function save(){await action(saveDraft)}
async function approve(){await action(async()=>{let reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer)throw Error('Enter a reviewer name');await saveDraft();await post('approve',{reviewer,notes:document.querySelector('#notes').value});document.querySelector('#status').textContent=current+' — reviewed and approved by '+reviewer})}
async function defer(){await action(async()=>{let reviewer=document.querySelector('#reviewer').value.trim(),notes=document.querySelector('#notes').value.trim();if(!reviewer||!notes)throw Error('Enter reviewer and notes explaining the unresolved issue');await saveDraft();await post('defer',{reviewer,notes});document.querySelector('#status').textContent=current+' — deferred; excluded from compilation'})}
async function init(){let d=await (await fetch('/api/records')).json();classes=d.classes;let c=document.querySelector('#cls');classes.forEach((x,i)=>c.add(new Option(i+': '+x,i)));let s=document.querySelector('#images');d.images.forEach(x=>s.add(new Option(x,x)));if(d.images.length){s.value=d.images[0];loadImage(s.value)}}init();
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});
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
    if tuple(names) != CLASSES:
        raise ValueError('Expected the fixed seven-class phenotype map')
    return root, names

def parse_label(path):
    if not path.exists(): return []
    result=[]
    for line in path.read_text().splitlines():
        fields=line.split(); coords=[float(v) for v in fields[1:]]
        if len(coords)<6 or len(coords)%2: raise ValueError(f'Malformed YOLO label: {path}')
        result.append({'class_id':int(fields[0]),'points':[list(p) for p in zip(coords[::2],coords[1::2])]})
    return result

def validate_labels(value, class_count):
    if not isinstance(value,list): raise ValueError('Expected a label list')
    for segment in value:
        if not isinstance(segment,dict) or type(segment.get('class_id')) is not int or not 0<=segment['class_id']<class_count: raise ValueError('Invalid class')
        points=segment.get('points');
        if not isinstance(points,list) or len(points)<3: raise ValueError('Every polygon needs three points')
        if any(not isinstance(p,list) or len(p)!=2 or not all(isinstance(v,(int,float)) and 0<=v<=1 for v in p) for p in points): raise ValueError('Invalid normalized polygon')
        x0,y0=points[0]
        area=sum((a[0]-x0)*(b[1]-y0)-(b[0]-x0)*(a[1]-y0) for a,b in zip(points,points[1:]+points[:1]))
        if area==0: raise ValueError('Polygon must have non-zero area')

def make_handler(root,names,split):
    images=root/split/'images'; labels=root/split/'labels'; audit=root/'manual_annotation_edits.jsonl'
    approvals_path=root/'review_approvals.json'
    write_lock=Lock()
    contract_path=root/'dataset-contract.json'
    contract=json.loads(contract_path.read_text()) if contract_path.exists() else {}
    inventory={row['image']:row for row in contract.get('prepared_images', [])}
    def editable(name):
        return contract.get('purpose')=='candidate_review' and inventory.get(name,{}).get('role') in ('train','pool')
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,kind,body): self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_HEAD(self):
            p=urlparse(self.path).path
            if p.startswith('/api/image/'):
                name=unquote(p.rsplit('/',1)[-1]); source=images/name
                if Path(name).name==name and name.endswith('.png') and source.resolve().parent==images.resolve() and source.exists():
                    self.send_response(200); self.send_header('Content-Type','image/png'); self.send_header('Content-Length',str(source.stat().st_size)); self.end_headers(); return
            self.send_error(404)
        def do_GET(self):
            p=urlparse(self.path).path
            try:
                if p=='/': return self.reply(200,'text/html',PAGE.encode())
                if p=='/favicon.ico': return self.reply(204,'image/x-icon',b'')
                if p=='/api/records': return self.reply(200,'application/json',json.dumps({'images':[x.name for x in sorted(images.glob('*.png'))],'classes':names}).encode())
                name=unquote(p.rsplit('/',1)[-1])
                if Path(name).name!=name or not name.endswith('.png') or (images/name).resolve().parent!=images.resolve():
                    raise ValueError('Invalid image name')
                source=images/name if p.startswith('/api/image/') else labels/(Path(name).stem+'.txt')
                if p.startswith('/api/review/'):
                    approvals=json.loads(approvals_path.read_text()) if approvals_path.exists() else {}
                    state=approvals.get(f'{split}/{name}',{})
                    valid=bool(state) and state.get('status','approved')=='approved' and state['image_sha256']==digest(images/name) and state['label_sha256']==digest(source)
                    return self.reply(200,'application/json',json.dumps(dict(state,valid=valid)).encode())
                if p.startswith('/api/image/'): return self.reply(200,'image/png',source.read_bytes())
                if p.startswith('/api/labels/'):
                    with write_lock:
                        state=read_registry(root).get(f'{split}/{name}',{})
                        revision={'label_sha256':digest(source),'image_sha256':digest(images/name)}
                        valid=state.get('status','approved')=='approved' and bool(state) and all(state.get(k)==v for k,v in revision.items())
                        return self.reply(200,'application/json',json.dumps(dict(revision,labels=parse_label(source),editable=editable(name),review=dict(state,valid=valid))).encode())
                self.reply(404,'text/plain',b'Not found')
            except Exception as e:self.reply(400,'text/plain',str(e).encode())
        def do_POST(self):
            with write_lock:
                self.handle_post()
        def handle_post(self):
            route=urlparse(self.path).path
            if not route.startswith(('/api/labels/','/api/approve/','/api/defer/')): return self.reply(404,'text/plain',b'Not found')
            try:
                name=unquote(urlparse(self.path).path.rsplit('/',1)[-1])
                if Path(name).name != name or not name.endswith('.png'):
                    raise ValueError('Invalid image name')
                image=images/name
                if not image.exists(): raise ValueError('Unknown image')
                if not editable(name): return self.reply(403,'text/plain',b'Read-only dataset or held-out acquisition. Export a separate review dataset.')
                value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if image.resolve().parent != images.resolve(): raise ValueError('Image outside selected split')
                target=labels/(image.stem+'.txt')
                if target.resolve().parent != labels.resolve(): raise ValueError('Label outside selected split')
                if not isinstance(value,dict): raise ValueError('Expected revision and labels object')
                if value.get('label_sha256')!=digest(target) or value.get('image_sha256')!=digest(image):
                    return self.reply(409,'text/plain',b'Image/labels changed since loading. Reload and reconcile; your edits were not saved.')
                if digest(image)!=inventory[name]['image_sha256']: raise ValueError('Review PNG changed; re-export and re-review')
                approvals=json.loads(approvals_path.read_text()) if approvals_path.exists() else {}
                approval_key=f'{split}/{name}'
                if route.startswith(('/api/approve/','/api/defer/')):
                    reviewer=value.get('reviewer','').strip()
                    if not reviewer: raise ValueError('Reviewer name is required')
                    notes=value.get('notes','').strip()
                    deferred=route.startswith('/api/defer/')
                    if deferred and not notes: raise ValueError('Deferred reviews require explanatory notes')
                    target=labels/(image.stem+'.txt')
                    segments=parse_label(target); validate_labels(segments,len(names))
                    approvals[approval_key]={'reviewer':reviewer,'reviewed_utc':now(),
                        'image_sha256':digest(image),'label_sha256':digest(target),
                        'status':'deferred' if deferred else 'approved','notes':notes}
                    atomic_text(approvals_path,json.dumps(approvals,indent=2)+'\n')
                    with audit.open('a') as f:
                        f.write(json.dumps(dict(approvals[approval_key],image=name,split=split,event='defer' if deferred else 'approve'))+'\n')
                    return self.reply(200,'application/json',json.dumps(approvals[approval_key]).encode())
                value=value['labels']
                validate_labels(value,len(names))
                target=labels/(image.stem+'.txt'); backup=target.with_suffix('.txt.manual-backup')
                if target.exists() and not backup.exists(): shutil.copy2(target,backup)
                text=''.join(f"{x['class_id']} "+' '.join(f'{v:.12f}' for p in x['points'] for v in p)+'\n' for x in value)
                atomic_text(target,text)
                if approval_key in approvals:
                    approvals[approval_key]['status']='changed_since_approval'
                    atomic_text(approvals_path,json.dumps(approvals,indent=2)+'\n')
                # Ultralytics caches can otherwise retain old contours.
                for cache in root.rglob('*.cache'): cache.unlink()
                with audit.open('a') as f:
                    f.write(json.dumps({
                        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                        'split': split, 'image': name, 'instances': len(value),
                        'label_path': str(target.relative_to(root)),
                    })+'\n')
                self.reply(200,'application/json',json.dumps({'label_sha256':digest(target),'image_sha256':digest(image)}).encode())
            except Exception as e:self.reply(400,'text/plain',str(e).encode())
        def log_message(self,*_):pass
    return Handler

def main():
    args=parse_args();root,names=resolve_dataset(args.dataset)
    marker=root/'dataset-contract.json'
    if marker.exists():
        from annotation_cycle import validate_contract
        info=json.loads(marker.read_text())
        validate_contract(info,root,allow_review=info.get('purpose')=='candidate_review')
    if args.host not in ('127.0.0.1','localhost','::1'):
        raise ValueError('This unauthenticated reviewer must remain localhost-only')
    server=ThreadingHTTPServer((args.host,args.port),make_handler(root,names,args.split))
    print(f'Open http://{args.host}:{args.port}');server.serve_forever()
if __name__=='__main__':main()
