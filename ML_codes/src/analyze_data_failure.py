import argparse, gc, json, math
from collections import Counter
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO

NAMES={0:'pipe',1:'shipwreck',2:'mine',3:'ghost_net'}

def parse():
 p=argparse.ArgumentParser()
 p.add_argument('--model',default=r'K:\Debris model\runs\drishti_v5_yolov8s_baseline-2\weights\best.pt')
 p.add_argument('--data',default=r'D:\DATASETS\DRISHTI-SSS-TRAIN')
 p.add_argument('--split',default='test')
 p.add_argument('--output',default=r'K:\Debris model\runs\drishti_v5_failure_analysis')
 p.add_argument('--imgsz',type=int,default=640); p.add_argument('--device',default='0')
 p.add_argument('--conf',type=float,default=.001); p.add_argument('--iou',type=float,default=.7)
 p.add_argument('--match-iou',type=float,default=.5); p.add_argument('--batch',type=int,default=1)
 p.add_argument('--gallery-top',type=int,default=30); p.add_argument('--max-det',type=int,default=300)
 return p.parse_args()

def read_labels(p):
 if not p.exists() or not p.read_text(encoding='utf-8').strip(): return []
 out=[]
 for n,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
  z=line.split()
  if len(z)!=5: continue
  try: c=int(float(z[0])); b=list(map(float,z[1:]))
  except ValueError: continue
  if c in NAMES: out.append((c,b,n))
 return out

def xywh(b,w,h):
 x,y,bw,bh=b; return [(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
 inter=max(0,x2-x1)*max(0,y2-y1); u=area(a)+area(b)-inter
 return inter/u if u else 0

def match(gt,pr,t):
 c=[]
 for gi,g in enumerate(gt):
  for pi,p in enumerate(pr):
   if g['c']==p['c']:
    s=iou(g['b'],p['b'])
    if s>=t:c.append((s,gi,pi))
 c.sort(reverse=True); ug=set();up=set(); pairs=[]
 for s,gi,pi in c:
  if gi not in ug and pi not in up: ug.add(gi);up.add(pi);pairs.append((gi,pi,s))
 return pairs,ug,up

def best(g,pr): return max(((iou(g['b'],p['b']),i) for i,p in enumerate(pr)),default=(0,None))

def draw(im,gt,pr,fg=None,fp=None):
 o=im.copy()
 for i,g in enumerate(gt):
  x1,y1,x2,y2=map(int,g['b']); cv2.rectangle(o,(x1,y1),(x2,y2),(255,255,255),2)
  cv2.putText(o,('>>> ' if i==fg else '')+'GT '+NAMES[g['c']],(max(0,x1),max(18,y1-5)),0,.55,(255,255,255),2)
 for i,p in enumerate(pr):
  x1,y1,x2,y2=map(int,p['b']); cv2.rectangle(o,(x1,y1),(x2,y2),(180,180,180),1)
  cv2.putText(o,('>>> ' if i==fp else '')+f"P {NAMES[p['c']]} {p['s']:.2f}",(max(0,x1),min(o.shape[0]-5,y2+18)),0,.5,(220,220,220),1)
 return o

def gallery(items,path):
 if not items:return
 cw,ch,cols=480,360,3; cells=[]
 for it in items:
  im=it['im']; sc=min(cw/im.shape[1],ch/im.shape[0]); nw,nh=max(1,int(im.shape[1]*sc)),max(1,int(im.shape[0]*sc))
  th=cv2.resize(im,(nw,nh)); cell=np.zeros((ch,cw,3),np.uint8); x,y=(cw-nw)//2,(ch-nh)//2; cell[y:y+nh,x:x+nw]=th
  cv2.putText(cell,it['cap'][:90],(8,22),0,.45,(255,255,255),1); cells.append(cell)
 sheet=np.zeros((math.ceil(len(cells)/cols)*ch,cols*cw,3),np.uint8)
 for i,c in enumerate(cells): r,k=divmod(i,cols); sheet[r*ch:(r+1)*ch,k*cw:(k+1)*cw]=c
 cv2.imwrite(str(path),sheet)

def main():
 a=parse(); root=Path(a.data); idir=root/a.split/'images'; ldir=root/a.split/'labels'; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 for d in ['false_negatives','localization_failures','confusions','false_positives']:(out/d).mkdir(exist_ok=True)
 paths=sorted(p for p in idir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
 if not paths: raise RuntimeError(f'No images found: {idir}')
 model=YOLO(a.model); stats={n:{'gt':0,'tp':0,'fn':0,'fp':0,'ious':[],'areas':[],'fnareas':[],'confs':[]} for n in NAMES.values()}; confs=Counter(); sizes=Counter(); recs=[]; cand={x:[] for x in ['false_negatives','localization_failures','confusions','false_positives']}; all_iou=[]; all_conf=[]
 print(f'Analyzing {len(paths)} images in batches of {a.batch}...')
 for start in range(0,len(paths),a.batch):
  bp=paths[start:start+a.batch]
  with torch.inference_mode(): res=model.predict(source=[str(x) for x in bp],imgsz=a.imgsz,device=a.device,conf=a.conf,iou=a.iou,max_det=a.max_det,batch=a.batch,verbose=False)
  for path,r in zip(bp,res):
   im=cv2.imread(str(path));
   if im is None:continue
   h,w=im.shape[:2]; gt=[]
   for c,b,_ in read_labels(ldir/f'{path.stem}.txt'):
    bb=xywh(b,w,h); f=area(bb)/(w*h); gt.append({'c':c,'b':bb,'a':f}); n=NAMES[c]; stats[n]['gt']+=1;stats[n]['areas'].append(f)
    sizes[(n,'tiny_<0.5%')]+=f<.005; sizes[(n,'small_0.5-2%')]+=.005<=f<.02; sizes[(n,'medium_2-5%')]+=.02<=f<.05; sizes[(n,'large_>=5%')]+=f>=.05
   pr=[]
   if r.boxes is not None and len(r.boxes):
    for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy().astype(int),r.boxes.conf.cpu().numpy()):
     if int(c) in NAMES: pr.append({'c':int(c),'b':[float(v) for v in b],'s':float(s)});all_conf.append(float(s))
   pairs,ug,up=match(gt,pr,a.match_iou); rec={'image':str(path),'gt':len(gt),'predictions':len(pr),'matches':[],'false_negatives':[],'false_positives':[],'confusions':[]}
   for gi,pi,s in pairs:
    g,p=gt[gi],pr[pi];n=NAMES[g['c']];stats[n]['tp']+=1;stats[n]['ious'].append(s);stats[n]['confs'].append(p['s']);all_iou.append(s);rec['matches'].append({'class':n,'iou':s,'confidence':p['s']})
    if s<.75:cand['localization_failures'].append((1-s,{'im':draw(im,gt,pr,gi,pi),'cap':f'{n}: IoU={s:.3f}, conf={p["s"]:.3f}'}))
   for gi,g in enumerate(gt):
    if gi in ug:continue
    n=NAMES[g['c']];stats[n]['fn']+=1;stats[n]['fnareas'].append(g['a']);bi,bpi=best(g,pr);reason='no_prediction'
    if bpi is not None and bi>=.2 and pr[bpi]['c']!=g['c']:
     other=NAMES[pr[bpi]['c']];reason='class_confusion';confs[(n,other)]+=1;rec['confusions'].append({'gt':n,'pred':other,'iou':bi,'confidence':pr[bpi]['s']});cand['confusions'].append((bi,{'im':draw(im,gt,pr,gi,bpi),'cap':f'GT {n} -> P {other}, IoU={bi:.3f}'}))
    rec['false_negatives'].append({'class':n,'area':g['a'],'best_iou':bi,'reason':reason});cand['false_negatives'].append((g['a'],{'im':draw(im,gt,pr,gi),'cap':f'FN {n}, area={g["a"]:.4f}, bestIoU={bi:.3f}, {reason}'}))
   for pi,p in enumerate(pr):
    if pi in up:continue
    n=NAMES[p['c']];stats[n]['fp']+=1; bg=max(((iou(g['b'],p['b']),NAMES[g['c']]) for g in gt),default=(0,None));cand['false_positives'].append((p['s'],{'im':draw(im,gt,pr,None,pi),'cap':f'FP {n}, conf={p["s"]:.3f}, bestGT={bg[1] or "none"} IoU={bg[0]:.3f}'}));rec['false_positives'].append({'class':n,'confidence':p['s'],'best_gt':bg[1],'best_iou':bg[0]})
   recs.append(rec)
  del res;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()
  done=min(start+a.batch,len(paths));print(f'  {done}/{len(paths)}')
 report={'model':a.model,'data':str(root),'split':a.split,'images_analyzed':len(recs),'match_iou':a.match_iou,'prediction_conf':a.conf,'batch_size':a.batch,'overall':{'ground_truth':sum(s['gt'] for s in stats.values()),'true_positives':sum(s['tp'] for s in stats.values()),'false_negatives':sum(s['fn'] for s in stats.values()),'unmatched_predictions':sum(s['fp'] for s in stats.values()),'mean_matched_iou':float(np.mean(all_iou)) if all_iou else None,'median_matched_iou':float(np.median(all_iou)) if all_iou else None},'classes':{},'confusions':[{'ground_truth':g,'prediction':p,'count':n} for (g,p),n in confs.most_common()],'size_buckets':{f'{g}/{b}':int(n) for (g,b),n in sorted(sizes.items())},'images':recs}
 for n,s in stats.items():report['classes'][n]={'ground_truth':s['gt'],'true_positives':s['tp'],'false_negatives':s['fn'],'unmatched_predictions':s['fp'],'recall_at_match_iou':s['tp']/s['gt'] if s['gt'] else None,'mean_matched_iou':float(np.mean(s['ious'])) if s['ious'] else None,'median_matched_iou':float(np.median(s['ious'])) if s['ious'] else None,'mean_gt_area_fraction':float(np.mean(s['areas'])) if s['areas'] else None,'median_gt_area_fraction':float(np.median(s['areas'])) if s['areas'] else None,'mean_fn_area_fraction':float(np.mean(s['fnareas'])) if s['fnareas'] else None,'median_fn_area_fraction':float(np.median(s['fnareas'])) if s['fnareas'] else None,'mean_tp_confidence':float(np.mean(s['confs'])) if s['confs'] else None}
 (out/'failure_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 lines=['DRISHTI-SSS SEALED TEST FAILURE ANALYSIS','='*60,f'Images analyzed: {len(recs)}',f'Match IoU: {a.match_iou:.2f}',f'Batch size: {a.batch}','\nCLASS SUMMARY','-'*60]
 for n,s in report['classes'].items():lines.append(f"{n:12s} GT={s['ground_truth']:4d} TP={s['true_positives']:4d} FN={s['false_negatives']:4d} Recall={s['recall_at_match_iou'] or 0:.3f} MeanIoU={s['mean_matched_iou'] or 0:.3f}")
 lines+=['\nCONFUSIONS','-'*60]+([f"{x['ground_truth']:12s} -> {x['prediction']:12s}: {x['count']}" for x in report['confusions']] or ['None'])+['\nOBJECT SIZE BUCKETS','-'*60]+[f'{k:30s}: {v}' for k,v in report['size_buckets'].items()];(out/'failure_analysis.txt').write_text('\n'.join(lines),encoding='utf-8')
 for folder,rev in [('false_negatives',True),('localization_failures',True),('confusions',False),('false_positives',False)]:
  arr=cand[folder];arr.sort(key=lambda x:x[0],reverse=rev);sel=[x[1] for x in arr[:a.gallery_top]];gallery(sel,out/folder/'gallery.jpg')
  for i,it in enumerate(sel[:10],1):cv2.imwrite(str(out/folder/f'{i:02d}.jpg'),it['im'])
 print('\nANALYSIS COMPLETE');print(f'JSON: {out/"failure_analysis.json"}');print(f'TXT : {out/"failure_analysis.txt"}')

if __name__=='__main__':main()
