from dataclasses import dataclass
import numpy as np
import pandas as pd
@dataclass
class SetupResult:
 found:bool;name:str;pivot:float|None=None;stop:float|None=None;quality:float=0;reason:str=''
def enrich(d,c):
 x=d.copy();t=c['trend'];x['sma50']=x.close.rolling(t['sma_short']).mean();x['vol50']=x.volume.rolling(50).mean();return x
def advance(x,start,look=126):
 p=x.iloc[max(0,start-look):start];return float(p.close.iloc[-1]/p.low.min()-1) if len(p)>=30 else -1
def detect_flat_base(d,c):
 x=enrich(d,c);q=c['flat_base']
 for n in range(q['min_length'],q['max_length']+1):
  b=x.iloc[-n:];depth=1-float(b.low.min()/b.high.max());prior=x.volume.iloc[-n-50:-n];vr=float(b.volume.mean()/prior.mean()) if len(prior)==50 else 99
  if advance(x,len(x)-n,q['prior_lookback'])>=q['prior_advance'] and depth<=q['max_depth'] and vr<=q['max_base_volume_ratio']:
   p=float(b.high.iloc[:-1].max());return SetupResult(True,'Flat base breakout',p,max(float(b.low.tail(20).min()*.995),p*.92),85,f'{n}d base, {depth:.1%} depth')
 return SetupResult(False,'Flat base breakout')
def detect_cup_handle(d,c):
 x=enrich(d,c);q=c['cup_handle']
 for cn in range(q['min_cup_length'],q['max_cup_length']+1,5):
  for hn in range(q['min_handle_length'],q['max_handle_length']+1):
   if cn+hn>=len(x):continue
   cup=x.iloc[-cn-hn:-hn];h=x.iloc[-hn:];left=float(cup.high.iloc[:max(2,cn//3)].max());right=float(cup.high.iloc[-max(2,cn//3):].max());bottom=float(cup.low.min());depth=1-bottom/left;trough=int(np.argmin(cup.low.to_numpy()));hd=1-float(h.low.min()/h.high.max())
   if advance(x,len(x)-cn-hn)>=q['prior_advance'] and q['min_cup_depth']<=depth<=q['max_cup_depth'] and cn//3<=trough<=2*cn//3 and abs(right/left-1)<=q['rim_tolerance'] and hd<=min(q['max_handle_depth'],depth/2) and float(h.low.min())>=bottom+.5*(left-bottom) and float(h.volume.mean())<float(x.volume.tail(50).mean()):return SetupResult(True,'Cup with handle breakout',float(h.high.iloc[:-1].max()),float(h.low.min()*.99),90,f'{cn}d cup, {hn}d handle')
 return SetupResult(False,'Cup with handle breakout')
def detect_vcp(d,c):
 x=enrich(d,c);q=c['vcp'];b=x.tail(q['max_base_length']);a=b.close.to_numpy();hs=[];ls=[]
 for i in range(3,len(a)-3):
  w=a[i-3:i+4]
  if a[i]==w.max():hs.append(i)
  if a[i]==w.min():ls.append(i)
 pairs=[]
 for h in hs:
  z=[l for l in ls if l>h]
  if z:pairs.append((h,z[0],1-a[z[0]]/a[h]))
 ds=[p[2] for p in pairs if p[2]>0]
 if q['min_contractions']<=len(ds)<=q['max_contractions'] and all(ds[i]<=ds[i-1]*(1-q['min_depth_reduction']) for i in range(1,len(ds))) and ds[-1]<=q['max_final_depth'] and advance(x,len(x)-len(b))>=q['prior_advance']:
  vs=[float(b.volume.iloc[h:l+1].mean()) for h,l,_ in pairs]
  if all(vs[i]<=vs[i-1] for i in range(1,len(vs))):
   z=b.iloc[pairs[-1][0]:];return SetupResult(True,'Volatility contraction pattern',float(z.high.iloc[:-1].max()),float(z.low.min()*.995),90,'shrinking contractions')
 return SetupResult(False,'Volatility contraction pattern')
def detect_pullback(d,c):
 x=enrich(d,c);q=c['pullback'];a=x.iloc[-1];p=x.iloc[-2];s=float(x.high.iloc[-q['swing_lookback']-1:-1].max());pb=1-float(a.close/s)
 ok=q['min_pullback']<=pb<=q['max_pullback'] and abs(float(a.low/a.sma50)-1)<=q['max_distance_sma50'] and a.close>=a.sma50 and a.close>a.open and a.close>=a.low+.5*(a.high-a.low) and a.close>p.high and float(x.volume.tail(5).mean())<float(a.vol50) and a.sma50>x.sma50.iloc[-21]
 return SetupResult(True,'Pullback to rising SMA50',float(a.close),float(min(x.low.tail(3).min(),a.sma50)*.99),90,f'{pb:.1%} pullback') if ok else SetupResult(False,'Pullback to rising SMA50')
