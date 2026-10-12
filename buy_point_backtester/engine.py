from dataclasses import dataclass,asdict
import math,numpy as np,pandas as pd
from .analysis import detect_flat_base,detect_cup_handle,detect_vcp,detect_pullback
@dataclass
class Candidate:
 signal_date:pd.Timestamp;symbol:str;strategy:str;setup:str;pivot:float;pattern_stop:float;score:float;volume_ratio:float;details:str
@dataclass
class Position:
 symbol:str;strategy:str;setup:str;signal_date:pd.Timestamp;entry_date:pd.Timestamp;entry_price:float;shares:int;initial_value:float;stop_price:float;pivot:float;score:float;holding_days:int=0

def _r2(series,w):
 x=np.arange(w,dtype=float);xm=x.mean();den=((x-xm)**2).sum(); y=np.log(series.astype(float))
 def f(a):
  ym=a.mean();sl=((x-xm)*(a-ym)).sum()/den;pred=ym+sl*(x-xm);tot=((a-ym)**2).sum();return 1-((a-pred)**2).sum()/tot if tot else 0
 return y.rolling(w).apply(f,raw=True)
def _prep(bars,cfg):
 t=cfg['trend'];prepared={};returns={}
 for s,d in bars.items():
  x=d.copy().sort_index();prev=x.close.shift();tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
  x['ss']=x.close.rolling(t['sma_short']).mean();x['sl']=x.close.rolling(t['sma_long']).mean();x['atrp']=tr.rolling(t['atr_period']).mean()/x.close;x['hi']=x.high.rolling(t['high_low_lookback']).max();x['lo']=x.low.rolling(t['high_low_lookback']).min();x['ret']=x.close/x.close.shift(t['rs_lookback'])-1;x['r2']=_r2(x.close,t['regression_lookback']);x['v50']=x.volume.rolling(50).mean();prepared[s]=x;returns[s]=x.ret
 return prepared,pd.DataFrame(returns).rank(axis=1,pct=True)
def generate_candidates(bars,start,end,cfg,strategies,progress=None):
 p,rank=_prep(bars,cfg);t=cfg['trend'];syms=[s for s in bars if s not in {'SPY','QQQ'}];dates=sorted(set().union(*[set(x.index) for x in bars.values()]));out=[];prev={s:False for s in syms};det=[detect_flat_base,detect_cup_handle,detect_vcp,detect_pullback];spy=p['SPY']
 for n,d in enumerate(dates):
  if not start<=d.date()<=end:continue
  if progress and n%10==0:progress(n/max(1,len(dates)),f"Generating signals {d.date()}")
  if d not in spy.index:continue
  sr=spy.loc[d];spyret=float(sr.ret);spyon=float(sr.close)>float(sr.ss)
  for s in syms:
   x=p[s]
   if d not in x.index:continue
   i=x.index.get_loc(d)
   if not isinstance(i,(int,np.integer)) or i+1<cfg['minimum_history']:continue
   r=x.iloc[i];rp=float(rank.loc[d,s]) if d in rank.index and s in rank and pd.notna(rank.loc[d,s]) else 0
   checks=[r.close>r.ss>r.sl,r.ss>x.ss.iloc[i-t['slope_lookback']],r.sl>x.sl.iloc[i-t['slope_lookback']],r.close>=r.hi*(1-t['max_below_high']),r.close>=r.lo*(1+t['min_above_low']),rp>=1-t['rs_top_percent'] or (t['allow_above_spy_fallback'] and r.ret>spyret),r.r2>=t['min_r_squared'],r.atrp<=t['max_atr_pct']]
   if not all(bool(v) for v in checks):prev[s]=False;continue
   close=float(r.close);vr=float(r.volume/r.v50) if r.v50 else 0;score=sum(w for w,v in zip([20,10,10,10,10,20,10,10],checks) if v)
   if 'TREND_ONLY' in strategies and not prev[s]:out.append(Candidate(d,s,'TREND_ONLY','Healthy trend transition',close,close*.95,score,vr,'Trend changed to PASS'))
   prev[s]=True;hist=bars[s].iloc[:i+1]
   for f in det:
    q=f(hist,cfg)
    if not q.found:continue
    if q.name=='Pullback to rising SMA50':status='BUY'
    elif close>q.pivot*(1+cfg['signal']['extended_above_pivot']):status='EXTENDED'
    elif close>q.pivot:status='BUY' if vr>=cfg['signal']['breakout_volume_multiple'] else 'BREAKOUT_LOW_VOLUME'
    elif close>=q.pivot*(1-cfg['signal']['watch_below_pivot']):status='WATCH'
    else:status='FORMING'
    if status=='BUY' and cfg['market_regime_required'] and not spyon:status='WATCH'
    if status in strategies:out.append(Candidate(d,s,status,q.name,float(q.pivot),float(q.stop),score,vr,q.reason))
 if progress:progress(1.0,f"Generated {len(out)} signals")
 return out

def simulate(bars,cands,start,end,cfg,progress=None):
 dates=sorted(d for d in set().union(*[set(x.index) for x in bars.values()]) if start<=d.date()<=end);by={};pending={};watch={}
 for c in cands:by.setdefault(c.signal_date,[]).append(c)
 cash=float(cfg['starting_capital']);pos={};trades=[];skipped=[];curve=[]
 for ix,d in enumerate(dates):
  if progress and ix%20==0:progress(ix/max(1,len(dates)),f"Simulating {d.date()}")
  for s,z in list(pos.items()):
   if d not in bars[s].index:continue
   h=bars[s].loc[:d];row=h.iloc[-1];z.holding_days+=1;ma=h.close.rolling(int(cfg['exit_sma_period'])).mean();reason=None;price=None
   if len(h)>cfg['exit_sma_period'] and float(h.close.iloc[-2])<float(ma.iloc[-2]):reason='Close below exit SMA';price=float(row.open)*(1-cfg['slippage_pct'])
   elif float(row.open)<=z.stop_price:reason='Gap through stop';price=float(row.open)*(1-cfg['slippage_pct'])
   elif float(row.low)<=z.stop_price:reason='Stop loss';price=z.stop_price*(1-cfg['slippage_pct'])
   elif cfg.get('profit_target_pct',0)>0 and float(row.high)>=z.entry_price*(1+cfg['profit_target_pct']):reason='Profit target';price=z.entry_price*(1+cfg['profit_target_pct'])*(1-cfg['slippage_pct'])
   elif cfg['max_holding_days'] and z.holding_days>=cfg['max_holding_days']:reason='Max holding';price=float(row.close)*(1-cfg['slippage_pct'])
   if reason:
    proc=z.shares*price;cash+=proc;profit=proc-z.initial_value;trades.append({**asdict(z),'exit_date':d,'exit_price':price,'exit_reason':reason,'profit':profit,'return_pct':profit/z.initial_value*100});del pos[s]
  for key,(c,left) in list(watch.items()):
   if d not in bars[c.symbol].index:continue
   row=bars[c.symbol].loc[d]
   if float(row.close)<c.pattern_stop or left<=0:del watch[key];continue
   avg=float(bars[c.symbol].loc[:d].volume.tail(50).mean());vr=float(row.volume/avg) if avg else 0
   if float(row.close)>c.pivot and vr>=cfg['signal']['breakout_volume_multiple']:
    nd=[x for x in bars[c.symbol].index if x>d and x.date()<=end]
    if nd:pending.setdefault(nd[0],[]).append(Candidate(d,c.symbol,'WATCH_TO_BREAKOUT',c.setup,c.pivot,c.pattern_stop,c.score,vr,c.details))
    del watch[key]
   else:watch[key]=(c,left-1)
  for c in by.get(d,[]):
   if c.strategy=='WATCH_TO_BREAKOUT':watch[(c.symbol,c.setup)]=(c,int(cfg['watch_tracking_days']));continue
   nd=[x for x in bars[c.symbol].index if x>d and x.date()<=end]
   if nd:pending.setdefault(nd[0],[]).append(c)
  for c in sorted(pending.pop(d,[]),key=lambda x:x.score,reverse=True):
   if c.symbol in pos:skipped.append({**asdict(c),'reason':'Position already open'});continue
   if len(pos)>=cfg['max_positions']:skipped.append({**asdict(c),'reason':'Maximum positions'});continue
   entry=float(bars[c.symbol].loc[d].open)*(1+cfg['slippage_pct']);shares=math.floor(cfg['position_size']/entry);cost=shares*entry
   if shares<1 or cost>cash:skipped.append({**asdict(c),'reason':'Insufficient cash'});continue
   cash-=cost;pos[c.symbol]=Position(c.symbol,c.strategy,c.setup,c.signal_date,d,entry,shares,cost,entry*(1-cfg['stop_loss_pct']),c.pivot,c.score)
  curve.append({'date':d,'portfolio':cash+sum(z.shares*float(bars[s].loc[:d].close.iloc[-1]) for s,z in pos.items()),'cash':cash,'open_positions':len(pos)})
 if dates:
  for s,z in pos.items():
   px=float(bars[s].loc[:dates[-1]].close.iloc[-1])*(1-cfg['slippage_pct']);profit=z.shares*px-z.initial_value;trades.append({**asdict(z),'exit_date':dates[-1],'exit_price':px,'exit_reason':'End of backtest','profit':profit,'return_pct':profit/z.initial_value*100})
 return pd.DataFrame(trades),pd.DataFrame(curve).set_index('date'),pd.DataFrame(skipped)
def add_benchmarks(eq,bars,capital):
 o=eq.copy()
 for s in ['SPY','QQQ']:
  if s in bars:
   p=bars[s].close.reindex(o.index).ffill();o[s]=capital*p/p.dropna().iloc[0]
 return o
def summary(t,e,sk,capital):
 end=float(e.portfolio.iloc[-1]) if len(e) else capital;peak=e.portfolio.cummax() if len(e) else pd.Series(dtype=float);dd=float((e.portfolio/peak-1).min()*100) if len(e) else 0;wins=int((t.profit>0).sum()) if len(t) else 0
 return pd.DataFrame([{'starting_capital':capital,'ending_value':end,'net_profit':end-capital,'portfolio_return_pct':(end/capital-1)*100,'trades':len(t),'wins':wins,'losses':len(t)-wins,'win_rate_pct':100*wins/len(t) if len(t) else 0,'average_trade_return_pct':t.return_pct.mean() if len(t) else 0,'median_trade_return_pct':t.return_pct.median() if len(t) else 0,'largest_winner':t.profit.max() if len(t) else 0,'largest_loser':t.profit.min() if len(t) else 0,'maximum_drawdown_pct':dd,'average_holding_days':t.holding_days.mean() if len(t) else 0,'skipped_signals':len(sk)}])
