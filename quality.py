from dataclasses import replace
import numpy as np
import pandas as pd

def _rank_table(bars, lookback=126):
    return pd.DataFrame({s:x.close/x.close.shift(lookback)-1 for s,x in bars.items() if s not in {'SPY','QQQ'}}).rank(axis=1,pct=True)

def rescore(raw,bars,cfg):
    rs=_rank_table(bars,int(cfg['trend']['rs_lookback'])); out=[]
    for c in raw:
        if c.symbol not in bars or c.signal_date not in bars[c.symbol].index: continue
        x=bars[c.symbol]; i=x.index.get_loc(c.signal_date)
        if not isinstance(i,(int,np.integer)) or i<60: continue
        row=x.iloc[i]; vol50=float(x.volume.iloc[max(0,i-49):i+1].mean()); vr=float(row.volume/vol50) if vol50 else 0
        r=float(rs.loc[c.signal_date,c.symbol]) if c.signal_date in rs.index and c.symbol in rs and pd.notna(rs.loc[c.signal_date,c.symbol]) else 0
        distance=max(0.0,float(row.close/c.pivot-1)) if c.pivot else 0
        setup={'Flat base breakout':18,'Cup with handle breakout':22,'Volatility contraction pattern':24,'Pullback to rising SMA50':18,'Healthy trend transition':10}.get(c.setup,10)
        score=setup + min(25,r*25) + min(20,max(0,vr-1)*20) + max(0,15-distance*300) + 10
        out.append(replace(c,score=round(float(score),1),volume_ratio=vr))
    return out

def build_confirmed_trend(raw,bars,min_score=60):
    out=[]
    for c in raw:
        if c.strategy!='TREND_ONLY' or c.score<min_score or c.symbol not in bars: continue
        x=bars[c.symbol]; later=x.index[x.index>c.signal_date]
        if len(later)<2: continue
        confirm=later[0]; signal=x.loc[c.signal_date]; row=x.loc[confirm]
        vol50=float(x.loc[:confirm].volume.tail(50).mean()); vr=float(row.volume/vol50) if vol50 else 0
        if float(row.close)>float(signal.high) and vr>=1.0:
            out.append(replace(c,signal_date=confirm,strategy='CONFIRMED_TREND',pivot=float(signal.high),score=min(100,c.score+8),volume_ratio=vr,details=c.details+'; next-day high and volume confirmation'))
    return out

def next_open_filter(cands,bars,max_extension=0.03):
    kept=[]
    for c in cands:
        if c.symbol not in bars: continue
        x=bars[c.symbol]; later=x.index[x.index>c.signal_date]
        if not len(later): continue
        op=float(x.loc[later[0]].open)
        if c.pivot and op>float(c.pivot)*(1+max_extension): continue
        kept.append(c)
    return kept

def candidate_cooldown(cands,days=20):
    result=[];last={}
    for c in sorted(cands,key=lambda z:(z.signal_date,-z.score)):
        if c.symbol in last and (c.signal_date-last[c.symbol]).days<days: continue
        result.append(c);last[c.symbol]=c.signal_date
    return result

def portfolios(raw,bars,cfg,min_score=60,max_extension=.03,cooldown_days=20):
    scored=rescore(raw,bars,cfg)
    good=lambda xs:candidate_cooldown(next_open_filter([c for c in xs if c.score>=min_score],bars,max_extension),cooldown_days)
    breakout=good([c for c in scored if c.strategy=='BUY' and c.setup!='Pullback to rising SMA50'])
    pullback=good([c for c in scored if c.setup=='Pullback to rising SMA50' and c.strategy in {'BUY','WATCH'}])
    watch=good([c for c in scored if c.strategy=='WATCH' and c.setup!='Pullback to rising SMA50'])
    confirmed=good(build_confirmed_trend(scored,bars,min_score))
    return {'Quality Combined':breakout+pullback+watch+confirmed,'Confirmed breakout':breakout,'SMA50 pullback':pullback,'WATCH immediate':watch,'Confirmed trend':confirmed}
