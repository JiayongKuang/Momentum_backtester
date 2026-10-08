from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

FEATURES=[
'ret_5','ret_20','ret_60','ret_126','ret_252','accel_5_20','accel_20_60',
'dist_sma10','dist_sma20','dist_sma50','dist_sma100','dist_sma200',
'slope_20','slope_50','slope_200','dist_high20','dist_high60','dist_high252','dist_low20','dist_low252',
'vol_5','vol_20','vol_60','vol_ratio20','vol_ratio50','dollar_volume20','atr_pct','drawdown20','drawdown60','up_day_ratio20',
'rs_spy20','rs_spy60','rs_qqq20','rs_qqq60','breadth20','breadth50','cross_dispersion20'
]

def _atr(x,n=14):
 prev=x.close.shift();tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1);return tr.rolling(n).mean()/x.close

def build_panel(bars,start,end,horizon=20,liquidity_floor=2_000_000):
 stock={s:x.sort_index().copy() for s,x in bars.items() if s not in {'SPY','QQQ'}}
 spy=bars['SPY'].sort_index();qqq=bars['QQQ'].sort_index();frames=[];above20={};above50={};r20={}
 for s,x in stock.items():
  above20[s]=(x.close>x.close.rolling(20).mean()).astype(float);above50[s]=(x.close>x.close.rolling(50).mean()).astype(float);r20[s]=x.close.pct_change(20)
 breadth20=pd.DataFrame(above20).mean(axis=1);breadth50=pd.DataFrame(above50).mean(axis=1);disp=pd.DataFrame(r20).std(axis=1)
 spy20=spy.close.pct_change(20);spy60=spy.close.pct_change(60);qqq20=qqq.close.pct_change(20);qqq60=qqq.close.pct_change(60)
 for s,x in stock.items():
  d=pd.DataFrame(index=x.index);d['date']=x.index;d['symbol']=s
  for n in [5,20,60,126,252]:d[f'ret_{n}']=x.close.pct_change(n)
  d['accel_5_20']=d.ret_5-d.ret_20;d['accel_20_60']=d.ret_20-d.ret_60
  for n in [10,20,50,100,200]:d[f'dist_sma{n}']=x.close/x.close.rolling(n).mean()-1
  for n in [20,50,200]:d[f'slope_{n}']=x.close.rolling(n).mean()/x.close.rolling(n).mean().shift(20)-1
  for n in [20,60,252]:d[f'dist_high{n}']=x.close/x.high.rolling(n).max()-1
  for n in [20,252]:d[f'dist_low{n}']=x.close/x.low.rolling(n).min()-1
  daily=x.close.pct_change()
  for n in [5,20,60]:d[f'vol_{n}']=daily.rolling(n).std()*np.sqrt(252)
  d['vol_ratio20']=x.volume/x.volume.rolling(20).mean();d['vol_ratio50']=x.volume/x.volume.rolling(50).mean();d['dollar_volume20']=(x.close*x.volume).rolling(20).mean();d['atr_pct']=_atr(x);d['drawdown20']=x.close/x.close.rolling(20).max()-1;d['drawdown60']=x.close/x.close.rolling(60).max()-1;d['up_day_ratio20']=(daily>0).rolling(20).mean()
  d['rs_spy20']=d.ret_20-spy20.reindex(x.index);d['rs_spy60']=d.ret_60-spy60.reindex(x.index);d['rs_qqq20']=d.ret_20-qqq20.reindex(x.index);d['rs_qqq60']=d.ret_60-qqq60.reindex(x.index);d['breadth20']=breadth20.reindex(x.index);d['breadth50']=breadth50.reindex(x.index);d['cross_dispersion20']=disp.reindex(x.index)
  entry=x.open.shift(-1);future=x.close.shift(-(horizon+1));raw=future/entry-1
  all_raw=pd.DataFrame({k:(v.close.shift(-(horizon+1))/v.open.shift(-1)-1) for k,v in stock.items()})
  # relative target is completed after concatenation to avoid recomputing per-stock medians here
  d['future_return']=raw;d['entry_open']=entry;d['close']=x.close;frames.append(d)
 panel=pd.concat(frames,ignore_index=True);panel['date']=pd.to_datetime(panel.date);panel=panel[(panel.date.dt.date>=start)&(panel.date.dt.date<=end)&(panel.dollar_volume20>=liquidity_floor)]
 panel['target']=panel.future_return-panel.groupby('date').future_return.transform('median')
 return panel.replace([np.inf,-np.inf],np.nan)

def rebalance_dates(panel,weekday=0):
 dates=pd.DatetimeIndex(sorted(panel.date.unique()));chosen=[]
 for _,g in pd.Series(dates,index=dates).groupby(dates.to_period('W')):
  monday=pd.Timestamp(g.index.min()).to_period('W').start_time
  chosen.append(min(g.index,key=lambda x:abs((x-monday).days)))
 return pd.DatetimeIndex(chosen)

def _fit_predict(train,test,model_name):
 from sklearn.impute import SimpleImputer
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 from sklearn.ensemble import HistGradientBoostingRegressor
 Xtr=train[FEATURES];y=train.target;Xte=test[FEATURES]
 if model_name=='Linear ranker':model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=20.0))
 else:model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=120,max_depth=3,learning_rate=.04,l2_regularization=5,min_samples_leaf=50,random_state=42))
 model.fit(Xtr,y);return model.predict(Xte),model

def walk_forward_predictions(panel,model_name,train_years=3,horizon=20,retrain='Quarterly'):
 dates=rebalance_dates(panel);rows=[];last_period=None;model=None;train_end=None
 for d in dates:
  period=d.to_period('Q' if retrain=='Quarterly' else 'M')
  test=panel[panel.date.eq(d)].dropna(subset=['entry_open']).copy()
  if test.empty:continue
  if model_name=='Simple momentum':
   test['prediction']=.25*test.ret_20+.35*test.ret_60+.40*test.ret_126;test['model_train_end']=pd.NaT;rows.append(test);continue
  if model is None or period!=last_period:
   unique=pd.DatetimeIndex(sorted(panel.date.unique()));prior=unique[unique<d]
   if len(prior)<=horizon:continue
   embargo_end=prior[-(horizon+1)]
   train_start=d-pd.DateOffset(years=train_years)
   train=panel[(panel.date>=train_start)&(panel.date<=embargo_end)].dropna(subset=['target'])
   if len(train)<2000:continue
   _,model=_fit_predict(train,train.iloc[:1],model_name);train_end=embargo_end;last_period=period
  test['prediction']=model.predict(test[FEATURES]);test['model_train_end']=train_end;rows.append(test)
 return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def information_coefficient(pred):
 return pred.groupby('date').apply(lambda x:x.prediction.corr(x.target,method='spearman'),include_groups=False).rename('rank_ic').reset_index()

def portfolio_from_predictions(pred,bars,top_n=20,cost_bps=10):
 if pred.empty:return pd.DataFrame(),pd.DataFrame()
 dates=sorted(pred.date.unique());capital=1.0;curve=[];holdings=[]
 for i,d in enumerate(dates):
  nxt=dates[i+1] if i+1<len(dates) else None
  picks=pred[pred.date.eq(d)].nlargest(top_n,'prediction').copy()
  rets=[]
  for _,r in picks.iterrows():
   x=bars[r.symbol];future=x.index[x.index>d]
   if not len(future):continue
   entry_day=future[0];exit_candidates=x.index[x.index>entry_day]
   if nxt is not None:
    exit_candidates=exit_candidates[exit_candidates>=nxt]
   if not len(exit_candidates):continue
   exit_day=exit_candidates[0];entry=float(x.loc[entry_day].open);exitp=float(x.loc[exit_day].open);ret=exitp/entry-1-2*cost_bps/10000
   rets.append(ret);holdings.append({'signal_date':d,'entry_date':entry_day,'exit_date':exit_day,'symbol':r.symbol,'prediction':r.prediction,'actual_relative_20d':r.target,'return_pct':ret*100,'model_train_end':r.model_train_end})
  if rets:capital*=1+float(np.mean(rets))
  curve.append({'date':d,'portfolio':capital,'holdings':len(rets),'cash_weight':max(0,1-len(rets)/top_n)})
 return pd.DataFrame(curve).set_index('date'),pd.DataFrame(holdings)

def benchmark_curve(bars,symbol,dates):
 x=bars[symbol];p=x.close.reindex(dates,method='ffill');return p/p.dropna().iloc[0]

def summary(curve,holdings,name):
 if curve.empty:return {'model':name}
 r=curve.portfolio.pct_change().dropna();years=max((curve.index[-1]-curve.index[0]).days/365.25,.01);cagr=curve.portfolio.iloc[-1]**(1/years)-1;dd=curve.portfolio/curve.portfolio.cummax()-1
 return {'model':name,'ending_multiple':curve.portfolio.iloc[-1],'CAGR_pct':cagr*100,'annual_vol_pct':r.std()*np.sqrt(52)*100 if len(r) else 0,'Sharpe':r.mean()/r.std()*np.sqrt(52) if r.std()>0 else 0,'max_drawdown_pct':dd.min()*100,'trades':len(holdings),'average_trade_pct':holdings.return_pct.mean() if len(holdings) else 0,'median_trade_pct':holdings.return_pct.median() if len(holdings) else 0,'win_rate_pct':(holdings.return_pct>0).mean()*100 if len(holdings) else 0,'average_cash_pct':curve.cash_weight.mean()*100}

# Behavioural strategy layer. Each sleeve has a separate hypothesis, eligible
# population, feature set and walk-forward model. No sleeve is a hard-coded
# recommendation; each is independently validated out of sample.
BEHAVIOURAL_SPECS = {
    'Continuation': {
        'features': ['ret_20','ret_60','ret_126','ret_252','rs_spy20','rs_spy60','rs_qqq20','rs_qqq60','dist_high60','dist_high252','drawdown60','up_day_ratio20','breadth50','cross_dispersion20'],
        'eligibility': 'continuation', 'model': 'linear'},
    'Early acceleration': {
        'features': ['ret_5','ret_20','ret_60','accel_5_20','accel_20_60','slope_20','slope_50','vol_ratio20','vol_ratio50','dist_high20','drawdown20','rs_spy20','breadth20','cross_dispersion20'],
        'eligibility': 'acceleration', 'model': 'tree'},
    'Recovery / reversal': {
        'features': ['ret_5','ret_20','ret_60','accel_5_20','accel_20_60','dist_sma20','dist_sma50','dist_sma200','dist_low20','dist_low252','drawdown20','drawdown60','vol_20','atr_pct','breadth20','cross_dispersion20'],
        'eligibility': 'recovery', 'model': 'tree'},
    'Low-volatility leadership': {
        'features': ['ret_20','ret_60','ret_126','vol_20','vol_60','atr_pct','drawdown20','drawdown60','up_day_ratio20','rs_spy60','rs_qqq60','dist_high252','breadth50'],
        'eligibility': 'low_vol', 'model': 'linear'},
    'High-volatility momentum': {
        'features': ['ret_5','ret_20','ret_60','ret_126','vol_5','vol_20','vol_60','atr_pct','vol_ratio20','dollar_volume20','rs_spy20','rs_qqq20','cross_dispersion20','breadth20'],
        'eligibility': 'high_vol', 'model': 'tree'},
}

def _eligible(frame, rule):
    if frame.empty: return frame
    daily_med_vol = frame.groupby('date').vol_20.transform('median')
    daily_q60_vol = frame.groupby('date').vol_20.transform(lambda x:x.quantile(.60))
    if rule == 'continuation': mask=(frame.ret_60>0)
    elif rule == 'acceleration': mask=(frame.accel_5_20>0)|(frame.accel_20_60>0)
    elif rule == 'recovery': mask=(frame.drawdown60<=-.08)|(frame.ret_60<0)
    elif rule == 'low_vol': mask=(frame.ret_60>0)&(frame.vol_20<=daily_med_vol)
    elif rule == 'high_vol': mask=(frame.ret_20>0)&(frame.vol_20>=daily_q60_vol)
    else: mask=pd.Series(True,index=frame.index)
    return frame[mask.fillna(False)].copy()

def _behaviour_model(features, kind):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    if kind=='linear':
        return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=25.0))
    return make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=100,max_depth=3,learning_rate=.04,l2_regularization=8,min_samples_leaf=75,random_state=42))

def behavioural_walk_forward(panel, sleeve, train_years=3, horizon=20, retrain='Quarterly'):
    spec=BEHAVIOURAL_SPECS[sleeve];features=spec['features'];dates=rebalance_dates(panel);rows=[];model=None;last_period=None;train_end=None
    for d in dates:
        period=d.to_period('Q' if retrain=='Quarterly' else 'M')
        test=_eligible(panel[panel.date.eq(d)].dropna(subset=['entry_open']),spec['eligibility'])
        if test.empty: continue
        if model is None or period!=last_period:
            prior=pd.DatetimeIndex(sorted(panel.loc[panel.date<d,'date'].unique()))
            if len(prior)<=horizon: continue
            embargo_end=prior[-(horizon+1)];train_start=d-pd.DateOffset(years=train_years)
            train=_eligible(panel[(panel.date>=train_start)&(panel.date<=embargo_end)].dropna(subset=['target']),spec['eligibility'])
            if len(train)<1200: continue
            model=_behaviour_model(features,spec['model']);model.fit(train[features],train.target);train_end=embargo_end;last_period=period
        test['prediction']=model.predict(test[features]);test['model_train_end']=train_end;test['sleeve']=sleeve;rows.append(test)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def ensemble_predictions(predictions_by_sleeve):
    ranked=[]
    for name,p in predictions_by_sleeve.items():
        if p.empty: continue
        q=p[['date','symbol','prediction','target','entry_open']].copy();q['rank']=q.groupby('date').prediction.rank(pct=True);q=q.rename(columns={'rank':name});ranked.append(q[['date','symbol',name]])
    if not ranked:return pd.DataFrame()
    wide=ranked[0]
    for q in ranked[1:]:wide=wide.merge(q,on=['date','symbol'],how='outer')
    sleeve_cols=[c for c in wide.columns if c not in {'date','symbol'}];wide['prediction']=wide[sleeve_cols].mean(axis=1,skipna=True);base=pd.concat([p for p in predictions_by_sleeve.values() if not p.empty],ignore_index=True).sort_values('date').drop_duplicates(['date','symbol'],keep='last')
    out=wide.merge(base.drop(columns=['prediction'],errors='ignore'),on=['date','symbol'],how='left');out['sleeve']='Behavioural ensemble';return out
