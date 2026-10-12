"""Additive weekly cash strategy. Existing strategy functions are preserved."""
import numpy as np
import pandas as pd
NAME = 'Dual Momentum - Weekly Cash'

def predictions(panel,bars,settings,engine):
    frames=[]
    for day in engine.rebalance_dates(panel):
        snapshot=panel[panel.date.eq(day)].copy()
        ranked=engine._weekly_cash_original_predictions(snapshot,bars,'Dual Momentum',settings)
        if ranked.empty:
            frames.append(pd.DataFrame([{'date':day,'strategy':NAME,'symbol':None,'prediction':np.nan,'cash_signal':True}]))
        else:
            ranked['strategy']=NAME;ranked['cash_signal']=False;frames.append(ranked)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def simulate(pred,bars,top_n=10,cost_bps=10,stop_pct=0):
    if pred.empty:return pd.DataFrame(),pd.DataFrame()
    dates=sorted(pd.to_datetime(pred.date.unique()));spy=bars['SPY'].sort_index()
    capital=1.;previous={};curves=[];trades=[]
    for i,day in enumerate(dates):
        execution=spy.index[spy.index>day]
        if not len(execution):continue
        entry_day=execution[0]
        final=i==len(dates)-1
        if final:boundary=spy.index[-1]
        else:
            future=spy.index[spy.index>dates[i+1]]
            if not len(future):continue
            boundary=future[0]
        if boundary<=entry_day:continue
        ranked=pred[pred.date.eq(day)].dropna(subset=['symbol','prediction']).sort_values(['prediction','symbol'],ascending=[False,True]).head(top_n)
        target={};survivors={};period=[];gross=0.;stop_weight=0.
        for _,row in ranked.iterrows():
            stock=bars.get(row.symbol)
            if stock is None or entry_day not in stock.index or boundary not in stock.index:continue
            entry=float(stock.loc[entry_day,'open']);exit_day=boundary
            exit_price=float(stock.loc[boundary,'close' if final else 'open']);reason='End of data' if final else 'Rotation';trigger=pd.NaT
            if not np.isfinite(entry) or entry<=0:continue
            if stop_pct:
                for t in stock.index[(stock.index>=entry_day)&(stock.index<boundary)]:
                    if float(stock.loc[t,'close'])<=entry*(1-stop_pct/100):
                        next_bar=stock.index[stock.index>t]
                        if len(next_bar) and next_bar[0]<=boundary:
                            exit_day=next_bar[0];exit_price=float(stock.loc[exit_day,'open']);reason='Stop loss';trigger=t
                        break
            if not np.isfinite(exit_price) or exit_price<=0:continue
            weight=1/top_n;target[row.symbol]=weight
            if reason=='Stop loss':stop_weight+=weight
            else:survivors[row.symbol]=weight
            ret=exit_price/entry-1;gross+=weight*ret
            period.append({'strategy':NAME,'signal_date':day,'entry_date':entry_day,'exit_date':exit_day,'scheduled_next_signal':dates[i+1] if not final else pd.NaT,'stop_trigger_date':trigger,'exit_reason':reason,'symbol':row.symbol,'rank':len(period)+1,'weight_pct':100*weight,'score':row.prediction,'entry_price':entry,'exit_price':exit_price,'return_pct':100*ret,'weighted_contribution_pct':100*weight*ret})
        turnover=sum(abs(target.get(s,0)-previous.get(s,0)) for s in set(target)|set(previous))+stop_weight
        cost=turnover*cost_bps/10000;net=gross-cost;capital*=1+net
        curves.append({'date':day,'portfolio':capital,'period_return':net,'gross_return':gross,'transaction_cost':cost,'turnover':turnover,'holdings':len(target),'cash_weight':max(0.,1-sum(survivors.values())),'stopped_positions':sum(t['exit_reason']=='Stop loss' for t in period)})
        trades.extend(period);previous=survivors
    return (pd.DataFrame(curves).set_index('date') if curves else pd.DataFrame()),pd.DataFrame(trades)

def scanner(original,bars,settings,strategies=None,top_n=10,liquidity_floor=2_000_000):
    import momentum_engine as engine
    chosen=list(strategies or engine.STRATEGIES)
    mapped=list(dict.fromkeys('Dual Momentum' if s==NAME else s for s in chosen))
    day,leaders,consensus,master=original(bars,settings,mapped,top_n,liquidity_floor)
    if NAME not in chosen:return day,leaders,consensus,master
    if not master.empty:
        for suffix in (' score',' rank'):master[NAME+suffix]=master['Dual Momentum'+suffix]
    output={}
    for s in chosen:
        table=leaders.get('Dual Momentum' if s==NAME else s,pd.DataFrame()).copy()
        if not table.empty:table['strategy']=s
        output[s]=table
    if not master.empty:
        for suffix in (' score',' rank'):
            if 'Dual Momentum' not in chosen:master=master.drop(columns=['Dual Momentum'+suffix],errors='ignore')
        cols=[s+' rank' for s in chosen]
        master['strategies_ranked']=master[cols].notna().sum(axis=1)
        master['strategies_top_n']=sum((master[c]<=top_n).fillna(False).astype(int) for c in cols)
        master=master.sort_values(['strategies_top_n','symbol'],ascending=[False,True])
        consensus=master.loc[master.strategies_top_n>0,['symbol','strategies_top_n','strategies_ranked']+cols].copy()
    return day,output,consensus,master
