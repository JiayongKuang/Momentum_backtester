from __future__ import annotations
import numpy as np
import pandas as pd

STRATEGIES = [
    'Base Momentum',
    'Relative Strength Momentum',
    'Percentile Momentum',
    'Volatility Adjusted Momentum',
    'Persistence Momentum',
    'Regime Momentum',
    'Dual Momentum',
    'Momentum Ensemble',
]

def rebalance_dates(panel):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel['date']).unique()))
    if len(dates) == 0:
        return dates
    chosen = []
    for _, values in pd.Series(dates, index=dates).groupby(dates.to_period('W')):
        chosen.append(values.index.min())
    return pd.DatetimeIndex(chosen)

def _normalise_weights(w20, w60, w126):
    total = float(w20 + w60 + w126)
    if total <= 0:
        raise ValueError('Momentum weights must total more than zero.')
    return np.array([w20, w60, w126], dtype=float) / total

def _pct_rank(s):
    return s.rank(pct=True, method='average')

def _stock_above_ma(symbol, day, bars, period):
    x = bars.get(symbol)
    if x is None or day not in x.index:
        return False
    hist = x.loc[:day, 'close']
    if len(hist) < period:
        return False
    return float(hist.iloc[-1]) > float(hist.tail(period).mean())

def _spy_regime(day, bars, period):
    x = bars['SPY'].loc[:day, 'close']
    if len(x) < period:
        return False
    return float(x.iloc[-1]) > float(x.tail(period).mean())

def momentum_predictions(panel, bars, strategy, settings):
    w = _normalise_weights(settings['w20'], settings['w60'], settings['w126'])
    rows = []
    for day in rebalance_dates(panel):
        x = panel[panel.date.eq(day)].copy()
        x = x.dropna(subset=['symbol','entry_open','ret_20','ret_60','ret_126'])
        if x.empty:
            continue
        raw = w[0]*x.ret_20 + w[1]*x.ret_60 + w[2]*x.ret_126
        raw_rank = _pct_rank(raw)
        rs_raw = w[0]*x.rs_spy20 + w[1]*x.rs_spy60 + w[2]*(x.ret_126 - x.ret_126.median())
        rs_rank = _pct_rank(rs_raw)
        percentile = w[0]*_pct_rank(x.ret_20) + w[1]*_pct_rank(x.ret_60) + w[2]*_pct_rank(x.ret_126)
        volatility = x.vol_20.clip(lower=0.01)
        vol_adjusted = raw / np.power(volatility, float(settings['vol_penalty']))
        persistence = (
            float(settings['persistence_momentum_weight'])*raw_rank
            + float(settings['persistence_weight'])*x.up_day_ratio20.clip(0,1)
        ) / max(1.0, float(settings['persistence_momentum_weight'] + settings['persistence_weight']))
        rs_score = (
            float(settings['rs_momentum_weight'])*raw_rank
            + float(settings['rs_weight'])*rs_rank
        ) / max(1.0, float(settings['rs_momentum_weight'] + settings['rs_weight']))
        components = pd.DataFrame({
            'base': raw_rank,
            'rs': _pct_rank(rs_score),
            'percentile': _pct_rank(percentile),
            'vol_adj': _pct_rank(vol_adjusted),
            'persistence': _pct_rank(persistence),
        }, index=x.index)
        spy_ok = _spy_regime(day, bars, int(settings['regime_period']))
        if strategy == 'Base Momentum': prediction = raw
        elif strategy == 'Relative Strength Momentum': prediction = rs_score
        elif strategy == 'Percentile Momentum': prediction = percentile
        elif strategy == 'Volatility Adjusted Momentum': prediction = vol_adjusted
        elif strategy == 'Persistence Momentum': prediction = persistence
        elif strategy == 'Regime Momentum':
            if not spy_ok: continue
            prediction = raw
        elif strategy == 'Dual Momentum':
            if not spy_ok: continue
            eligible = x.symbol.map(lambda s: _stock_above_ma(s, day, bars, int(settings['stock_ma_period'])))
            x=x.loc[eligible].copy(); components=components.loc[x.index]; raw=raw.loc[x.index]
            if x.empty: continue
            prediction=raw
        elif strategy == 'Momentum Ensemble': prediction = components.mean(axis=1)
        else: raise ValueError(f'Unknown strategy: {strategy}')
        x['prediction'] = prediction
        x = x.dropna(subset=['prediction'])
        if x.empty: continue
        x['strategy'] = strategy
        x['model_train_end'] = pd.NaT
        x['base_rank'] = components.loc[x.index,'base']
        x['rs_rank'] = components.loc[x.index,'rs']
        x['percentile_rank'] = components.loc[x.index,'percentile']
        x['vol_adj_rank'] = components.loc[x.index,'vol_adj']
        x['persistence_rank'] = components.loc[x.index,'persistence']
        rows.append(x)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def _next_open(x, after_day):
    future = x.index[x.index > after_day]
    if not len(future):
        return None, None
    d = future[0]
    return d, float(x.loc[d, 'open'])

def _stopped_exit(stock, entry_day, entry_price, planned_exit, stop_pct):
    """Close-based stop; exit at next available open, never the trigger close."""
    if not stop_pct:
        return planned_exit, float(stock.loc[planned_exit, 'open']), None, None
    threshold = entry_price * (1 - stop_pct / 100.0)
    for day in stock.index[(stock.index >= entry_day) & (stock.index < planned_exit)]:
        if float(stock.loc[day, 'close']) <= threshold:
            following = stock.index[stock.index > day]
            if len(following) and following[0] <= planned_exit:
                exit_day = following[0]
                return exit_day, float(stock.loc[exit_day, 'open']), 'Stop loss', day
    return planned_exit, float(stock.loc[planned_exit, 'open']), None, None


def portfolio_from_rankings_with_stop(pred, bars, top_n=10, cost_bps=10, stop_pct=0):
    """Simulate next-open close stops and hold stopped allocations in cash until rotation.

    The existing no-stop simulator is intentionally left unchanged. For stop tests,
    all stocks in a signal period use the same scheduled next signal boundary.
    """
    if not stop_pct:
        return portfolio_from_rankings(pred, bars, top_n, cost_bps)
    if stop_pct not in (5, 10, 20):
        raise ValueError('Stop loss must be Off, 5%, 10%, or 20%.')
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    signal_dates = sorted(pd.to_datetime(pred.date).unique())
    capital = 1.0
    prior_weights = {}
    curve, holdings = [], []
    for i, day in enumerate(signal_dates):
        nxt = signal_dates[i + 1] if i + 1 < len(signal_dates) else None
        candidates = pred[pred.date.eq(day)].dropna(subset=['prediction']).nlargest(top_n, 'prediction')
        selected = []
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            stock = bars.get(row.symbol)
            if stock is None or stock.empty:
                continue
            entry_day, entry = _next_open(stock, day)
            if entry_day is None or not np.isfinite(entry) or entry <= 0:
                continue
            if nxt is None:
                remaining = stock.index[stock.index > entry_day]
                if not len(remaining):
                    continue
                planned_exit = remaining[-1]
                final_period = True
            else:
                planned_exit, _ = _next_open(stock, nxt)
                if planned_exit is None:
                    continue
                final_period = False
            exit_day, exit_price, reason, trigger_day = _stopped_exit(
                stock, entry_day, entry, planned_exit, stop_pct)
            if reason is None and final_period:
                exit_price = float(stock.loc[planned_exit, 'close'])
                reason = 'End of data'
            elif reason is None:
                reason = 'Rotation'
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue
            selected.append((row, rank, entry_day, entry, exit_day, exit_price, reason, trigger_day))
        weight = 1.0 / top_n
        new_weights = {r.symbol: weight for r, _, _, _, _, _, reason, _ in selected
                       if reason != 'Stop loss'}
        turnover = sum(abs(new_weights.get(s, 0) - prior_weights.get(s, 0))
                       for s in set(new_weights) | set(prior_weights))
        # Retained positions have zero turnover; stops are additional exits.
        stopped_weight = sum(weight for r, _, _, _, _, _, reason, _ in selected
                             if reason == 'Stop loss')
        transaction_cost = (turnover + stopped_weight) * float(cost_bps) / 10000.0
        gross_return = 0.0
        for row, rank, entry_day, entry, exit_day, exit_price, reason, trigger_day in selected:
            ret = exit_price / entry - 1.0
            gross_return += weight * ret
            holdings.append({
                'strategy': row.strategy, 'signal_date': day, 'entry_date': entry_day,
                'exit_date': exit_day, 'stop_trigger_date': trigger_day,
                'scheduled_next_signal': nxt, 'exit_reason': reason,
                'rank': rank, 'symbol': row.symbol, 'weight_pct': weight * 100,
                'score': row.prediction, 'entry_price': entry, 'exit_price': exit_price,
                'return_pct': ret * 100, 'weighted_contribution_pct': weight * ret * 100,
                'base_rank': row.get('base_rank', np.nan),
                'rs_rank': row.get('rs_rank', np.nan),
                'percentile_rank': row.get('percentile_rank', np.nan),
                'vol_adj_rank': row.get('vol_adj_rank', np.nan),
                'persistence_rank': row.get('persistence_rank', np.nan),
            })
        net_return = gross_return - transaction_cost
        capital *= 1 + net_return
        curve.append({'date': day, 'portfolio': capital, 'period_return': net_return,
                      'gross_return': gross_return, 'transaction_cost': transaction_cost,
                      'turnover': turnover + stopped_weight, 'holdings': len(selected),
                      'stopped_positions': sum(v[6] == 'Stop loss' for v in selected),
                      'cash_weight': max(0.0, 1.0 - len(new_weights) / top_n)})
        prior_weights = new_weights
    return pd.DataFrame(curve).set_index('date'), pd.DataFrame(holdings)


def portfolio_from_rankings(pred, bars, top_n=10, cost_bps=10):
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    signal_dates = sorted(pd.to_datetime(pred.date).unique())
    capital=1.0; curve=[]; holdings=[]; prior_weights={}
    for i, day in enumerate(signal_dates):
        nxt = signal_dates[i+1] if i+1 < len(signal_dates) else None
        candidates = pred[pred.date.eq(day)].dropna(subset=['prediction']).nlargest(top_n,'prediction').copy()
        selected=[]
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            stock=bars.get(row.symbol)
            if stock is None: continue
            entry_day, entry = _next_open(stock, day)
            if entry_day is None: continue
            if nxt is None:
                future=stock.index[stock.index>entry_day]
                if not len(future): continue
                exit_day=future[-1]; exit_price=float(stock.loc[exit_day,'close'])
            else:
                exit_day, exit_price = _next_open(stock, nxt)
                if exit_day is None: continue
            selected.append((row,rank,entry_day,entry,exit_day,exit_price))
        target_weight = 1.0/top_n
        new_weights={r.symbol:target_weight for r,_,_,_,_,_ in selected}
        symbols=set(prior_weights)|set(new_weights)
        turnover=sum(abs(new_weights.get(s,0)-prior_weights.get(s,0)) for s in symbols)
        transaction_cost=turnover*float(cost_bps)/10000.0
        gross_return=0.0
        for row, rank, entry_day, entry, exit_day, exit_price in selected:
            stock_return=exit_price/entry-1
            gross_return += target_weight*stock_return
            holdings.append({
                'strategy':row.strategy,'signal_date':day,'entry_date':entry_day,'exit_date':exit_day,
                'rank':rank,'symbol':row.symbol,'weight_pct':target_weight*100,'score':row.prediction,
                'return_pct':stock_return*100,'weighted_contribution_pct':target_weight*stock_return*100,
                'base_rank':row.get('base_rank',np.nan),'rs_rank':row.get('rs_rank',np.nan),
                'percentile_rank':row.get('percentile_rank',np.nan),'vol_adj_rank':row.get('vol_adj_rank',np.nan),
                'persistence_rank':row.get('persistence_rank',np.nan)})
        net_return=gross_return-transaction_cost
        capital*=1+net_return
        curve.append({'date':day,'portfolio':capital,'period_return':net_return,'gross_return':gross_return,
                      'transaction_cost':transaction_cost,'turnover':turnover,'holdings':len(selected),
                      'cash_weight':max(0.0,1.0-len(selected)/top_n)})
        prior_weights=new_weights
    return pd.DataFrame(curve).set_index('date'), pd.DataFrame(holdings)

def benchmark_curve(bars, symbol, dates):
    p=bars[symbol].close.reindex(pd.DatetimeIndex(dates),method='ffill').dropna()
    return p/p.iloc[0]

def strategy_summary(curve, holdings, name):
    if curve.empty:return {'strategy':name}
    r=curve.period_return.dropna();years=max((curve.index[-1]-curve.index[0]).days/365.25,.01)
    cagr=curve.portfolio.iloc[-1]**(1/years)-1;dd=curve.portfolio/curve.portfolio.cummax()-1
    downside=r[r<0].std();sortino=(r.mean()/downside*np.sqrt(52)) if downside and downside>0 else np.nan
    return {'strategy':name,'ending_multiple':curve.portfolio.iloc[-1],'CAGR_pct':cagr*100,
      'annual_vol_pct':r.std()*np.sqrt(52)*100 if len(r)>1 else 0,
      'Sharpe':r.mean()/r.std()*np.sqrt(52) if len(r)>1 and r.std()>0 else 0,
      'Sortino':sortino,'max_drawdown_pct':dd.min()*100,'rebalance_periods':len(curve),
      'average_period_return_pct':r.mean()*100,'positive_periods_pct':(r>0).mean()*100,
      'average_turnover_pct':curve.turnover.mean()*100,'average_cash_pct':curve.cash_weight.mean()*100}

def contribution_table(holdings):
    if holdings.empty:return pd.DataFrame()
    return holdings.groupby(['strategy','symbol'],as_index=False).agg(
        appearances=('symbol','size'),contribution_pct=('weighted_contribution_pct','sum'),
        average_holding_return_pct=('return_pct','mean'),win_rate_pct=('return_pct',lambda x:(x>0).mean()*100)
    ).sort_values(['strategy','contribution_pct'],ascending=[True,False])

def scan_latest_rankings(bars, settings, strategies=None, top_n=10, liquidity_floor=2_000_000):
    """Rank the latest available complete daily bar for every eligible universe symbol."""
    strategies = list(strategies or STRATEGIES)
    if 'SPY' not in bars or bars['SPY'].empty:
        raise ValueError('SPY data is required for the latest scan.')
    spy = bars['SPY'].sort_index()
    scan_day = pd.Timestamp(spy.index.max())
    spy_hist = spy.loc[:scan_day]
    if len(spy_hist) < 127:
        raise ValueError('SPY does not have enough history for the 126-day scan.')
    spy_ret20 = float(spy_hist.close.iloc[-1] / spy_hist.close.iloc[-21] - 1)
    spy_ret60 = float(spy_hist.close.iloc[-1] / spy_hist.close.iloc[-61] - 1)
    spy_ok = _spy_regime(scan_day, bars, int(settings['regime_period']))
    rows = []
    for symbol, frame in bars.items():
        if symbol in {'SPY','QQQ'} or frame is None or frame.empty:
            continue
        x = frame.sort_index().loc[:scan_day]
        if len(x) < max(127, int(settings['stock_ma_period'])):
            continue
        close = x.close
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1)
        ret60 = float(close.iloc[-1] / close.iloc[-61] - 1)
        ret126 = float(close.iloc[-1] / close.iloc[-127] - 1)
        daily = close.pct_change()
        vol20 = float(daily.tail(20).std() * np.sqrt(252))
        up20 = float((daily.tail(20) > 0).mean())
        dollar_volume20 = float((x.close * x.volume).tail(20).mean())
        if not np.isfinite(dollar_volume20) or dollar_volume20 < float(liquidity_floor):
            continue
        stock_above_ma = _stock_above_ma(symbol, scan_day, bars, int(settings['stock_ma_period']))
        rows.append({
            'scan_date': scan_day, 'symbol': symbol, 'close': float(close.iloc[-1]),
            'ret_20': ret20, 'ret_60': ret60, 'ret_126': ret126,
            'rs_spy20': ret20-spy_ret20, 'rs_spy60': ret60-spy_ret60,
            'vol_20': vol20, 'up_day_ratio20': up20,
            'dollar_volume20': dollar_volume20, 'stock_above_ma': bool(stock_above_ma),
            'spy_regime_pass': bool(spy_ok),
        })
    universe = pd.DataFrame(rows)
    if universe.empty:
        return scan_day, {}, pd.DataFrame(), pd.DataFrame()
    w = _normalise_weights(settings['w20'], settings['w60'], settings['w126'])
    raw = w[0]*universe.ret_20 + w[1]*universe.ret_60 + w[2]*universe.ret_126
    raw_rank = _pct_rank(raw)
    rs_raw = w[0]*universe.rs_spy20 + w[1]*universe.rs_spy60 + w[2]*(universe.ret_126-universe.ret_126.median())
    rs_rank = _pct_rank(rs_raw)
    percentile = w[0]*_pct_rank(universe.ret_20) + w[1]*_pct_rank(universe.ret_60) + w[2]*_pct_rank(universe.ret_126)
    vol_adjusted = raw / np.power(universe.vol_20.clip(lower=.01), float(settings['vol_penalty']))
    persistence = (
        float(settings['persistence_momentum_weight'])*raw_rank
        + float(settings['persistence_weight'])*universe.up_day_ratio20.clip(0,1)
    ) / max(1., float(settings['persistence_momentum_weight']+settings['persistence_weight']))
    rs_score = (
        float(settings['rs_momentum_weight'])*raw_rank
        + float(settings['rs_weight'])*rs_rank
    ) / max(1., float(settings['rs_momentum_weight']+settings['rs_weight']))
    score_map = {
        'Base Momentum': raw,
        'Relative Strength Momentum': rs_score,
        'Percentile Momentum': percentile,
        'Volatility Adjusted Momentum': vol_adjusted,
        'Persistence Momentum': persistence,
        'Regime Momentum': raw if spy_ok else pd.Series(np.nan,index=universe.index),
        'Dual Momentum': raw.where(universe.stock_above_ma) if spy_ok else pd.Series(np.nan,index=universe.index),
    }
    components = pd.DataFrame({
        'base': _pct_rank(raw), 'rs': _pct_rank(rs_score),
        'percentile': _pct_rank(percentile), 'vol_adj': _pct_rank(vol_adjusted),
        'persistence': _pct_rank(persistence),
    })
    score_map['Momentum Ensemble'] = components.mean(axis=1)
    leaders = {}; master = universe.copy()
    for strategy in strategies:
        score = score_map[strategy]
        rank = score.rank(ascending=False,method='min')
        master[f'{strategy} score'] = score
        master[f'{strategy} rank'] = rank.astype('Int64')
        eligible = master.loc[score.notna()].copy()
        eligible['strategy'] = strategy
        eligible['score'] = score.loc[eligible.index]
        eligible['rank'] = rank.loc[eligible.index].astype(int)
        leaders[strategy] = eligible.nsmallest(top_n,'rank').sort_values('rank')
    rank_cols = [f'{s} rank' for s in strategies]
    master['strategies_ranked'] = master[rank_cols].notna().sum(axis=1)
    master['strategies_top_n'] = sum((master[c] <= top_n).fillna(False).astype(int) for c in rank_cols)
    master = master.sort_values(['strategies_top_n','strategies_ranked','symbol'],ascending=[False,False,True])
    consensus = master.loc[master.strategies_top_n>0, ['symbol','strategies_top_n','strategies_ranked']+rank_cols].copy()
    return scan_day, leaders, consensus, master
