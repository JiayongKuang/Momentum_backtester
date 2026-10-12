from datetime import date,timedelta
from pathlib import Path
import hashlib,json,yaml,pandas as pd,plotly.express as px,streamlit as st
from buy_point_backtester.data import AlpacaStore,load_symbols,feature_cache_dir
from ml_rank_engine import build_panel
from momentum_engine import STRATEGIES,momentum_predictions,portfolio_from_rankings,benchmark_curve,strategy_summary,contribution_table,portfolio_from_rankings_with_stop,scan_latest_rankings
st.set_page_config(page_title='Momentum Research Lab v8',layout='wide')
st.title('Momentum Research Lab v8')
st.caption('Research only, not financial advice. V7 focuses on transparent momentum variants and removes machine-learning models from the active workflow.')
cfg=yaml.safe_load(Path('backtest_config.yaml').read_text())
with st.sidebar:
 from dotenv import dotenv_values
 import os
 _feed_default=str(dotenv_values('.env').get('ALPACA_DATA_FEED','sip')).strip().lower()
 _feed_choice=st.selectbox('Market data feed',['sip','iex'],index=0 if _feed_default!='iex' else 1)
 os.environ['ALPACA_DATA_FEED']=_feed_choice
 st.caption('Separate feed caches. Historical data only; no live trading orders.')
 st.header('Research period');today=date.today();start=st.date_input('Start',today-timedelta(days=3650));end=st.date_input('End',today)
 st.header('Portfolio');top_n=st.slider('Stocks held each week',1,50,10);cost=st.number_input('One-way transaction cost, basis points',0.,100.,10.,1.);liquidity=st.number_input('Minimum 20-day average dollar volume',0.,1_000_000_000.,2_000_000.,1_000_000.)
 st.header('Stop loss');stop_pct=st.selectbox('Stop loss per position (close trigger, next open exit)',[0,5,10,20],format_func=lambda v:'Off' if v==0 else f'{v}%');st.caption('Stopped positions remain cash until the next strategy rotation.')
 st.header('Momentum weights');w20=st.number_input('20-day weight %',0.,100.,25.,1.);w60=st.number_input('60-day weight %',0.,100.,35.,1.);w126=st.number_input('126-day weight %',0.,100.,40.,1.);weight_total=w20+w60+w126;st.caption(f'Total weight: {weight_total:.0f}%')
 st.header('Strategies');selected=st.multiselect('Compare momentum strategies',STRATEGIES,default=['Base Momentum','Percentile Momentum','Relative Strength Momentum','Regime Momentum','Dual Momentum','Momentum Ensemble'])
 st.header('Relative-strength mix');rs_mom=st.slider('Momentum contribution %',0,100,50);rs_weight=100-rs_mom;st.caption(f'RS contribution: {rs_weight}%')
 st.header('Volatility adjustment');vol_penalty=st.slider('Volatility penalty exponent',0.0,2.0,1.0,.1)
 st.header('Persistence mix');pers_mom=st.slider('Momentum contribution % ',0,100,50,key='pers_mom');pers_weight=100-pers_mom;st.caption(f'Persistence contribution: {pers_weight}%')
 st.header('Regime settings');regime_period=st.slider('SPY moving-average period',50,300,200,10);stock_ma_period=st.slider('Stock moving-average period',50,300,200,10)
 force_panel=st.checkbox('Rebuild feature dataset',False)
run=st.button('Run momentum comparison',type='primary',use_container_width=True)
scan=st.button("Scan latest universe rankings",use_container_width=True)
def universe_hash(symbols):return hashlib.sha256('|'.join(sorted(symbols)).encode()).hexdigest()[:16]
if run:
 if start>=end:st.error('Start must be before End.');st.stop()
 if not selected:st.error('Select at least one strategy.');st.stop()
 if weight_total<=0:st.error('Momentum weights must total more than zero.');st.stop()
 if abs(weight_total-100)>0.001:st.warning('Weights are automatically normalised because the total is not 100%.')
 syms=load_symbols();req=sorted(set(syms+['SPY','QQQ']));warm=start-timedelta(days=max(900,(max(regime_period,stock_ma_period)+300)*2));bar=st.progress(0,text='Loading market-data cache')
 try:bars=AlpacaStore().fetch(req,warm,end,int(cfg.get('batch_size',20)),lambda v,m:bar.progress(v or 0,text=m))
 except Exception as e:st.error(str(e));st.stop()
 valid={s:x for s,x in bars.items() if len(x)>=max(300,regime_period,stock_ma_period)}
 if 'SPY' not in valid or 'QQQ' not in valid:st.error('SPY or QQQ history is unavailable.');st.stop()
 cache=feature_cache_dir();cache.mkdir(parents=True,exist_ok=True);key=hashlib.sha256(json.dumps({'u':universe_hash(syms),'start':str(start),'end':str(end),'liq':liquidity,'feature_version':'v7.0'},sort_keys=True).encode()).hexdigest();panel_file=cache/f'momentum_panel_{key}.parquet'
 if panel_file.exists() and not force_panel:panel=pd.read_parquet(panel_file);panel['date']=pd.to_datetime(panel.date);panel_used=True
 else:bar.progress(.05,text='Building full-universe momentum feature panel');panel=build_panel(valid,start,end,20,liquidity);panel.to_parquet(panel_file,index=False);panel_used=False
 settings={'w20':w20,'w60':w60,'w126':w126,'rs_momentum_weight':rs_mom,'rs_weight':rs_weight,'vol_penalty':vol_penalty,'persistence_momentum_weight':pers_mom,'persistence_weight':pers_weight,'regime_period':regime_period,'stock_ma_period':stock_ma_period}
 curves={};reports=[];hold_sets=[]
 for i,name in enumerate(selected):
  bar.progress((i+.2)/len(selected),text=f'Running {name}');pred=momentum_predictions(panel,valid,name,settings);curve,holds=portfolio_from_rankings_with_stop(pred,valid,top_n,cost,stop_pct)
  if curve.empty:continue
  curves[name]=curve.portfolio;reports.append(strategy_summary(curve,holds,name));hold_sets.append(holds)
 dates=pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in curves.values()]))) if curves else pd.DatetimeIndex([]);comparison=pd.DataFrame(curves).reindex(dates).ffill()
 if len(dates):comparison['SPY']=benchmark_curve(valid,'SPY',dates);comparison['QQQ']=benchmark_curve(valid,'QQQ',dates)
 bar.empty();st.success('Reused cached momentum feature panel.' if panel_used else 'Built and cached the momentum feature panel.')
 report=pd.DataFrame(reports);holds=pd.concat(hold_sets,ignore_index=True) if hold_sets else pd.DataFrame();contrib=contribution_table(holds)
 tabs=st.tabs(['Performance','Leaderboard','Holdings','Contribution','Correlation','Definitions'])
 with tabs[0]:
  if not comparison.empty:
   comparison.index.name='date';m=comparison.reset_index().melt('date',var_name='Series',value_name='Growth of $1');st.plotly_chart(px.line(m,x='date',y='Growth of $1',color='Series'),use_container_width=True)
 with tabs[1]:
  sort_by=st.selectbox('Rank strategies by',['CAGR_pct','Sharpe','Sortino','max_drawdown_pct']);ascending=sort_by=='max_drawdown_pct';st.dataframe(report.sort_values(sort_by,ascending=ascending),use_container_width=True,hide_index=True)
 with tabs[2]:st.dataframe(holds,use_container_width=True,hide_index=True)
 with tabs[3]:
  st.caption('Contribution is the sum of each equal-weight holding contribution before portfolio-level turnover costs.');st.dataframe(contrib,use_container_width=True,hide_index=True)
 with tabs[4]:
  if not comparison.empty:st.dataframe(comparison.pct_change().corr(),use_container_width=True)
 with tabs[5]:
  st.markdown('''**Base Momentum:** weighted raw 20-, 60-, and 126-day returns.\n\n**Relative Strength Momentum:** blends the Base rank with performance relative to SPY.\n\n**Percentile Momentum:** combines cross-sectional percentile ranks rather than raw returns.\n\n**Volatility Adjusted Momentum:** divides raw momentum by 20-day volatility raised to the selected exponent.\n\n**Persistence Momentum:** blends momentum rank with the proportion of positive sessions over 20 days.\n\n**Regime Momentum:** uses Base Momentum only while SPY is above its selected moving average; otherwise cash.\n\n**Dual Momentum:** requires both SPY and each selected stock to be above their selected moving averages.\n\n**Momentum Ensemble:** averages the percentile ranks of Base, Relative Strength, Percentile, Volatility Adjusted, and Persistence Momentum.''')
 st.download_button('Download v7 summaries',report.to_csv(index=False),'v7_summaries.csv','text/csv');st.download_button('Download v7 holdings',holds.to_csv(index=False),'v7_holdings.csv','text/csv');st.download_button('Download contribution analysis',contrib.to_csv(index=False),'v7_contributions.csv','text/csv');st.download_button('Download equity curves',comparison.reset_index().to_csv(index=False),'v7_equity.csv','text/csv')
 st.warning('Known limitation: universe.csv represents current membership rather than point-in-time historical membership, so historical tests may contain survivorship bias.')


if scan:
 if not selected:st.error('Select at least one strategy.');st.stop()
 if weight_total<=0:st.error('Momentum weights must total more than zero.');st.stop()
 settings={'w20':w20,'w60':w60,'w126':w126,'rs_momentum_weight':rs_mom,'rs_weight':rs_weight,'vol_penalty':vol_penalty,'persistence_momentum_weight':pers_mom,'persistence_weight':pers_weight,'regime_period':regime_period,'stock_ma_period':stock_ma_period}
 syms=load_symbols();req=sorted(set(syms+['SPY','QQQ']));scan_end=date.today();scan_warm=scan_end-timedelta(days=max(900,(max(regime_period,stock_ma_period)+300)*2));bar=st.progress(0,text='Loading latest market data')
 try:scan_bars=AlpacaStore().fetch(req,scan_warm,scan_end,int(cfg.get('batch_size',20)),lambda v,m:bar.progress(v or 0,text=m))
 except Exception as e:st.error(str(e));st.stop()
 scan_valid={s:x for s,x in scan_bars.items() if len(x)>=max(127,regime_period,stock_ma_period)}
 if 'SPY' not in scan_valid:st.error('SPY history is unavailable for the current scan.');st.stop()
 try:scan_day,leaders,consensus,master=scan_latest_rankings(scan_valid,settings,selected,top_n,liquidity)
 except Exception as e:st.error(str(e));st.stop()
 bar.empty();st.subheader("Latest universe rankings");st.caption(f"Latest completed market-data date: {scan_day.date()}. Ranked {len(master):,} eligible symbols from universe.csv.")
 if not bool(master.spy_regime_pass.iloc[0]):st.warning(f"SPY is below its {regime_period}-day moving average. Regime Momentum and Dual Momentum therefore return no leaders.")
 tabs_scan=st.tabs(['Consensus']+list(leaders)+['All ranks'])
 with tabs_scan[0]:
  st.dataframe(consensus,use_container_width=True,hide_index=True)
  st.download_button('Download consensus leaders',consensus.to_csv(index=False),'latest_consensus_leaders.csv','text/csv')
 for i,(name,table) in enumerate(leaders.items(),start=1):
  with tabs_scan[i]:
   show=['rank','symbol','score','close','ret_20','ret_60','ret_126','rs_spy20','rs_spy60','vol_20','up_day_ratio20','stock_above_ma','spy_regime_pass']
   st.dataframe(table[[c for c in show if c in table.columns]],use_container_width=True,hide_index=True)
 with tabs_scan[-1]:
  st.dataframe(master,use_container_width=True,hide_index=True)
  st.download_button('Download all latest ranks',master.to_csv(index=False),'latest_all_strategy_ranks.csv','text/csv')
