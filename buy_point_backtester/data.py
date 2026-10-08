from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import os, json, time, hashlib, warnings
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pandas as pd
from dotenv import load_dotenv

PATCH_VERSION = 'feed-isolated-v1'

def load_symbols(path='data/universe.csv'):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'Missing {p}. Add one column named Symbol.')
    df=pd.read_csv(p);cols={str(c).strip().lower():c for c in df.columns}
    if 'symbol' not in cols: raise ValueError('Universe CSV must contain a column named Symbol.')
    values=df[cols['symbol']].dropna().astype(str).str.strip().str.upper()
    return [s for s in values.unique().tolist() if s]

def active_feed():
    load_dotenv()
    value=os.getenv('ALPACA_DATA_FEED','sip').strip().lower()
    if value not in ('sip','iex'): raise ValueError('ALPACA_DATA_FEED must be sip or iex.')
    return value

def feature_cache_dir():
    feed=active_feed();manifest=Path('data/cache')/feed/'coverage.json'
    revision=hashlib.sha256(manifest.read_bytes() if manifest.exists() else b'empty').hexdigest()[:16]
    return Path('results/learning_cache')/feed/(PATCH_VERSION+'_'+revision)

class AlpacaStore:
    def __init__(self,cache_dir='data/cache'):
        load_dotenv()
        self.key=os.getenv('ALPACA_API_KEY');self.secret=os.getenv('ALPACA_SECRET_KEY')
        if not self.key or not self.secret: raise RuntimeError('Add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env')
        self.feed_name=active_feed();self.cache_dir=Path(cache_dir)/self.feed_name
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.manifest_path=self.cache_dir/'coverage.json'
        self.coverage=json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self.diagnostics=[]
    def _path(self,s): return self.cache_dir/(s.replace('/','_')+'.parquet')
    def _read(self,s):
        p=self._path(s)
        if not p.exists(): return pd.DataFrame()
        try:
            x=pd.read_parquet(p);x.index=pd.to_datetime(x.index)
            if x.index.tz is not None:x.index=x.index.tz_localize(None)
            return x.sort_index()
        except Exception as exc: raise RuntimeError(f'Cannot read {self.feed_name} cache for {s}: {type(exc).__name__}. Restore or rename only that cache file.') from exc
    def _save_coverage(self):
        tmp=self.manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.coverage,sort_keys=True));tmp.replace(self.manifest_path)
    def _request_page(self,params):
        from urllib.parse import urlencode
        req=Request('https://data.alpaca.markets/v2/stocks/bars?'+urlencode(params),headers={'APCA-API-KEY-ID':self.key,'APCA-API-SECRET-KEY':self.secret})
        for attempt in range(4):
            try:
                with urlopen(req,timeout=90) as response:return json.load(response)
            except HTTPError as exc:
                status=exc.code
                if status in (401,403):raise RuntimeError(f'Alpaca HTTP {status}: authentication or {self.feed_name.upper()} entitlement denied. Check account permissions; no automatic feed substitution was made.') from None
                if status==400:raise RuntimeError('Alpaca HTTP 400: invalid request. Check symbols and date range.') from None
                if attempt==3:raise RuntimeError(f'Alpaca HTTP {status}: request failed after retries.') from None
                time.sleep(2**attempt)
            except (OSError,ValueError):
                if attempt==3:raise RuntimeError('Alpaca network/response failure after retries.') from None
                time.sleep(2**attempt)
    def _download(self,batch,start,end):
        # Inclusive requested date range; end timestamp is next midnight UTC.
        params={'symbols':','.join(batch),'timeframe':'1Day','start':str(start)+'T00:00:00Z','end':str(end+timedelta(days=1))+'T00:00:00Z','feed':self.feed_name,'adjustment':'all','limit':10000,'sort':'asc'}
        collected={s:[] for s in batch};seen=set()
        while True:
            page=self._request_page(params)
            for symbol,rows in page.get('bars',{}).items():
                if symbol in collected:collected[symbol].extend(rows)
            token=page.get('next_page_token')
            if not token:break
            if token in seen:raise RuntimeError('Alpaca repeated a pagination token; download aborted rather than caching incomplete data.')
            seen.add(token);params['page_token']=token
        # Only commit after all pages have succeeded.
        for s,rows in collected.items():
            if rows:
                x=pd.DataFrame(rows).rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
                idx=pd.to_datetime(x.pop('t'),utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
                x.index=pd.DatetimeIndex(idx);x=x[['open','high','low','close','volume']].sort_index()
                x=x.loc[(x.index.date>=start)&(x.index.date<=end)]
                old=self._read(s)
                if not old.empty:x=pd.concat([old,x]).sort_index()
                x=x[~x.index.duplicated(keep='last')]
                tmp=self._path(s).with_suffix('.tmp.parquet');x.to_parquet(tmp);tmp.replace(self._path(s))
            intervals=self.coverage.setdefault(s,{}).setdefault('intervals',[])
            intervals.append([str(start),str(end)])
            self.coverage[s]['updated_utc']=datetime.now(timezone.utc).isoformat()
        self._save_coverage()
    def _missing(self,s,start,end):
        # Coverage records successful requests, including empty pre-listing dates.
        intervals=self.coverage.get(s,{}).get('intervals',[]) if self._path(s).exists() else []
        cursor=start;missing=[]
        for a,b in sorted((date.fromisoformat(a),date.fromisoformat(b)) for a,b in intervals):
            if b<cursor:continue
            if a>end:break
            if a>cursor:missing.append((cursor,min(end,a-timedelta(days=1))))
            cursor=max(cursor,b+timedelta(days=1))
            if cursor>end:break
        if cursor<=end:missing.append((cursor,end))
        return missing
    def fetch(self,symbols,start:date,end:date,batch_size=20,progress=None):
        from zoneinfo import ZoneInfo
        start=pd.Timestamp(start).date();end=pd.Timestamp(end).date()
        if start>end:raise ValueError('Download start must not exceed end.')
        original_start=start
        floor=date(2016,1,1) if self.feed_name=='sip' else date(2020,1,1)
        # Completed sessions only: excludes the current New York date.
        ny_today=datetime.now(ZoneInfo('America/New_York')).date()
        end=min(end,ny_today-timedelta(days=1));start=max(start,floor)
        if end<start:raise RuntimeError(f'{self.feed_name.upper()} request has no supported completed dates. Use SIP for pre-2020 tests; this downloader supports SIP from 2016.')
        wanted=sorted(set(symbols));jobs={};errors={}
        for s in wanted:
            for a,b in self._missing(s,start,end):jobs.setdefault((a,b),[]).append(s)
        batches=[(a,b,items[i:i+max(1,int(batch_size))]) for (a,b),items in jobs.items() for i in range(0,len(items),max(1,int(batch_size)))]
        for i,(a,b,batch) in enumerate(batches):
            if progress:progress(i/max(1,len(batches)),f'{self.feed_name.upper()} download {a} to {b}: {batch[0]}')
            try:self._download(batch,a,b)
            except RuntimeError as exc:
                # Retry symbols separately to isolate invalid symbols, except access failures.
                if 'HTTP 401' in str(exc) or 'HTTP 403' in str(exc):raise
                for s in batch:
                    try:self._download([s],a,b)
                    except RuntimeError as individual:errors[s]=str(individual)
        out={};self.diagnostics=[]
        for s in wanted:
            x=self._read(s)
            if not x.empty:x=x.loc[(x.index.date>=start)&(x.index.date<=end)].copy()
            row={'symbol':s,'feed':self.feed_name,'rows':len(x),'first_date':str(x.index.min().date()) if len(x) else '', 'last_date':str(x.index.max().date()) if len(x) else '', 'error':errors.get(s,'')}
            self.diagnostics.append(row)
            if len(x):out[s]=x
        pd.DataFrame(self.diagnostics).to_csv(self.cache_dir/'last_download_diagnostics.csv',index=False)
        for s in ('SPY','QQQ'):
            if s in wanted:
                x=out.get(s,pd.DataFrame())
                if not len(x) or s in errors:
                    raise RuntimeError(f'{s} {self.feed_name.upper()} history failed: {errors.get(s,"no bars returned")}. See {self.cache_dir}/last_download_diagnostics.csv.')
                if (end-start).days>=600 and len(x)<300:
                    raise RuntimeError(f'{s} {self.feed_name.upper()} returned only {len(x)} rows ({x.index.min().date()} to {x.index.max().date()}); at least 300 are required by the app. See diagnostics CSV.')
        if errors:warnings.warn(f'{len(errors)} stock downloads failed. Details: {self.cache_dir}/last_download_diagnostics.csv',RuntimeWarning)
        if progress:progress(1.,f'{self.feed_name.upper()} cache ready: {len(out)}/{len(wanted)} symbols. Effective range {start} to {end}.')
        return out
