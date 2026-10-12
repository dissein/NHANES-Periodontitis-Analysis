from pathlib import Path
import json,hashlib,urllib.request,time,io
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
def main():
    sources=json.loads((OUT/'results/raw_sources.json').read_text(encoding='utf-8'))
    docs=OUT/'docs/official_codebooks';docs.mkdir(exist_ok=True)
    def check(rec):
        name=rec['module'];req=urllib.request.Request(rec['codebook'],headers={'User-Agent':'NHANES academic analysis'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=60) as r:html=r.read()
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
        (docs/f'{name}.html').write_bytes(html)
        out={'module':name,'codebook_saved':True}
        if name.endswith('_H'):
            req=urllib.request.Request(rec['url'],headers={'User-Agent':'NHANES academic analysis'})
            with urllib.request.urlopen(req,timeout=120) as r:data=r.read()
            assert b'HEADER RECORD' in data[:160]
            h=hashlib.sha256(data).hexdigest();out['official_current_sha256']=h;out['matches_local_bytes']=h==rec['sha256']
            if not out['matches_local_bytes']:
                a=pd.read_sas(io.BytesIO(data),format='xport');b=pd.read_sas(OUT/'data/raw'/f'{name}.xpt',format='xport')
                try:pd.testing.assert_frame_equal(a,b);out['same_released_values']=True
                except AssertionError:out['same_released_values']=False
            else:out['same_released_values']=True
        else:out['same_released_values']=True
        print(name,'verified',flush=True);return out
    with ThreadPoolExecutor(max_workers=4) as ex:checks=list(ex.map(check,sources))
    (OUT/'results/official_source_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(x['same_released_values'] for x in checks),'Official H data values differ: reanalysis required'
    print('Official source refresh verified',len(checks),flush=True)
if __name__=='__main__':main()
