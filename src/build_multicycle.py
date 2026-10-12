from pathlib import Path
import json, hashlib, shutil, urllib.request, time, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'src'))
from label_utils import clean,label,TEETH
ARCHIVE=OUT.parent
MODULES=['DEMO','BMX','BPX','CBC','GHB','BIOPRO','TCHOL','ALB_CR','COT','FOLFMS','VID','OHQ','OHXPER','OHXDEN','SMQ']
CAT={'RIAGENDR':[1,2],'RIDRETH3':[1,2,3,4,6,7],'DMDEDUC2':[1,2,3,4,5],'OHQ030':[1,2,3,4,5,6,7]}

def acquire(item):
    cycle,module=item; year={'G':2011,'H':2013}[cycle];name=f'{module}_{cycle}'
    if name=='COT_G': name='COTNAL_G'
    p=OUT/'data/raw'/f'{name}.xpt'
    url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt'
    src=ARCHIVE/'01_数据/原始XPT'/p.name
    if not src.exists() and module=='OHXDEN' and cycle=='H': src=ARCHIVE/'05_全面复核与修订_20261010/data/raw_supplement/OHXDEN_H.xpt'
    if not p.exists():
        if src.exists(): shutil.copy2(src,p)
        else:
            for attempt in range(3):
                try:
                    req=urllib.request.Request(url,headers={'User-Agent':'NHANES academic reproducibility project'})
                    with urllib.request.urlopen(req,timeout=120) as r: data=r.read()
                    if b'HEADER RECORD' not in data[:160]: raise ValueError(f'Not XPT: {url}')
                    p.write_bytes(data);break
                except Exception:
                    if attempt==2: raise
                    time.sleep(2)
    data=p.read_bytes()
    assert b'HEADER RECORD' in data[:160]
    print(name,len(data),flush=True)
    return {'module':name,'url':url,'local_source':str(src) if src.exists() else None,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'codebook':url.replace('.xpt','.htm')}

def smoking(a,b):
    return pd.Series(np.select([a.eq(2),a.eq(1)&b.eq(3),a.eq(1)&b.isin([1,2])],[0.,1.,2.],default=np.nan),index=a.index)

def main():
    with ThreadPoolExecutor(max_workers=4) as ex: sources=list(ex.map(acquire,[(c,m) for c in 'HG' for m in MODULES]))
    (OUT/'results/raw_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    dictionary_path=ARCHIVE/'05_全面复核与修订_20261010/data/data_dictionary.csv'
    if not dictionary_path.exists():dictionary_path=OUT/'data/data_dictionary_legacy_features.csv'
    olddict=pd.read_csv(dictionary_path)
    flow=[];coverage=[];harmon=[];frames=[]
    for cycle in 'HG':
        tables={m:clean(pd.read_sas(OUT/'data/raw'/('COTNAL_G.xpt' if m=='COT' and cycle=='G' else f'{m}_{cycle}.xpt'),format='xport')).set_index('SEQN',drop=False) for m in MODULES}
        for m,v in tables.items(): assert v.index.is_unique,m
        per=tables['OHXPER']; demo=tables['DEMO']; per['label']=label(per)
        natural=tables['OHXDEN'][[f'OHX{t:02d}TC' for t in TEETH]].isin([1,2]).sum(axis=1)
        al=np.stack([per[[f'OHX{t:02d}LA{s}' for s in 'DSPA']].to_numpy(float) for t in TEETH],axis=1)
        pd_=np.stack([per[[f'OHX{t:02d}PC{s}' for s in 'DSPA']].to_numpy(float) for t in TEETH],axis=1)
        valid=(np.isfinite(al)&(al<90))&(np.isfinite(pd_)&(pd_<90))
        audit=per[['SEQN','OHDPDSTS','OHDEXCLU','label']].copy()
        audit['natural_teeth']=natural.reindex(per.index);audit['paired_sites']=valid.sum(axis=(1,2))
        audit['teeth_with_paired_sites']=valid.any(axis=2).sum(axis=1)
        audit['included']=per.OHDPDSTS.eq(1)&per.label.notna()&demo.WTMEC2YR.reindex(per.index).gt(0)
        audit.to_csv(OUT/'data'/f'eligibility_audit_{cycle}.csv',index=False)
        cohort=per.loc[audit.included].copy(); new=pd.DataFrame(index=cohort.index)
        for r in olddict.to_dict('records'):
            c=r['variable'];m=r['module'].rsplit('_',1)[0]
            if c=='Periodontitis_Status': new[c]=cohort.label
            elif c=='SEQN': new[c]=new.index
            else:
                source='URDACT' if c=='UACR' else c
                if source not in tables[m]: raise KeyError((cycle,m,source))
                new[c]=tables[m][source].reindex(new.index)
                if c in CAT:new[c]=new[c].where(new[c].isin(CAT[c]))
            harmon.append({'cycle':cycle,'variable':c,'module':'COTNAL_G' if m=='COT' and cycle=='G' else m+'_'+cycle,'source':'URDACT' if c=='UACR' else r['source_variable'],'missing_n':int(new[c].isna().sum()),'missing_fraction':float(new[c].isna().mean())})
        sm=tables['SMQ'].reindex(new.index)
        new['SmokingStatus']=smoking(sm.SMQ020,sm.SMQ040)
        new['NaturalTeeth']=natural.reindex(new.index)
        new['PairedSites']=audit.paired_sites.reindex(new.index)
        new['PairedTeeth']=audit.teeth_with_paired_sites.reindex(new.index)
        new['Cycle']=cycle
        new.to_csv(OUT/'data'/f'analysis_{cycle}.csv',index=False);frames.append(new)
        for stage,n in [('MEC age >=30',int(((demo.RIDAGEYR>=30)&(demo.WTMEC2YR>0)).sum())),('periodontal source records',len(per)),('complete exam status',int(per.OHDPDSTS.eq(1).sum())),('complete but no label',int((per.OHDPDSTS.eq(1)&per.label.isna()).sum())),('included',len(new)),('partial exam with some label',int((per.OHDPDSTS.eq(2)&per.label.notna()).sum()))]:flow.append({'cycle':cycle,'stage':stage,'n':n})
        elig=demo.loc[(demo.RIDAGEYR>=30)&(demo.WTMEC2YR>0)].copy()
        elig['Included']=elig.SEQN.isin(new.index);elig['NaturalTeeth']=natural.reindex(elig.index)
        for included,sub in elig.groupby('Included'):
            for c in ['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','NaturalTeeth']:
                v=sub[c].notna();w=sub.loc[v,'WTMEC2YR'];z=sub.loc[v,c]
                if c in CAT:
                    for k in CAT[c]:coverage.append({'cycle':cycle,'included':bool(included),'variable':c,'level':k,'n':len(sub),'value':float(np.average(z.eq(k),weights=w))})
                else:coverage.append({'cycle':cycle,'included':bool(included),'variable':c,'level':'mean','n':len(sub),'value':float(np.average(z,weights=w))})
        # Complete vs partial counts and partial labels are diagnostic only.
        print(cycle,len(new),new.Periodontitis_Status.value_counts().to_dict(),flush=True)
    assert not set(frames[0].SEQN)&set(frames[1].SEQN)
    old_path=ARCHIVE/'05_全面复核与修订_20261010/data/analysis_rebuilt.csv'
    if old_path.exists():
        old=pd.read_csv(old_path).sort_values('SEQN').reset_index(drop=True)
        curr=frames[0].reset_index(drop=True).sort_values('SEQN').reset_index(drop=True)
        pd.testing.assert_frame_equal(old,curr[old.columns],check_dtype=False)
    pd.DataFrame(flow).to_csv(OUT/'results/sample_flow.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'results/inclusion_selection_descriptive.csv',index=False)
    pd.DataFrame(harmon).to_csv(OUT/'results/cycle_harmonization.csv',index=False)
    olddict.to_csv(OUT/'data/data_dictionary_legacy_features.csv',index=False)
    extra=[{'variable':'SmokingStatus','source':'SMQ020, SMQ040','rule':'0=SMQ020==2; 1=SMQ020==1 & SMQ040==3; 2=SMQ020==1 & SMQ040 in [1,2]; unknown stays missing','role':'predictor'}, {'variable':'NaturalTeeth','source':'OHXDEN OHX02TC...OHX31TC excluding third molars','rule':'status 1 or 2 count','role':'sensitivity stratifier, NEVER predictor'},{'variable':'PairedSites','source':'OHXPER','rule':'four interproximal sites per tooth with both CAL and PD finite <90','role':'quality sensitivity, NEVER predictor'},{'variable':'PairedTeeth','source':'OHXPER','rule':'number teeth with at least one paired interproximal site','role':'quality sensitivity, NEVER predictor'}]
    extra.append({'variable':'Cycle','source':'NHANES release suffix G/H','rule':'metadata G=2011-2012; H=2013-2014','role':'cohort metadata, NEVER predictor'})
    pd.DataFrame(extra).to_csv(OUT/'data/data_dictionary_added.csv',index=False)
    full_dictionary=pd.concat([olddict,pd.DataFrame(extra)],ignore_index=True)
    full_dictionary.to_csv(OUT/'data/data_dictionary.csv',index=False)
    (OUT/'results/history_check.json').write_text(json.dumps({'historical_H_comparison_performed':old_path.exists(),'cycles_overlap':0}),encoding='utf-8')
    print('History comparison',old_path.exists(),'; cycles have zero overlapping IDs',flush=True)

if __name__=='__main__': main()
