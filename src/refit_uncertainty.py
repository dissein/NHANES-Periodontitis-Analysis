from pathlib import Path
import sys,time
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'src'))
import study_multicycle as s

def main(B=200):
    threadpool_limits(limits=2);start=time.time()
    H=pd.read_csv(OUT/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True);G=pd.read_csv(OUT/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
    y=H.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();ye=G.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy()
    w=H.WTMEC2YR.to_numpy();we=G.WTMEC2YR.to_numpy();rng=np.random.default_rng(s.SEED+987)
    path=OUT/'results/refit_bootstrap_replicates.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else [];done={int(r['replicate']) for r in rows}
    for b in range(B):
        wm=w*s.multiplier(H,rng);wem=we*s.multiplier(G,rng)
        if b in done:continue
        ix=np.flatnonzero(wm>0);dev=H.iloc[ix].reset_index(drop=True);yy=y[ix];ww=wm[ix]
        record={'replicate':b,'development_n_retained':len(ix)}
        for panel in ['NonLabBP','Expanded']:
            cols=s.PANELS[panel];params,ip,score=s.tune(dev,dev[cols],'Logistic',yy,ww,s.SEED+100)
            fitted=s.fit('Logistic',params,dev[cols],yy,ww,s.SEED+100);p=s.predict(fitted,G[cols],'Logistic')
            # Validation outcome is used only for evaluation of frozen fitted predictions.
            m=s.metric(ye,p,wem);cut=s.target_cut(yy,ip,ww,.8);op=s.operating(ye,p,wem,cut)
            for c in ['auc','brier']:record[panel+'_'+c]=m[c]
            for c in ['sensitivity','specificity','ppv']:record[panel+'_'+c]=op[c]
        rows.append(record)
        if (b+1)%10==0:
            pd.DataFrame(rows).to_csv(path,index=False);print('Refit bootstrap',b+1,'/',B,'elapsed',round(time.time()-start),flush=True)
    a=pd.DataFrame(rows).sort_values('replicate');a.to_csv(path,index=False);out=[]
    for panel in ['NonLabBP','Expanded']:
        for metric in ['auc','brier','sensitivity','specificity','ppv']:
            lo,hi=np.quantile(a[panel+'_'+metric],[.025,.975]);out.append({'comparison':panel,'metric':metric,'low':lo,'high':hi,'B':len(a),'scope':'development retuning/preprocessing/refitting + independent validation design resampling'})
    for metric in ['auc','brier','sensitivity','specificity','ppv']:
        v=a['Expanded_'+metric]-a['NonLabBP_'+metric];lo,hi=np.quantile(v,[.025,.975]);out.append({'comparison':'Expanded_minus_NonLabBP','metric':metric,'low':lo,'high':hi,'B':len(a),'scope':'paired two-cycle full development-procedure bootstrap'})
    pd.DataFrame(out).to_csv(OUT/'results/refit_uncertainty.csv',index=False)
    print('Refit uncertainty completed',len(a),round(time.time()-start),flush=True)

if __name__=='__main__':main()
