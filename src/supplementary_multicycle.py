from pathlib import Path
import sys,json,time,urllib.request,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.stats import t
from sklearn.metrics import roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'src'))
import study_multicycle as s

def table1(d,cycle):
    out=[]
    for label,sub in [('All',d)]+list(d.groupby('Periodontitis_Status')):
      for c in s.FULL+['NaturalTeeth','PairedSites']:
        ok=sub[c].notna();z=sub.loc[ok,c];w=sub.loc[ok,'WTMEC2YR']
        if c in s.u.CAT:
          for v in s.u.CAT[c]:out.append({'cycle':cycle,'group':label,'variable':c,'category':v,'n':len(sub),'missing_n':int((~ok).sum()),'weighted_value':float(np.average(z.eq(v),weights=w))})
        else:out.append({'cycle':cycle,'group':label,'variable':c,'category':'mean','n':len(sub),'missing_n':int((~ok).sum()),'weighted_value':float(np.average(z,weights=w)),'q25_unweighted':z.quantile(.25),'median_unweighted':z.median(),'q75_unweighted':z.quantile(.75)})
    return out

def prevalence(d,cycle):
    design=pd.read_sas(OUT/'data/raw'/f'DEMO_{cycle}.xpt',format='xport');design=design.loc[design.WTMEC2YR>0].copy()
    dfree=len(design[['SDMVSTRA','SDMVPSU']].drop_duplicates())-design.SDMVSTRA.nunique();w=d.WTMEC2YR.to_numpy();total=w.sum();out=[]
    for label,pos in [('None',['No Periodontitis']),('Mild',['Mild']),('Moderate',['Moderate']),('Severe',['Severe']),('ModSev',['Moderate','Severe']),('Any',['Mild','Moderate','Severe'])]:
        y=d.Periodontitis_Status.isin(pos).astype(int).to_numpy();p=np.average(y,weights=w)
        inf=pd.Series(w*(y-p)/total,index=d.SEQN);design['influence']=design.SEQN.map(inf).fillna(0)
        g=design.groupby(['SDMVSTRA','SDMVPSU']).influence.sum();var=sum(len(z)/(len(z)-1)*np.sum((z-z.mean())**2) for _,z in g.groupby(level=0))
        half=t.ppf(.975,dfree)*np.sqrt(var)/(p*(1-p));ci=expit(logit(p)+np.array([-half,half]))
        out.append({'cycle':cycle,'outcome':label,'n':int(y.sum()),'weighted_prevalence':p,'ci_low':ci[0],'ci_high':ci[1],'design_df':dfree})
    return out

def associations(d,cycle):
    # Separate explanatory model, explicitly exploratory; categorical reference fixed.
    cols=s.Q+['BMXBMI'];a=d.dropna(subset=cols).copy()
    x=pd.DataFrame({'intercept':1.,'Age_per10':a.RIDAGEYR/10,'PIR_per1':a.INDFMPIR,'BMI_per5':a.BMXBMI/5},index=a.index)
    refs={'RIAGENDR':1,'RIDRETH3':3,'DMDEDUC2':5,'OHQ030':1,'SmokingStatus':0}
    for c,ref in refs.items():
        for k in s.u.CAT[c]:
            if k!=ref:x[f'{c}_{k}_vs_{ref}']=a[c].eq(k).astype(float)
    A=x.to_numpy(float);w=a.WTMEC2YR.to_numpy();w=w/w.mean();y=a.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy()
    assert np.linalg.matrix_rank(A)==A.shape[1]
    obj=lambda b:np.sum(w*(np.logaddexp(0,A@b)-y*(A@b)))
    grad=lambda b:A.T@(w*(expit(A@b)-y))
    sol=minimize(obj,np.zeros(A.shape[1]),jac=grad,method='BFGS',options={'gtol':1e-6,'maxiter':3000});assert max(abs(grad(sol.x)))<1e-3
    p=expit(A@sol.x);H=A.T@((w*p*(1-p))[:,None]*A);bread=np.linalg.inv(H)
    demo=pd.read_sas(OUT/'data/raw'/f'DEMO_{cycle}.xpt',format='xport');design=demo.loc[demo.WTMEC2YR>0,['SEQN','SDMVSTRA','SDMVPSU']]
    dfree=len(design[['SDMVSTRA','SDMVPSU']].drop_duplicates())-design.SDMVSTRA.nunique()
    scores=pd.DataFrame(A*(w*(y-p))[:,None],index=a.SEQN,columns=x.columns)
    sums=design.set_index('SEQN').join(scores).fillna(0).groupby(['SDMVSTRA','SDMVPSU'])[list(x.columns)].sum();meat=np.zeros_like(H)
    for _,z in sums.groupby(level=0):
        v=z.to_numpy()-z.to_numpy().mean(0);meat+=len(z)/(len(z)-1)*(v.T@v)
    se=np.sqrt(np.maximum(0,np.diag(bread@meat@bread)));crit=t.ppf(.975,dfree)
    return [{'cycle':cycle,'term':name,'n_complete':len(a),'POR':np.exp(sol.x[i]),'ci_low':np.exp(sol.x[i]-crit*se[i]),'ci_high':np.exp(sol.x[i]+crit*se[i]),'design_df':dfree} for i,name in enumerate(x.columns) if i>0]

def main():
    threadpool_limits(limits=2);start=time.time()
    H=pd.read_csv(OUT/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True);G=pd.read_csv(OUT/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
    pd.DataFrame(table1(H,'H')+table1(G,'G')).to_csv(OUT/'results/table1.csv',index=False)
    pd.DataFrame(prevalence(H,'H')+prevalence(G,'G')).to_csv(OUT/'results/prevalence.csv',index=False)
    pd.DataFrame(associations(H,'H')+associations(G,'G')).to_csv(OUT/'results/associations_nonlab.csv',index=False)
    # Same frozen development protocol in the other chronological direction.
    rows=[]
    for panel in ['NonLabBP','Expanded']:
      for name in s.SPECS:
        key=panel+'_'+name;cols=s.PANELS[panel];folder=OUT/'results/reverse_models'/key;folder.mkdir(parents=True,exist_ok=True)
        y=G.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();w=G.WTMEC2YR.to_numpy();yy=H.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();ww=H.WTMEC2YR.to_numpy()
        params,ip,score=s.tune(G,G[cols],name,y,w,s.SEED+202);fitted=s.fit(name,params,G[cols],y,w,s.SEED+202);p=s.predict(fitted,H[cols],name)
        cut=s.target_cut(y,ip,w,.8);joblib.dump({'fitted':fitted,'params':params,'features':cols,'train_ids':G.SEQN.tolist()},folder/'model.joblib')
        H[['SEQN']].assign(y=yy,prob=p).to_csv(folder/'predictions_H.csv',index=False)
        rows.append({'key':key,'development':'G','validation':'H',**s.metric(yy,p,ww),**s.operating(yy,p,ww,cut)})
        print('Reverse completed',key,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'results/reverse_transport.csv',index=False)
    # Fixed panel ablations, exploratory and trained without validation outcomes.
    rows=[]
    for variant,cols in [('NonLab_without_dental_visit',[c for c in s.N if c!='OHQ030']),('NonLab_without_race',[c for c in s.N if c!='RIDRETH3']),('NonLab_without_smoking',[c for c in s.N if c!='SmokingStatus'])]:
        y=H.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();w=H.WTMEC2YR.to_numpy();ye=G.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();we=G.WTMEC2YR.to_numpy()
        params,ip,score=s.tune(H,H[cols],'Logistic',y,w,s.SEED+100);f=s.fit('Logistic',params,H[cols],y,w,s.SEED+100);p=s.predict(f,G[cols],'Logistic')
        joblib.dump({'fitted':f,'params':params,'features':cols},OUT/'results/models'/f'{variant}.joblib')
        G[['SEQN']].assign(y=ye,prob=p).to_csv(OUT/'results'/f'ablation_{variant}.csv',index=False)
        rows.append({'variant':variant,'n_features':len(cols),**s.metric(ye,p,we)})
    pd.DataFrame(rows).to_csv(OUT/'results/feature_ablations_external.csv',index=False)
    sub=[];bins=[];dec=[];sens=[];secondary=[];imp=[];rng=np.random.default_rng(s.SEED+444)
    keys=[f'{p}_{m}' for p in ['Questionnaire','NonLab','NonLabBP','Expanded'] for m in ['Logistic','EBM']]
    for cycle,d,filename in [('H',H,'internal_predictions.csv'),('G',G,'external_predictions.csv')]:
      for key in keys:
        rr=pd.read_csv(OUT/'results/models'/key/filename);y=rr.y.to_numpy();p=rr.prob.to_numpy();w=d.WTMEC2YR.to_numpy();cut=rr.cut80.to_numpy()
        strata=[('sex',d.RIAGENDR),('age',pd.cut(d.RIDAGEYR,[29,49,64,100],labels=['30-49','50-64','65+'])),('race',d.RIDRETH3),('income',pd.cut(d.INDFMPIR,[-.01,1,3,6],labels=['<=1','1-3','>3']))]
        for typ,values in strata:
          for level in values.dropna().unique():
            ix=np.flatnonzero(values.eq(level));ny=int(y[ix].sum())
            if ny<15 or len(ix)-ny<15:continue
            out={'cycle':cycle,'key':key,'subgroup':typ,'level':str(level),'n':len(ix),'n_cases':ny,**s.metric(y[ix],p[ix],w[ix]),**s.operating(y[ix],p[ix],w[ix],cut[ix])}
            boot=[]
            for b in range(300):
                wb=w*s.multiplier(d,rng)
                if wb[ix][y[ix]==1].sum()>0 and wb[ix][y[ix]==0].sum()>0:boot.append(roc_auc_score(y[ix],p[ix],sample_weight=wb[ix]))
            out['auc_low'],out['auc_high']=np.quantile(boot,[.025,.975]);sub.append(out)
        for k in range(10):
            ix=(p>=k/10)&((p<(k+1)/10) if k<9 else p<=1)
            if ix.any():bins.append({'cycle':cycle,'key':key,'bin':k,'n':int(ix.sum()),'predicted':np.average(p[ix],weights=w[ix]),'observed':np.average(y[ix],weights=w[ix]),'weight_fraction':w[ix].sum()/w.sum()})
        for threshold in np.arange(.05,.61,.025):
            pred=p>=threshold;tp=np.sum(w*pred*y)/w.sum();fp=np.sum(w*pred*(1-y))/w.sum();prev=np.average(y,weights=w)
            dec.append({'cycle':cycle,'key':key,'threshold':threshold,'net_benefit':tp-fp*threshold/(1-threshold),'examine_all':prev-(1-prev)*threshold/(1-threshold),'examine_none':0.})
        for desc,ok in [('at_least20_natural_teeth',d.NaturalTeeth>=20),('at_least90pct_paired_sites',d.PairedSites>=.9*4*d.NaturalTeeth),('complete_NonLab',d[s.N].notna().all(axis=1)),('complete_Expanded',d[s.FULL].notna().all(axis=1))]:
            ix=np.flatnonzero(ok)
            sens.append({'cycle':cycle,'key':key,'restriction':desc,'n':len(ix),**s.metric(y[ix],p[ix],w[ix])})
        for endpoint,pos in [('Any',['Mild','Moderate','Severe']),('Severe',['Severe'])]:
            yy=d.Periodontitis_Status.isin(pos).astype(int).to_numpy();secondary.append({'cycle':cycle,'key':key,'endpoint':endpoint,'note':'AUC of same frozen moderate/severe score, NOT endpoint-specific probability','auc':roc_auc_score(yy,p,sample_weight=w)})
        if cycle=='G' and key in ['NonLab_Logistic','Expanded_Logistic','Expanded_EBM']:
            pack=joblib.load(OUT/'results/models'/key/'final_model.joblib');name=key.rsplit('_',1)[1];cols=pack['features'];baseline=roc_auc_score(y,p,sample_weight=w)
            for c in cols:
                losses=[]
                for repeat in range(5):
                    xx=d[cols].copy();xx[c]=rng.permutation(xx[c].to_numpy());pr=s.predict(pack['fitted'],xx,name);losses.append(baseline-roc_auc_score(y,pr,sample_weight=w))
                imp.append({'key':key,'variable':c,'auc_decrease':np.mean(losses),'sd_permutation':np.std(losses,ddof=1)})
    for name,rows in [('subgroups',sub),('calibration_bins',bins),('decision_curves',dec),('quality_sensitivity',sens),('secondary_score_discrimination',secondary),('external_permutation',imp)]:pd.DataFrame(rows).to_csv(OUT/'results'/f'{name}.csv',index=False)
    print('SUPPLEMENT COMPLETE',round(time.time()-start),flush=True)

if __name__=='__main__':main()
