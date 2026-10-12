from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'src'))
import study_multicycle as s

def main():
    threadpool_limits(limits=2)
    H=pd.read_csv(OUT/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True);G=pd.read_csv(OUT/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
    y=H.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();ye=G.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();w=H.WTMEC2YR.to_numpy();we=G.WTMEC2YR.to_numpy();cols=['RIDAGEYR']
    folds=pd.read_csv(OUT/'data/outer_folds.csv').fold.to_numpy();p=np.zeros(len(H));folder=OUT/'results/age_baseline';folder.mkdir(exist_ok=True)
    for k in range(1,6):
        tr=np.flatnonzero(folds!=k);te=np.flatnonzero(folds==k);params,ip,score=s.tune(H.iloc[tr],H[cols].iloc[tr],'Logistic',y[tr],w[tr],s.SEED+k);fit=s.fit('Logistic',params,H[cols].iloc[tr],y[tr],w[tr],s.SEED+k)
        p[te]=s.predict(fit,H[cols].iloc[te],'Logistic');joblib.dump({'fitted':fit,'params':params},folder/f'fold{k}.joblib')
    params,ip,score=s.tune(H,H[cols],'Logistic',y,w,s.SEED+100);fit=s.fit('Logistic',params,H[cols],y,w,s.SEED+100);pe=s.predict(fit,G[cols],'Logistic');joblib.dump({'fitted':fit,'params':params},folder/'final_model.joblib')
    H[['SEQN']].assign(y=y,prob=p).to_csv(folder/'internal.csv',index=False);G[['SEQN']].assign(y=ye,prob=pe).to_csv(folder/'external.csv',index=False)
    rows=[]
    for stage,d,yy,pp,ww,filename in [('Internal',H,y,p,w,'internal_predictions.csv'),('External',G,ye,pe,we,'external_predictions.csv')]:
        qp=pd.read_csv(OUT/'results/models/Questionnaire_Logistic'/filename).prob.to_numpy();rng=np.random.default_rng(s.SEED+679);vals=[]
        for b in range(1000):
            wb=ww*s.multiplier(d,rng);vals.append(roc_auc_score(yy,qp,sample_weight=wb)-roc_auc_score(yy,pp,sample_weight=wb))
        lo,hi=np.quantile(vals,[.025,.975]);rows.append({'stage':stage,**s.metric(yy,pp,ww),'questionnaire_minus_age_auc':roc_auc_score(yy,qp,sample_weight=ww)-roc_auc_score(yy,pp,sample_weight=ww),'delta_low':lo,'delta_high':hi})
    pd.DataFrame(rows).to_csv(OUT/'results/age_only_baseline.csv',index=False)
    checks=[]
    for k in range(1,6):
        ix=np.flatnonzero(folds==k);pack=joblib.load(folder/f'fold{k}.joblib');pp=s.predict(pack['fitted'],H[cols].iloc[ix],'Logistic');error=float(np.max(abs(pp-p[ix])));assert error<1e-6;checks.append({'stage':f'fold{k}','max_prediction_error':error})
    pack=joblib.load(folder/'final_model.joblib');pp=s.predict(pack['fitted'],G[cols],'Logistic');error=float(np.max(abs(pp-pe)));assert error<1e-6;checks.append({'stage':'external','max_prediction_error':error})
    pd.DataFrame(checks).to_csv(OUT/'results/age_baseline_replay_checks.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
