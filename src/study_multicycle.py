from pathlib import Path
import sys,json,time,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.preprocessing import SplineTransformer,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,roc_curve,brier_score_loss,average_precision_score,confusion_matrix
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'src'))
import model_utils as u
SEED=20261010
Q=['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','OHQ030','SmokingStatus']
N=Q+['BMXBMI','BMXWAIST']
CBC=['LBXEOPCT','LBXHGB','LBXLYPCT','LBXMCVSI','LBXMOPCT','LBXMPSI','LBXNEPCT','LBXPLTSI','LBXRDW','LBXWBCSI']
FULL=pd.read_csv(OUT/'data/data_dictionary_legacy_features.csv').query('role=="predictor"').variable.tolist()+['SmokingStatus']
PANELS={'Questionnaire':Q,'NonLab':N,'NonLabBP':N+['BPXSY1','BPXDI1'],'CBC':N+CBC,'Expanded':FULL}
SPECS={'Logistic':[{'C':.1},{'C':1.}], 'EBM':[{'interactions':0,'max_leaves':2},{'interactions':0,'max_leaves':3}], 'XGBoost':[{'max_depth':2,'min_child_weight':10},{'max_depth':3,'min_child_weight':15}]}
SKEW=['LBXCOT','UACR','LBXSASSI','LBXSATSI','LBXSCK','LBXSGTSI','LBXSTR','LBXSAPSI','LBXWBCSI']

def transform(X,name):
    X=X.copy()
    if name=='Logistic':
        for c in SKEW:
            if c in X: X[c]=np.log1p(X[c].clip(lower=0))
    return X

def fit(name,params,X,y,w,seed):
    X=transform(X,name)
    if name!='Logistic':return u.fit(name,params,X,y,w,seed)
    cat=[c for c in X if c in u.CAT]; num=[c for c in X if c not in cat and c!='RIDAGEYR']
    pp=ColumnTransformer([
       ('age',Pipeline([('fill',SimpleImputer(strategy='median')),('spline',SplineTransformer(n_knots=4,degree=3,include_bias=False,extrapolation='constant')),('scale',StandardScaler())]),['RIDAGEYR']),
       ('num',Pipeline([('fill',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
       ('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value=-1)),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],sparse_threshold=0)
    a=pp.fit_transform(X);m=LogisticRegression(C=params['C'],max_iter=3000,solver='lbfgs',random_state=seed)
    m.fit(a,y,sample_weight=w/w.mean())
    assert m.n_iter_.max()<m.max_iter
    return pp,m

def predict(fitted,X,name):return u.predict(fitted,transform(X,name))

def group(d):return d.SDMVSTRA.astype(int).astype(str)+'_'+d.SDMVPSU.astype(int).astype(str)

def target_cut(y,p,w,target):
    f,t,c=roc_curve(y,p,sample_weight=w)
    valid=np.isfinite(c)&(c<=1)&(t>=target)
    assert valid.any()
    return float(c[valid].max())

def cal_fit(y,p,w):
    x=logit(np.clip(p,1e-6,1-1e-6)); ww=w/w.sum()
    obj=lambda b:np.dot(ww,np.logaddexp(0,b[0]+b[1]*x)-y*(b[0]+b[1]*x))
    z=minimize(obj,[0.,1.],method='BFGS')
    assert z.success or np.linalg.norm(z.jac)<1e-4
    return z.x

def recal(p,b):return expit(b[0]+b[1]*logit(np.clip(p,1e-6,1-1e-6)))

def tune(d,X,name,y,w,seed):
    g=group(d);folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,g));scored=[]
    for i,params in enumerate(SPECS[name]):
        ip=np.zeros(len(X));fs=[]
        for k,(tr,te) in enumerate(folds):
            assert not set(g.iloc[tr])&set(g.iloc[te])
            fitted=fit(name,params,X.iloc[tr],y[tr],w[tr],seed+k)
            ip[te]=predict(fitted,X.iloc[te],name);fs.append(roc_auc_score(y[te],ip[te],sample_weight=w[te]))
        scored.append((np.mean(fs),i,ip))
    z=max(scored,key=lambda z:z[0]);return SPECS[name][z[1]],z[2],float(z[0])

def metric(y,p,w):
    return {'auc':roc_auc_score(y,p,sample_weight=w),'auc_unweighted':roc_auc_score(y,p),'ap':average_precision_score(y,p,sample_weight=w),'brier':brier_score_loss(y,p,sample_weight=w),'prevalence':float(np.average(y,weights=w)),**u.calibration(y,p,w)}

def operating(y,p,w,cut):
    z=p>=cut;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1],sample_weight=w).ravel();total=w.sum()
    ppv=tp/(tp+fp) if tp+fp else np.nan
    return {'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':ppv,'npv':tn/(tn+fn),'flagged_fraction':(tp+fp)/total,'detected_per1000':tp/total*1000,'false_positive_per1000':fp/total*1000,'missed_per1000':fn/total*1000,'examinations_per_detected_case':1/ppv if ppv else np.nan}

_DESIGNS={}
def multiplier(d,rng):
    out=np.zeros(len(d));s=d.SDMVSTRA.to_numpy();p=d.SDMVPSU.to_numpy()
    cycle=str(d.Cycle.iloc[0])
    if cycle not in _DESIGNS:
        demo=pd.read_sas(OUT/'data/raw'/f'DEMO_{cycle}.xpt',format='xport');demo=demo.loc[demo.WTMEC2YR>0]
        _DESIGNS[cycle]={v:np.unique(z.SDMVPSU) for v,z in demo.groupby('SDMVSTRA')}
    for v,units in _DESIGNS[cycle].items():
        m=len(units);assert m>=2
        picked=rng.choice(units,size=m-1,replace=True)
        for unit in picked:out[(s==v)&(p==unit)]+=m/(m-1)
    return out

def bootstrap(d,pred,B=1000):
    rng=np.random.default_rng(SEED+811);w=d.WTMEC2YR.to_numpy();y=d.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy()
    vals={k:[] for k in pred}
    for b in range(B):
        wb=w*multiplier(d,rng)
        for k,r in pred.items(): vals[k].append([roc_auc_score(y,r['prob'],sample_weight=wb),brier_score_loss(y,r['prob'],sample_weight=wb)])
    intervals=[];pairs=[]
    for k,v in vals.items():
        lo,hi=np.quantile(v,[.025,.975],axis=0)
        intervals.append({'key':k,'auc_low':lo[0],'auc_high':hi[0],'brier_low':lo[1],'brier_high':hi[1]})
    pairspec=[]
    for m in SPECS:
        for panel,ref in [('Expanded','NonLabBP'),('Expanded','NonLab'),('CBC','NonLab'),('NonLabBP','NonLab'),('NonLab','Questionnaire')]:pairspec.append((panel+'_'+m,ref+'_'+m,'information_increment'))
    for p in PANELS:
        for m in ['EBM','XGBoost']:pairspec.append((p+'_'+m,p+'_Logistic','algorithm_increment'))
    for a,b,kind in pairspec:
        delta=np.array(vals[a])-np.array(vals[b]);lo,hi=np.quantile(delta,[.025,.975],axis=0)
        ap=pred[a]['prob'];bp=pred[b]['prob']
        pairs.append({'model':a,'reference':b,'comparison':kind,'delta_auc':roc_auc_score(y,ap,sample_weight=w)-roc_auc_score(y,bp,sample_weight=w),'auc_delta_low':lo[0],'auc_delta_high':hi[0],'delta_brier':brier_score_loss(y,ap,sample_weight=w)-brier_score_loss(y,bp,sample_weight=w),'brier_delta_low':lo[1],'brier_delta_high':hi[1]})
    return pd.DataFrame(intervals),pd.DataFrame(pairs)

def main():
    start=time.time();threadpool_limits(limits=2)
    dev=pd.read_csv(OUT/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True)
    ext=pd.read_csv(OUT/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
    y=dev.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();w=dev.WTMEC2YR.to_numpy();g=group(dev)
    ye=ext.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();we=ext.WTMEC2YR.to_numpy()
    outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(dev,dev.Periodontitis_Status,g))
    assignments=np.zeros(len(dev),int)
    for k,(tr,te) in enumerate(outer):assignments[te]=k+1
    dev[['SEQN','SDMVSTRA','SDMVPSU']].assign(fold=assignments).to_csv(OUT/'data/outer_folds.csv',index=False)
    selected=[];summaries=[];ops=[];predictions={'Internal':{},'External':{}};foldmetrics=[]
    for panel,cols in PANELS.items():
      for name in SPECS:
        key=f'{panel}_{name}';folder=OUT/'results/models'/key;folder.mkdir(parents=True,exist_ok=True)
        cache=folder/'external_predictions.csv'
        if cache.exists() and (folder/'metadata.json').exists():
            ir=pd.read_csv(folder/'internal_predictions.csv');er=pd.read_csv(cache);meta=json.loads((folder/'metadata.json').read_text())
            for fold in range(1,6):
                pack=joblib.load(folder/f'fold{fold}.joblib');rr=ir.loc[ir.fold.eq(fold)]
                selected.append({'key':key,'stage':f'outer{fold}','parameters':json.dumps(pack['params']),'inner_auc':pack['inner_auc']})
                foldmetrics.append({'key':key,'fold':fold,**metric(rr.y.to_numpy(),rr.prob.to_numpy(),rr.WTMEC2YR.to_numpy())})
            print('resume',key,flush=True)
        else:
            p=np.zeros(len(dev));pc=p.copy();c80=p.copy();c90=p.copy();cy=p.copy()
            for fold,(tr,te) in enumerate(outer,1):
                assert not set(g.iloc[tr])&set(g.iloc[te])
                checkpoint=folder/f'fold{fold}.joblib'
                if checkpoint.exists():
                    pack=joblib.load(checkpoint);params=pack['params'];ip=pack['inner_probability'];b=pack['calibration'];fitted=pack['fitted'];s=pack['inner_auc']
                else:
                    params,ip,s=tune(dev.iloc[tr],dev[cols].iloc[tr],name,y[tr],w[tr],SEED+fold)
                    b=cal_fit(y[tr],ip,w[tr]);fitted=fit(name,params,dev[cols].iloc[tr],y[tr],w[tr],SEED+fold)
                    pack={'fitted':fitted,'params':params,'features':cols,'calibration':b,'inner_probability':ip,'inner_auc':s,'train_ids':dev.SEQN.iloc[tr].tolist(),'test_ids':dev.SEQN.iloc[te].tolist()}
                    joblib.dump(pack,checkpoint)
                p[te]=predict(fitted,dev[cols].iloc[te],name);pc[te]=recal(p[te],b)
                c80[te]=target_cut(y[tr],ip,w[tr],.8);c90[te]=target_cut(y[tr],ip,w[tr],.9);cy[te]=u.threshold(y[tr],ip,w[tr])
                selected.append({'key':key,'stage':f'outer{fold}','parameters':json.dumps(params),'inner_auc':s})
                foldmetrics.append({'key':key,'fold':fold,**metric(y[te],p[te],w[te])})
                print(key,fold,'AUC',round(foldmetrics[-1]['auc'],4),'elapsed',round(time.time()-start),flush=True)
            ir=dev[['SEQN','SDMVSTRA','SDMVPSU','WTMEC2YR']].assign(y=y,prob=p,prob_calibrated=pc,cut80=c80,cut90=c90,cut_youden=cy,fold=assignments)
            params,ip,s=tune(dev,dev[cols],name,y,w,SEED+100)
            b=cal_fit(y,ip,w);fitted=fit(name,params,dev[cols],y,w,SEED+100)
            cuts={'cut80':target_cut(y,ip,w,.8),'cut90':target_cut(y,ip,w,.9),'cut_youden':u.threshold(y,ip,w)}
            pack={'fitted':fitted,'params':params,'features':cols,'calibration':b,'thresholds':cuts,'train_ids':dev.SEQN.tolist(),'inner_probability':ip,'inner_auc':s}
            joblib.dump(pack,folder/'final_model.joblib')
            pe=predict(fitted,ext[cols],name);er=ext[['SEQN','SDMVSTRA','SDMVPSU','WTMEC2YR']].assign(y=ye,prob=pe,prob_calibrated=recal(pe,b),**cuts)
            meta={'key':key,'features':cols,'n_features':len(cols),'final_parameters':params,'thresholds':cuts,'final_inner_auc':s,'calibration_coefficients':b.tolist(),'external_cycle':'G','external_no_refit':True}
            ir.to_csv(folder/'internal_predictions.csv',index=False);er.to_csv(cache,index=False)
            (folder/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        selected.append({'key':key,'stage':'final_all_H','parameters':json.dumps(meta['final_parameters']),'inner_auc':meta['final_inner_auc']})
        for stage,d,r in [('Internal',dev,ir),('External',ext,er)]:
            ww=d.WTMEC2YR.to_numpy();yy=r.y.to_numpy();pp=r.prob.to_numpy()
            predictions[stage][key]=r
            for variant,c in [('raw','prob'),('training_calibrated','prob_calibrated')]:summaries.append({'stage':stage,'key':key,'panel':panel,'algorithm':name,'variant':variant,'n':len(d),'n_features':len(cols),**metric(yy,r[c].to_numpy(),ww)})
            for target,c in [('sensitivity80','cut80'),('sensitivity90','cut90'),('youden','cut_youden')]:ops.append({'stage':stage,'key':key,'threshold_rule':target,'mean_cut':r[c].mean(),**operating(yy,pp,ww,r[c].to_numpy())})
        pd.DataFrame(summaries).to_csv(OUT/'results/performance_progress.csv',index=False)
    intervals=[];pairs=[]
    for stage,d in [('Internal',dev),('External',ext)]:
        ci,pa=bootstrap(d,predictions[stage]);ci['stage']=stage;pa['stage']=stage;intervals.append(ci);pairs.append(pa)
    pd.DataFrame(summaries).merge(pd.concat(intervals).assign(variant='raw'),on=['stage','key','variant'],how='left').to_csv(OUT/'results/performance.csv',index=False)
    pd.concat(pairs).to_csv(OUT/'results/paired_comparisons.csv',index=False)
    pd.DataFrame(ops).to_csv(OUT/'results/operating_points.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'results/selected_parameters.csv',index=False)
    pd.DataFrame(foldmetrics).to_csv(OUT/'results/fold_performance.csv',index=False)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','interpret-core','joblib','matplotlib']}
    (OUT/'src/requirements_lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items()),encoding='utf-8')
    (OUT/'results/run_metadata.json').write_text(json.dumps({'completed':True,'seconds':time.time()-start,'seed':SEED,'bootstrap':1000,'versions':versions,'primary_raw_probability_CI_only':True},indent=2),encoding='utf-8')
    print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
