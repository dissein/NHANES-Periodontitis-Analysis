"""Reproducible exploratory nested, PSU-grouped model comparison.
No evaluation fold is used for tuning, imputation, threshold choice or stopping.
"""
from pathlib import Path
import os, json, time, warnings, shutil, importlib.metadata
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import t
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve, confusion_matrix, f1_score
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT
NON=['SEQN','Periodontitis_Status','WTMEC2YR','SDMVPSU','SDMVSTRA']
CAT={'RIAGENDR':[1,2],'RIDRETH3':[1,2,3,4,6,7],'DMDEDUC2':[1,2,3,4,5],'OHQ030':[1,2,3,4,5,6,7],'SmokingStatus':[0,1,2]}

SEED=20261010
TASKS={'AnyVsNone':['Mild','Moderate','Severe'],'ModSevVsRest':['Moderate','Severe'],'SevVsRest':['Severe']}
SPECS={
 'AgeOnly':[{}],
 'NonLabLogistic':[{'C':.1},{'C':1.}],
 'ElasticNet':[{'C':.1,'l1_ratio':.2},{'C':1.,'l1_ratio':.5}],
 'RandomForest':[{'max_depth':5,'min_samples_leaf':15},{'max_depth':9,'min_samples_leaf':10}],
 'XGBoost':[{'max_depth':2,'min_child_weight':10},{'max_depth':3,'min_child_weight':15}],
 'EBM':[{'interactions':0},{'interactions':5}],
 'TabNet':[{'n_d':8,'n_a':8,'n_steps':3,'epochs':40}]
}
BASE=['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','BMXBMI','BMXWAIST','OHQ030']

def prep(cols):
    cat=[c for c in cols if c in CAT]; num=[c for c in cols if c not in CAT]
    return ColumnTransformer([
      ('num',Pipeline([('fill',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
      ('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value=-1)),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],sparse_threshold=0)

def fit(name,params,X,y,w,seed):
    # Unweighted transformation fit on training fold only. Survey weights in loss.
    pp=prep(X.columns); a=pp.fit_transform(X); wn=w/w.mean()
    if name in ['AgeOnly','NonLabLogistic']:
        model=LogisticRegression(C=params.get('C',100.),solver='lbfgs',max_iter=3000,random_state=seed)
    elif name=='ElasticNet':
        model=LogisticRegression(solver='saga',l1_ratio=params['l1_ratio'],C=params['C'],max_iter=4000,tol=1e-4,random_state=seed)
    elif name=='RandomForest':
        model=RandomForestClassifier(n_estimators=200,max_features=.7,n_jobs=2,random_state=seed,**params)
    elif name=='XGBoost':
        model=XGBClassifier(n_estimators=250,learning_rate=.04,subsample=.8,colsample_bytree=.8,reg_lambda=10,reg_alpha=.2,tree_method='hist',n_jobs=2,random_state=seed,**params)
    elif name=='EBM':
        model=ExplainableBoostingClassifier(outer_bags=4,learning_rate=.03,max_rounds=1000,max_bins=64,max_interaction_bins=16,validation_size=.15,early_stopping_rounds=30,n_jobs=2,random_state=seed,**params)
    else:
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        torch.set_num_threads(2)
        model=TabNetClassifier(n_d=params['n_d'],n_a=params['n_a'],n_steps=params['n_steps'],optimizer_params={'lr':.02},seed=seed,verbose=0,device_name='cpu')
        # Survey-weighted sampling approximates the weighted empirical loss.
        model.fit(a.astype('float32'),y,weights=wn,max_epochs=params['epochs'],patience=0,batch_size=256,virtual_batch_size=128,num_workers=0,drop_last=False)
        return pp,model
    model.fit(a,y,sample_weight=wn)
    if hasattr(model,'n_iter_') and name in ['ElasticNet','NonLabLogistic','AgeOnly']:
        if np.max(model.n_iter_)>=model.max_iter: raise RuntimeError(f'{name}: convergence not reached')
    return pp,model

def predict(fitted,X):
    pp,model=fitted
    return model.predict_proba(pp.transform(X).astype('float32'))[:,1]

def threshold(y,p,w):
    fpr,tpr,cut=roc_curve(y,p,sample_weight=w)
    valid=np.isfinite(cut)&(cut>=0)&(cut<=1)
    return float(cut[valid][np.argmax((tpr-fpr)[valid])])

def metrics(y,p,w,cut):
    pred=p>=cut
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=w).ravel()
    return {'auc':roc_auc_score(y,p,sample_weight=w),'average_precision':average_precision_score(y,p,sample_weight=w),'brier':brier_score_loss(y,p,sample_weight=w),'log_loss':log_loss(y,np.clip(p,1e-6,1-1e-6),sample_weight=w),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp) if tp+fp else np.nan,'npv':tn/(tn+fn) if tn+fn else np.nan,'f1':f1_score(y,pred,sample_weight=w,zero_division=0),'threshold_mean':float(np.mean(cut))}

def calibration(y,p,w):
    x=logit(np.clip(p,1e-6,1-1e-6)); w=w/w.sum()
    def objective(b):
        z=b[0]+b[1]*x
        return np.dot(w,np.logaddexp(0,z)-y*z)
    sol=minimize(objective,[0.,1.],method='BFGS')
    if not sol.success and np.linalg.norm(sol.jac)>1e-4: raise RuntimeError('Calibration fit failed')
    return {'calibration_intercept':float(sol.x[0]),'calibration_slope':float(sol.x[1])}

def cluster_bootstrap(oof,df,B=400):
    rng=np.random.default_rng(SEED); strata=df.SDMVSTRA.to_numpy(); psu=df.SDMVPSU.to_numpy(); w=df.WTMEC2YR.to_numpy()
    values={key:[] for key in oof}
    for b in range(B):
        mult=np.zeros(len(df))
        # Rao-Wu rescaled bootstrap: select m-1 PSUs with replacement per stratum.
        for s in np.unique(strata):
            units=np.unique(psu[strata==s]); m=len(units)
            assert m>=2,'single PSU stratum'
            choices=rng.choice(units,size=m-1,replace=True)
            for u in choices: mult[(strata==s)&(psu==u)]+=m/(m-1)
        for key,v in oof.items():
            wb=w*mult
            values[key].append(roc_auc_score(v['y'],v['prob'],sample_weight=wb))
    rows=[]
    for (task,name),vals in values.items():
        lo,hi=np.quantile(vals,[.025,.975]); rows.append({'task':task,'model':name,'auc_ci_low_conditional':lo,'auc_ci_high_conditional':hi})
    pairs=[]
    for task in TASKS:
        for name in SPECS:
            if name=='NonLabLogistic': continue
            a=np.array(values[(task,name)])-np.array(values[(task,'NonLabLogistic')]); lo,hi=np.quantile(a,[.025,.975])
            pairs.append({'task':task,'model':name,'reference':'NonLabLogistic','delta_auc':roc_auc_score(oof[(task,name)]['y'],oof[(task,name)]['prob'],sample_weight=w)-roc_auc_score(oof[(task,'NonLabLogistic')]['y'],oof[(task,'NonLabLogistic')]['prob'],sample_weight=w),'delta_ci_low_conditional':lo,'delta_ci_high_conditional':hi})
        delta=np.array(values[(task,'EBM')])-np.array(values[(task,'XGBoost')]); lo,hi=np.quantile(delta,[.025,.975])
        pairs.append({'task':task,'model':'EBM','reference':'XGBoost','delta_auc':roc_auc_score(oof[(task,'EBM')]['y'],oof[(task,'EBM')]['prob'],sample_weight=w)-roc_auc_score(oof[(task,'XGBoost')]['y'],oof[(task,'XGBoost')]['prob'],sample_weight=w),'delta_ci_low_conditional':lo,'delta_ci_high_conditional':hi})
    pd.DataFrame(pairs).to_csv(OUT/'results'/'paired_auc_differences.csv',index=False)
    return pd.DataFrame(rows)
