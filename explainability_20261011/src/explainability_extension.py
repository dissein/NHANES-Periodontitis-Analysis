"""Frozen exploratory extension: nested group ablations, grouped SHAP, EBM shapes.
Run plan, train --algorithm ..., explain, evaluate, deliver, in that order.
"""
from pathlib import Path
import sys, json, time, hashlib, argparse, importlib.metadata, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'explainability_20261011'
sys.path.insert(0,str(ROOT/'src'))
import study_multicycle as s
SEED=s.SEED
LABELS={'RIDAGEYR':'年龄','RIAGENDR':'性别','RIDRETH3':'种族／族裔','DMDEDUC2':'教育程度','INDFMPIR':'收入贫困比','OHQ030':'距上次看牙时间','SmokingStatus':'吸烟状态','LBXCOT':'血清可替宁','BMXBMI':'BMI','BMXWAIST':'腰围','LBXWBCSI':'白细胞计数','LBXGH':'糖化血红蛋白','LBXTC':'总胆固醇','LBXSTR':'甘油三酯'}
for _,r in pd.read_csv(ROOT/'data/data_dictionary.csv').iterrows():
    if r.variable not in LABELS: LABELS[r.variable]=str(r.label)

def writejson(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v: v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def data(cycle): return pd.read_csv(ROOT/f'data/analysis_{cycle}.csv').sort_values('SEQN').reset_index(drop=True)
def target(d): return d.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy()
def variants():
    out=[{'variant':'Q_without_'+c,'panel':'Questionnaire','removed':[c]} for c in s.Q]
    out += [{'variant':'Q_without_SES','panel':'Questionnaire','removed':['DMDEDUC2','INDFMPIR']},
            {'variant':'BP_without_body_size','panel':'NonLabBP','removed':['BMXBMI','BMXWAIST']},
            {'variant':'Expanded_without_smoking','panel':'Expanded','removed':['SmokingStatus']},
            {'variant':'Expanded_without_cotinine','panel':'Expanded','removed':['LBXCOT']},
            {'variant':'Expanded_without_both_smoking','panel':'Expanded','removed':['SmokingStatus','LBXCOT']},
            {'variant':'Expanded_without_CBC','panel':'Expanded','removed':s.CBC},
            {'variant':'Expanded_without_glycemic_lipids','panel':'Expanded','removed':['LBXGH','LBXTC','LBXSTR']}]
    for v in out: v['features']=[c for c in s.PANELS[v['panel']] if c not in v['removed']]
    return out

original_fit=s.fit
def fit(name,params,X,y,w,seed):
    if name!='Logistic' or 'RIDAGEYR' in X: return original_fit(name,params,X,y,w,seed)
    # The parent Logistic fit assumes age always exists. Deleting age requires
    # omitting the entire spline block, not retaining a hidden age surrogate.
    X=s.transform(X,name);cat=[c for c in X if c in s.u.CAT];num=[c for c in X if c not in cat]
    pp=ColumnTransformer([
        ('num',Pipeline([('fill',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
        ('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value=-1)),('onehot',s.OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],sparse_threshold=0)
    a=pp.fit_transform(X);m=LogisticRegression(C=params['C'],max_iter=3000,solver='lbfgs',random_state=seed)
    m.fit(a,y,sample_weight=w/w.mean());assert m.n_iter_.max()<m.max_iter
    return pp,m
s.fit=fit

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'analysis_plan.json').exists(): return
    files=list((ROOT/'results/models').glob('*/*.joblib'))+[ROOT/f'data/analysis_{c}.csv' for c in ['H','G']]+[ROOT/'data/outer_folds.csv',ROOT/'results/performance.csv',ROOT/'results/paired_comparisons.csv']
    writejson(OUT/'preservation_before.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    writejson(OUT/'analysis_plan.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),'exploratory':True,'prior_G_results_already_seen':True,'task':'current moderate/severe versus none/mild, same dentate observable domain','development':'H','validation':'G; no tuning or selection in G','algorithms':list(s.SPECS),'variants':variants(),'nested_outer_folds':'original stored 5 PSU-grouped folds','inner':'3 StratifiedGroupKFold; same 2 parent candidates, same seeds','refit':'all variants independently reselect parameters within H; no evaluation early stopping','bootstrap':'1000 paired full-MEC-design Rao-Wu replicates, conditional on fitted models, no multiplicity correction','explanation':'Q7 and Expanded47 XGBoost grouped encoded TreeSHAP; EBM additive eval_terms; H outer-heldout and G final model','shap_primary':'interventional, raw log-odds, H training-only weighted background 128 draws with replacement','shap_sensitivity':'tree_path_dependent final models in G','grouping':'sum encoded attributions including missing indicators to original variable; not a separately estimated coalition Shapley value','EBM_shapes':'native additive contribution relative to fixed training reference; uncertainty shown by five separately fitted outer models, NOT confidence bands','interpretation':'predictive contributions, not causal effects; delete/refit can reflect redundancy, not prove absence of biological association'})
    print('PLAN frozen',len(variants())*3,'new variants',flush=True)

def train(algorithm):
    assert (OUT/'analysis_plan.json').exists()
    threadpool_limits(limits=2);H=data('H');y=target(H);w=H.WTMEC2YR.to_numpy();groups=s.group(H)
    fd=pd.read_csv(ROOT/'data/outer_folds.csv');assert np.array_equal(fd.SEQN,H.SEQN)
    assignments=fd.fold.to_numpy();start=time.time()
    for v in variants():
        key=v['variant']+'_'+algorithm;folder=OUT/'models'/key;folder.mkdir(parents=True,exist_ok=True);cols=v['features']
        if (folder/'completed.json').exists(): print('resume completed',key,flush=True);continue
        prob=np.zeros(len(H));checks=[]
        for k in range(1,6):
            tr=np.flatnonzero(assignments!=k);te=np.flatnonzero(assignments==k)
            assert not set(groups.iloc[tr])&set(groups.iloc[te])
            f=folder/f'fold{k}.joblib'
            if f.exists():pack=joblib.load(f)
            else:
                params,ip,score=s.tune(H.iloc[tr],H[cols].iloc[tr],algorithm,y[tr],w[tr],SEED+k)
                fitted=fit(algorithm,params,H[cols].iloc[tr],y[tr],w[tr],SEED+k)
                pack={'fitted':fitted,'features':cols,'params':params,'inner_auc':score,'train_ids':H.SEQN.iloc[tr].tolist(),'test_ids':H.SEQN.iloc[te].tolist()};joblib.dump(pack,f)
            assert pack['train_ids']==H.SEQN.iloc[tr].tolist() and pack['test_ids']==H.SEQN.iloc[te].tolist()
            prob[te]=s.predict(pack['fitted'],H[cols].iloc[te],algorithm)
            checks.append({'fold':k,'params':pack['params'],'inner_auc':pack['inner_auc'],'n_train':len(tr),'n_test':len(te)})
        f=folder/'final_model.joblib'
        if f.exists():pack=joblib.load(f)
        else:
            params,ip,score=s.tune(H,H[cols],algorithm,y,w,SEED+100);fitted=fit(algorithm,params,H[cols],y,w,SEED+100)
            pack={'fitted':fitted,'features':cols,'params':params,'inner_auc':score,'train_ids':H.SEQN.tolist()};joblib.dump(pack,f)
        H[['SEQN']].assign(y=y,prob=prob,fold=assignments).to_csv(folder/'internal_predictions.csv',index=False)
        writejson(folder/'completed.json',{'key':key,'variant':v,'algorithm':algorithm,'folds':checks,'final_params':pack['params'],'final_inner_auc':pack['inner_auc'],'G_not_loaded_in_training':True})
        print('TRAIN',key,'H OOF AUC',round(roc_auc_score(y,prob,sample_weight=w),5),'seconds',round(time.time()-start),flush=True)

def encoded_groups(pp):
    """Explicit mapping of numeric values/indicators and one-hot columns."""
    names=[]
    for block,pipe,cols in pp.transformers_:
        if block=='remainder':continue
        if block=='num':
            imp=pipe.named_steps['fill'];names.extend(cols)
            names.extend([cols[i] for i in imp.indicator_.features_])
        elif block=='cat':
            enc=pipe.named_steps['onehot']
            for c,categories in zip(cols,enc.categories_):names.extend([c]*len(categories))
        else: raise ValueError(block)
    assert len(names)==len(pp.get_feature_names_out())
    return names

def aggregate(vals,mapping,cols):
    return np.column_stack([vals[:,np.array(mapping)==c].sum(axis=1) for c in cols])

def explained(pack,d,training,algorithm,seed,path=False):
    cols=pack['features'];pp,m=pack['fitted'];a=pp.transform(d[cols]).astype('float32');mapping=encoded_groups(pp)
    rng=np.random.default_rng(seed)
    if algorithm=='XGBoost':
        import shap
        if path:
            explainer=shap.TreeExplainer(m,feature_perturbation='tree_path_dependent',model_output='raw');bgids=[]
        else:
            ids=rng.choice(len(training),128,replace=True,p=training.WTMEC2YR.to_numpy()/training.WTMEC2YR.sum())
            bg=pp.transform(training[cols].iloc[ids]).astype('float32')
            # Avoid SHAP's default background subsampling, which would silently
            # replace our survey-weighted empirical background by <=100 draws.
            masker=shap.maskers.Independent(bg,max_samples=len(bg))
            explainer=shap.TreeExplainer(m,data=masker,feature_perturbation='interventional',model_output='raw');bgids=training.SEQN.iloc[ids].tolist()
        vals=explainer.shap_values(a,check_additivity=True);base=float(explainer.expected_value);margin=m.predict(a,output_margin=True)
    else:
        assert all(len(t)==1 for t in m.term_features_),'EBM must be additive'
        termvals=m.eval_terms(a);vals=np.zeros_like(a,dtype=float)
        for k,t in enumerate(m.term_features_):vals[:,t[0]]+=termvals[:,k]
        base=float(np.asarray(m.intercept_).ravel()[0]);margin=m.decision_function(a);bgids=[]
    grouped=aggregate(vals,mapping,cols);err=float(np.max(np.abs(base+grouped.sum(axis=1)-margin)))
    perr=float(np.max(np.abs(expit(base+grouped.sum(axis=1))-m.predict_proba(a)[:,1])))
    assert err<2e-4 and perr<5e-5,(algorithm,err,perr)
    return grouped,base,{'margin_additivity_max_error':err,'probability_reconstruction_max_error':perr,'background_ids':bgids,'encoded_mapping':mapping}

def summarize(vals,d,cols,panel,algorithm,stage,method):
    w=d.WTMEC2YR.to_numpy();a=np.average(np.abs(vals),axis=0,weights=w);ranks=pd.Series(a).rank(ascending=False,method='min')
    return [{'panel':panel,'algorithm':algorithm,'stage':stage,'method':method,'variable':c,'label':LABELS[c],'weighted_mean_abs_logodds':a[j],'weighted_mean_signed_logodds':np.average(vals[:,j],weights=w),'rank':int(ranks[j])} for j,c in enumerate(cols)]

def explain():
    threadpool_limits(limits=2);H=data('H');G=data('G');fold=pd.read_csv(ROOT/'data/outer_folds.csv').fold.to_numpy();rows=[];checks=[];stability=[]
    (OUT/'explanations').mkdir(parents=True,exist_ok=True)
    for panel in ['Questionnaire','Expanded']:
      for algorithm in ['XGBoost','EBM']:
        key=panel+'_'+algorithm;cols=s.PANELS[panel];folder=ROOT/'results/models'/key;oof=np.zeros((len(H),len(cols)));bases=np.zeros(len(H));foldstats=[]
        for k in range(1,6):
            tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);pack=joblib.load(folder/f'fold{k}.joblib')
            assert pack['train_ids']==H.SEQN.iloc[tr].tolist()
            val,base,check=explained(pack,H.iloc[te],H.iloc[tr],algorithm,SEED+700+k)
            assert not set(check['background_ids'])&set(H.SEQN.iloc[te])
            oof[te]=val;bases[te]=base;check.update(key=key,stage='H_outer_holdout',fold=k);checks.append(check)
            rr=summarize(val,H.iloc[te],cols,panel,algorithm,'H_fold'+str(k),'interventional_SHAP' if algorithm=='XGBoost' else 'native_EBM');rows.extend(rr);foldstats.append([r['weighted_mean_abs_logodds'] for r in rr])
        rows.extend(summarize(oof,H,cols,panel,algorithm,'H_OOF','interventional_SHAP' if algorithm=='XGBoost' else 'native_EBM'))
        H[['SEQN']].join(pd.DataFrame(oof,columns=cols)).assign(base_logodds=bases).to_csv(OUT/'explanations'/f'{key}_H_OOF_contributions.csv',index=False)
        pack=joblib.load(folder/'final_model.joblib');val,base,check=explained(pack,G,H,algorithm,SEED+800);check.update(key=key,stage='G_external',fold=0);checks.append(check)
        method='interventional_SHAP' if algorithm=='XGBoost' else 'native_EBM';rows.extend(summarize(val,G,cols,panel,algorithm,'G_external',method))
        G[['SEQN']].join(pd.DataFrame(val,columns=cols)).assign(base_logodds=base).to_csv(OUT/'explanations'/f'{key}_G_contributions.csv',index=False)
        rho=spearmanr(np.average(np.abs(oof),axis=0,weights=H.WTMEC2YR),np.average(np.abs(val),axis=0,weights=G.WTMEC2YR)).statistic
        stability.append({'key':key,'comparison':'H_OOF_vs_G','spearman_importance':rho})
        for k,importance in enumerate(foldstats,1):stability.append({'key':key,'comparison':f'H_fold{k}_vs_G','spearman_importance':spearmanr(importance,np.average(np.abs(val),axis=0,weights=G.WTMEC2YR)).statistic})
        if algorithm=='XGBoost':
            pathval,pathbase,check=explained(pack,G,H,algorithm,SEED+800,path=True);check.update(key=key,stage='G_path_sensitivity',fold=0);checks.append(check)
            rows.extend(summarize(pathval,G,cols,panel,algorithm,'G_external','path_dependent_SHAP'))
            stability.append({'key':key,'comparison':'SHAP_interventional_vs_path','spearman_importance':spearmanr(np.average(np.abs(val),axis=0,weights=G.WTMEC2YR),np.average(np.abs(pathval),axis=0,weights=G.WTMEC2YR)).statistic})
        print('EXPLAIN complete',key,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'importance.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'importance_stability.csv',index=False);writejson(OUT/'explanation_checks.json',checks)
    shape_rows=[]
    for panel,variables in [('Questionnaire',s.Q),('Expanded',['RIDAGEYR','INDFMPIR','BMXBMI','BMXWAIST','SmokingStatus','LBXCOT','LBXWBCSI','LBXGH'])]:
        folder=ROOT/'results/models'/(panel+'_EBM');cols=s.PANELS[panel]
        reference={c:float(H[c].dropna().mode().iloc[0]) if c in s.u.CAT else float(H[c].median()) for c in cols}
        for c in variables:
            cat=c in s.u.CAT
            grid=np.array(s.u.CAT[c],float) if cat else np.unique(np.quantile(H[c].dropna(),np.linspace(.01,.99,100)))
            for k in range(0,6):
                pack=joblib.load(folder/('final_model.joblib' if k==0 else f'fold{k}.joblib'));pp,m=pack['fitted'];mapping=encoded_groups(pp)
                frame=pd.DataFrame([reference]*len(grid));frame[c]=grid;rf=pd.DataFrame([reference]);a=pp.transform(frame).astype('float32');ar=pp.transform(rf).astype('float32')
                terms=m.eval_terms(a);rt=m.eval_terms(ar)
                idx=[j for j,t in enumerate(m.term_features_) if mapping[t[0]]==c]
                contribution=terms[:,idx].sum(axis=1);delta=contribution-rt[:,idx].sum()
                # Only this variable's additive terms change. This checks the
                # dummy grouping against actual decision-function differences.
                assert np.max(abs(delta-(m.decision_function(a)-m.decision_function(ar))))<1e-8
                for x,z,raw in zip(grid,delta,contribution):shape_rows.append({'panel':panel,'variable':c,'label':LABELS[c],'fit':'final_H' if k==0 else f'outer{k}','value':x,'reference_value':reference[c],'delta_logodds_from_reference':z,'raw_grouped_term':raw,'categorical':cat})
    pd.DataFrame(shape_rows).to_csv(OUT/'ebm_shapes.csv',index=False)

def evaluate():
    threadpool_limits(limits=2);H=data('H');G=data('G');performance=[];pred={'H':{},'G':{}};comparisons=[];replay=[]
    for panel in ['Questionnaire','NonLab','NonLabBP','Expanded']:
      for algorithm in s.SPECS:
        key=panel+'_'+algorithm;folder=ROOT/'results/models'/key;cols=s.PANELS[panel]
        for cycle,d,fn in [('H',H,'internal_predictions.csv'),('G',G,'external_predictions.csv')]:
            rr=pd.read_csv(folder/fn);assert np.array_equal(rr.SEQN,d.SEQN);pred[cycle][key]=rr.prob.to_numpy()
        pack=joblib.load(folder/'final_model.joblib');p=s.predict(pack['fitted'],G[cols],algorithm)
        err=float(np.max(abs(p-pred['G'][key])));assert err<2e-6
        replay.append({'key':key,'fit':'baseline_final','max_abs_probability_error':err})
    for v in variants():
      for algorithm in s.SPECS:
        key=v['variant']+'_'+algorithm;folder=OUT/'models'/key;assert (folder/'completed.json').exists()
        rr=pd.read_csv(folder/'internal_predictions.csv');assert np.array_equal(rr.SEQN,H.SEQN);pred['H'][key]=rr.prob.to_numpy()
        pack=joblib.load(folder/'final_model.joblib');assert pack['features']==v['features']
        assert set(pack['train_ids'])==set(H.SEQN) and not set(pack['train_ids'])&set(G.SEQN)
        p=s.predict(pack['fitted'],G[v['features']],algorithm);pred['G'][key]=p
        G[['SEQN']].assign(y=target(G),prob=p).to_csv(folder/'external_predictions.csv',index=False)
        for k in range(1,6):
            pk=joblib.load(folder/f'fold{k}.joblib');ix=np.flatnonzero(rr.fold.eq(k));pr=s.predict(pk['fitted'],H[v['features']].iloc[ix],algorithm)
            err=float(np.max(abs(pr-pred['H'][key][ix])));assert err<2e-6
            assert not set(pk['train_ids'])&set(pk['test_ids'])
            tr=H.SEQN.isin(pk['train_ids']);te=H.SEQN.isin(pk['test_ids'])
            assert not set(s.group(H.loc[tr]))&set(s.group(H.loc[te]))
            assert pk['features']==v['features']
            assert list(pk['fitted'][0].feature_names_in_)==v['features']
            replay.append({'key':key,'fit':f'fold{k}','max_abs_probability_error':err})
        comparisons.append((key,v['panel']+'_'+algorithm,'deletion',','.join(v['removed'])))
    for algorithm in s.SPECS:
        comparisons.extend([('Questionnaire_'+algorithm,'NonLabBP_'+algorithm,'deletion','BMXBMI,BMXWAIST,BPXSY1,BPXDI1'),('NonLab_'+algorithm,'NonLabBP_'+algorithm,'deletion','BPXSY1,BPXDI1'),('NonLabBP_'+algorithm,'Expanded_'+algorithm,'deletion','all_36_laboratory_variables')])
        # Direct redundancy contrasts, same expanded panel and nested protocol.
        comparisons.extend([('Expanded_without_both_smoking_'+algorithm,'Expanded_without_cotinine_'+algorithm,'conditional_deletion','SmokingStatus_when_cotinine_absent'),('Expanded_without_both_smoking_'+algorithm,'Expanded_without_smoking_'+algorithm,'conditional_deletion','LBXCOT_when_smoking_absent')])
    pairs=[];rng=np.random.default_rng(SEED+931)
    for cycle,d in [('H',H),('G',G)]:
        y=target(d);w=d.WTMEC2YR.to_numpy();keys=list(pred[cycle]);vals=np.empty((1000,len(keys),2))
        # Compute AUC without sklearn's repeated input-validation overhead.
        orders={k:np.argsort(p,kind='stable') for k,p in pred[cycle].items()}
        ends={k:np.r_[np.flatnonzero(np.diff(pred[cycle][k][o]))+1,len(d)] for k,o in orders.items()}
        for b in range(1000):
            wb=w*s.multiplier(d,rng)
            for j,k in enumerate(keys):
                p=pred[cycle][k];o=orders[k];e=ends[k];cp=np.r_[0,np.cumsum(wb[o]*y[o])];cn=np.r_[0,np.cumsum(wb[o]*(1-y[o]))];tp=np.diff(cp[np.r_[0,e]]);tn=np.diff(cn[np.r_[0,e]])
                auc=np.sum(tp*(cn[e]-tn/2))/(cp[-1]*cn[-1]);vals[b,j]=[auc,np.average((y-p)**2,weights=wb)]
        for j,k in enumerate(keys):
            p=pred[cycle][k];lo,hi=np.quantile(vals[:,j],[.025,.975],axis=0)
            o=orders[k];e=ends[k];cp=np.r_[0,np.cumsum(w[o]*y[o])];cn=np.r_[0,np.cumsum(w[o]*(1-y[o]))];auc=np.sum(np.diff(cp[np.r_[0,e]])*(cn[e]-np.diff(cn[np.r_[0,e]])/2))/(cp[-1]*cn[-1]);assert abs(auc-roc_auc_score(y,p,sample_weight=w))<1e-12
            performance.append({'cycle':cycle,'key':k,'auc':auc,'auc_low':lo[0],'auc_high':hi[0],'brier':np.average((y-p)**2,weights=w),'brier_low':lo[1],'brier_high':hi[1]})
        for a,b,kind,removed in comparisons:
            ia=keys.index(a);ib=keys.index(b);delta=vals[:,ib]-vals[:,ia];lo,hi=np.quantile(delta,[.025,.975],axis=0)
            pairs.append({'cycle':cycle,'reduced_model':a,'full_model':b,'comparison':kind,'removed':removed,'auc_loss':roc_auc_score(y,pred[cycle][b],sample_weight=w)-roc_auc_score(y,pred[cycle][a],sample_weight=w),'auc_loss_low':lo[0],'auc_loss_high':hi[0],'brier_increase':np.average((y-pred[cycle][a])**2-(y-pred[cycle][b])**2,weights=w),'brier_increase_low':-hi[1],'brier_increase_high':-lo[1]})
        print('EVALUATE',cycle,'1000 design bootstrap complete',flush=True)
    pd.DataFrame(performance).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'deletion_comparisons.csv',index=False);pd.DataFrame(replay).to_csv(OUT/'prediction_replay_checks.csv',index=False)
    # Background-method importance intervals: sampling uncertainty conditional
    # on fixed G model, not uncertainty from development or background draws.
    imp=pd.read_csv(OUT/'importance.csv');ci=[];rng=np.random.default_rng(SEED+932)
    for panel in ['Questionnaire','Expanded']:
      for algorithm in ['XGBoost','EBM']:
        cols=s.PANELS[panel];arr=pd.read_csv(OUT/'explanations'/f'{panel}_{algorithm}_G_contributions.csv')[cols].to_numpy();boot=[]
        for b in range(1000):boot.append(np.average(abs(arr),axis=0,weights=G.WTMEC2YR.to_numpy()*s.multiplier(G,rng)))
        low,high=np.quantile(boot,[.025,.975],axis=0)
        for j,c in enumerate(cols):ci.append({'panel':panel,'algorithm':algorithm,'stage':'G_external','method':'interventional_SHAP' if algorithm=='XGBoost' else 'native_EBM','variable':c,'importance_low_conditional':low[j],'importance_high_conditional':high[j]})
    imp.merge(pd.DataFrame(ci),on=['panel','algorithm','stage','method','variable'],how='left').to_csv(OUT/'importance.csv',index=False)
    before=json.loads((OUT/'preservation_before.json').read_text(encoding='utf-8'));preserved={p:sha(ROOT/p)==h for p,h in before.items()};assert all(preserved.values());writejson(OUT/'preservation_after.json',preserved)
    writejson(OUT/'verification.json',{'all_original_data_models_results_unchanged':all(preserved.values()),'original_files_checked':len(preserved),'new_variants':len(variants())*3,'new_saved_model_fits':len(variants())*3*6,'expected_training_fits_including_inner_CV':len(variants())*3*42,'prediction_checks':len(replay),'prediction_max_error':max(r['max_abs_probability_error'] for r in replay),'external_not_used_for_training':True,'bootstrap_replicates_per_cycle':1000,'conditional_intervals_not_full_training_uncertainty':True})
    writejson(OUT/'runtime_versions.json',{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','interpret-core','shap','joblib','matplotlib']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['plan','train','explain','evaluate']);parser.add_argument('--algorithm',choices=list(s.SPECS));args=parser.parse_args()
    {'plan':plan,'train':lambda:train(args.algorithm),'explain':explain,'evaluate':evaluate}[args.mode]()
