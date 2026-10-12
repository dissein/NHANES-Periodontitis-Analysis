"""Exploratory extension. All architecture/parameter decisions are made in H only."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.metadata,copy
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from rtdl_revisiting_models import FTTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score,roc_curve
from scipy.special import logit,expit
from scipy.optimize import minimize

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'transformer_trial_20261011'
OUT.mkdir(exist_ok=True)
(OUT/'models').mkdir(exist_ok=True)
SEED=20261010
Q=['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','OHQ030','SmokingStatus']
FULL=pd.read_csv(BASE/'data/data_dictionary_legacy_features.csv').query('role=="predictor"').variable.tolist()+['SmokingStatus']
PANELS={'Questionnaire':Q,'NonLabBP':Q+['BMXBMI','BMXWAIST','BPXSY1','BPXDI1'],'Expanded':FULL}
CAT={'RIAGENDR':[1,2],'RIDRETH3':[1,2,3,4,6,7],'DMDEDUC2':[1,2,3,4,5],'OHQ030':[1,2,3,4,5,6,7],'SmokingStatus':[0,1,2]}
SKEW=['LBXCOT','UACR','LBXSASSI','LBXSATSI','LBXSCK','LBXSGTSI','LBXSTR','LBXSAPSI','LBXWBCSI']
VARIANTS=['FTStandard','FTPiecewise','FTPiecewiseMissing']
CONFIGS=[{'width':32,'lr':.001,'epochs':40},{'width':64,'lr':.0005,'epochs':40}]
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def groups(d):return d.SDMVSTRA.astype(int).astype(str)+'_'+d.SDMVPSU.astype(int).astype(str)
def y_of(d):return d.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy()
def numeric_values(d,cols):
    a=d[cols].to_numpy(dtype=float,copy=True)
    for i,c in enumerate(cols):
        if c in SKEW:a[:,i]=np.log1p(np.maximum(a[:,i],0))
    a[~np.isfinite(a)]=np.nan
    return a

class Preprocess:
    def fit(self,d,cols):
        self.num=[c for c in cols if c not in CAT];self.cat=[c for c in cols if c in CAT]
        a=numeric_values(d,self.num)
        self.median=np.nanmedian(a,axis=0);self.median=np.nan_to_num(self.median)
        a=np.where(np.isnan(a),self.median,a)
        self.mean=a.mean(0);self.std=a.std(0);self.std[self.std<1e-6]=1
        z=np.clip((a-self.mean)/self.std,-10,10)
        self.edges=np.zeros((len(self.num),9),dtype=np.float32)
        for j in range(len(self.num)):
            observed=np.isfinite(numeric_values(d,self.num)[:,j])
            e=np.quantile(z[observed,j],np.linspace(0,1,9)) if observed.any() else np.linspace(-1,1,9)
            # Deduplicate without changing interval count: quantile plateaus are
            # given epsilon width and receive no artificial nonzero observations.
            for k in range(1,len(e)):e[k]=max(e[k],e[k-1]+1e-5)
            self.edges[j]=e
        self.cardinalities=[len(CAT[c])+1 for c in self.cat]
        return self
    def transform(self,d):
        a=numeric_values(d,self.num);missing=np.isnan(a).astype(np.float32)
        a=np.where(np.isnan(a),self.median,a);a=np.clip((a-self.mean)/self.std,-10,10).astype(np.float32)
        cat=np.column_stack([d[c].map({v:i+1 for i,v in enumerate(CAT[c])}).fillna(0).to_numpy(np.int64) for c in self.cat])
        return torch.from_numpy(a),torch.from_numpy(cat),torch.from_numpy(missing)

class PiecewiseEmbedding(nn.Module):
    def __init__(self,edges,width):
        super().__init__();self.register_buffer('edges',torch.as_tensor(edges))
        self.weight=nn.Parameter(torch.empty(len(edges),8,width));self.bias=nn.Parameter(torch.zeros(len(edges),width))
        nn.init.normal_(self.weight,std=.02)
    def forward(self,x):
        e=self.edges
        a=((x[:,:,None]-e[None,:,:-1])/(e[:,1:]-e[:,:-1])[None,:,:]).clamp(0,1)
        return torch.einsum('bfk,fkd->bfd',a,self.weight)+self.bias

class TrialModel(nn.Module):
    def __init__(self,pp,variant,config):
        super().__init__();d=config['width'];self.variant=variant
        self.ft=FTTransformer(n_cont_features=len(pp.num),cat_cardinalities=pp.cardinalities,d_out=1,n_blocks=2,d_block=d,attention_n_heads=4,attention_dropout=.1,ffn_d_hidden=None,ffn_d_hidden_multiplier=4/3,ffn_dropout=.1,residual_dropout=0.)
        if variant!='FTStandard':self.ft.cont_embeddings=PiecewiseEmbedding(pp.edges,d)
        if variant=='FTPiecewiseMissing':
            self.missing_embedding=nn.Parameter(torch.zeros(len(pp.num),d))
            nn.init.normal_(self.missing_embedding,std=.02)
    def forward(self,a,c,m):
        if self.variant!='FTPiecewiseMissing':return self.ft(a,c).squeeze(-1)
        tokens=[self.ft.cls_embedding(a.shape[:-1]),self.ft.cont_embeddings(a)+m[:,:,None]*self.missing_embedding,self.ft.cat_embeddings(c)]
        return self.ft.backbone(torch.cat(tokens,dim=1)).squeeze(-1)

def train(d,cols,variant,config,seed):
    torch.manual_seed(seed);np.random.seed(seed)
    pp=Preprocess().fit(d,cols);a,c,m=[v.to(DEVICE) for v in pp.transform(d)]
    model=TrialModel(pp,variant,config).to(DEVICE)
    optimizer=torch.optim.AdamW(model.ft.make_parameter_groups()+([{'params':[model.missing_embedding]}] if variant=='FTPiecewiseMissing' else []),lr=config['lr'],weight_decay=.001)
    y=torch.tensor(y_of(d),dtype=torch.float32,device=DEVICE);w=torch.tensor(d.WTMEC2YR.to_numpy()/d.WTMEC2YR.mean(),dtype=torch.float32,device=DEVICE)
    rng=torch.Generator().manual_seed(seed)
    losses=[]
    for epoch in range(config['epochs']):
        model.train();order=torch.randperm(len(d),generator=rng);epochloss=0
        for ids in order.split(512):
            ids=ids.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits=model(a[ids],c[ids],m[ids]);loss=(F.binary_cross_entropy_with_logits(logits,y[ids],reduction='none')*w[ids]).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();epochloss+=float(loss.detach())*len(ids)
        losses.append(epochloss/len(d))
    model.eval()
    return {'preprocess':pp.__dict__,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'variant':variant,'config':config,'seed':seed,'train_ids':d.SEQN.tolist(),'losses':losses,'parameter_count':sum(p.numel() for p in model.parameters())}

def predict(pack,d):
    pp=Preprocess();pp.__dict__.update(pack['preprocess']);model=TrialModel(pp,pack['variant'],pack['config']).to(DEVICE);model.load_state_dict(pack['state']);model.eval()
    a,c,m=[v.to(DEVICE) for v in pp.transform(d)]
    with torch.no_grad():p=torch.cat([torch.sigmoid(model(a[i:i+512],c[i:i+512],m[i:i+512])) for i in range(0,len(d),512)]).cpu().numpy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p

def select(d,cols,variant,seed):
    inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(d,y_of(d),groups(d)));scores=[]
    for ci,config in enumerate(CONFIGS):
        p=np.zeros(len(d));fs=[]
        for k,(tr,va) in enumerate(inner):
            assert not set(groups(d.iloc[tr]))&set(groups(d.iloc[va]))
            pack=train(d.iloc[tr],cols,variant,config,seed+k)
            p[va]=predict(pack,d.iloc[va]);fs.append(roc_auc_score(y_of(d.iloc[va]),p[va],sample_weight=d.WTMEC2YR.iloc[va]))
        scores.append({'config_index':ci,'mean_inner_auc':float(np.mean(fs)),'inner_predictions':p})
    best=max(scores,key=lambda x:x['mean_inner_auc'])
    return CONFIGS[best['config_index']],best['inner_predictions'],[{k:v for k,v in s.items() if k!='inner_predictions'} for s in scores]

def metric(d,p):
    y=y_of(d);w=d.WTMEC2YR.to_numpy();x=logit(np.clip(p,1e-6,1-1e-6));ww=w/w.sum()
    obj=lambda b:np.dot(ww,np.logaddexp(0,b[0]+b[1]*x)-y*(b[0]+b[1]*x))
    sol=minimize(obj,[0.,1.],method='BFGS')
    assert sol.success or np.linalg.norm(sol.jac)<1e-4
    return {'n':len(d),'auc':roc_auc_score(y,p,sample_weight=w),'brier':brier_score_loss(y,p,sample_weight=w),'ap':average_precision_score(y,p,sample_weight=w),'calibration_intercept':sol.x[0],'calibration_slope':sol.x[1]}

def thresholds(d,p):
    f,t,c=roc_curve(y_of(d),p,sample_weight=d.WTMEC2YR)
    return {str(z):float(c[np.isfinite(c)&(c<=1)&(t>=z)].max()) for z in [.8,.9]}

def boot_multiplier(d,design,rng):
    out=np.zeros(len(d))
    for s,units in design.items():
        m=len(units)
        for u in rng.choice(units,size=m-1,replace=True):out[(d.SDMVSTRA.to_numpy()==s)&(d.SDMVPSU.to_numpy()==u)]+=m/(m-1)
    return out

def main():
    start=time.time()
    plan={'seed':SEED,'variants':VARIANTS,'panels':PANELS,'candidates':CONFIGS,'blocks':2,'heads':4,'batch_size':512,'inner_folds':3,'outer_folds':'same saved 5 folds as main study','selection':'mean weighted inner fold AUC; no G selection','epochs':'fixed 40, no evaluation early stopping','bootstrap':1000,'exploratory':True,'G_already_examined_in_previous_study':True,'preprocessing':'train-only median fill, log1p selected skew variables, StandardScaler-equivalent mean/std, clip z to [-10,10], training quantile 8-bin piecewise encoding','architecture_source':'official rtdl_revisiting_models.FTTransformer, compact nondefault configuration','prior_results_unchanged':True}
    planpath=OUT/'analysis_plan.json'
    if planpath.exists():assert json.loads(planpath.read_text())==plan
    else:planpath.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    d=pd.read_csv(BASE/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True)
    folds=pd.read_csv(BASE/'data/outer_folds.csv').set_index('SEQN').fold.reindex(d.SEQN).to_numpy();assert len(folds)==len(d) and not pd.isna(folds).any()
    # Fit and choose all models in H before opening G for this experiment.
    active=PANELS
    if '--panel' in sys.argv:
        chosen=sys.argv[sys.argv.index('--panel')+1];active={chosen:PANELS[chosen]}
    if '--evaluate-only' in sys.argv:active={}
    for panel,cols in active.items():
      for variant in VARIANTS:
        folder=OUT/'models'/f'{panel}_{variant}';folder.mkdir(exist_ok=True)
        for fold in range(1,6):
            path=folder/f'fold{fold}.pt'
            if path.exists():continue
            tr=np.where(folds!=fold)[0];va=np.where(folds==fold)[0]
            assert not set(groups(d.iloc[tr]))&set(groups(d.iloc[va]))
            config,ip,log=select(d.iloc[tr],cols,variant,SEED+fold)
            pack=train(d.iloc[tr],cols,variant,config,SEED+fold);pack.update({'validation_ids':d.SEQN.iloc[va].tolist(),'selection':log,'thresholds':thresholds(d.iloc[tr],ip)})
            torch.save(pack,path)
            p=predict(pack,d.iloc[va]);print(panel,variant,'fold',fold,'auc',round(metric(d.iloc[va],p)['auc'],4),'elapsed',round(time.time()-start),flush=True)
        if not (folder/'final.pt').exists():
            config,ip,log=select(d,cols,variant,SEED+100)
            pack=train(d,cols,variant,config,SEED+100);pack.update({'selection':log,'thresholds':thresholds(d,ip)})
            torch.save(pack,folder/'final.pt')
        print('H complete',panel,variant,flush=True)
    if '--train-only' in sys.argv:return
    assert all((OUT/'models'/f'{p}_{v}'/'final.pt').exists() for p in PANELS for v in VARIANTS)
    evaluation_start=time.time()
    ext=pd.read_csv(BASE/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
    assert not set(d.SEQN)&set(ext.SEQN)
    preds={'Internal':{},'External':{}};rows=[]
    for panel in PANELS:
      for variant in VARIANTS:
        key=f'{panel}_{variant}';folder=OUT/'models'/key
        p=np.zeros(len(d));cut=np.zeros(len(d))
        for f in range(1,6):
            pack=torch.load(folder/f'fold{f}.pt',weights_only=False);ids=np.where(folds==f)[0]
            assert set(pack['validation_ids'])==set(d.SEQN.iloc[ids])
            p[ids]=predict(pack,d.iloc[ids]);cut[ids]=pack['thresholds']['0.8']
        final=torch.load(folder/'final.pt',weights_only=False);pe=predict(final,ext)
        for stage,df,pr,cuts in [('Internal',d,p,cut),('External',ext,pe,np.repeat(final['thresholds']['0.8'],len(ext)))]:
            preds[stage][key]=pr
            df[['SEQN','SDMVSTRA','SDMVPSU','WTMEC2YR']].assign(y=y_of(df),prob=pr,cut80=cuts).to_csv(folder/(stage.lower()+'_predictions.csv'),index=False)
            rows.append({'stage':stage,'key':key,'panel':panel,'algorithm':variant,**metric(df,pr),'parameter_count':final['parameter_count']})
        for ref in ['Logistic','EBM','XGBoost']:
            for stage in preds:
                if f'{panel}_{ref}' not in preds[stage]:
                    file=BASE/'results/models'/f'{panel}_{ref}'/('internal_predictions.csv' if stage=='Internal' else 'external_predictions.csv')
                    r=pd.read_csv(file).set_index('SEQN');df=d if stage=='Internal' else ext
                    preds[stage][f'{panel}_{ref}']=r.prob.reindex(df.SEQN).to_numpy()
    pd.DataFrame(rows).to_csv(OUT/'performance.csv',index=False)
    pairs=[];intervals=[]
    for stage,df,cycle in [('Internal',d,'H'),('External',ext,'G')]:
        demo=pd.read_sas(BASE/'data/raw'/f'DEMO_{cycle}.xpt',format='xport');demo=demo.loc[demo.WTMEC2YR>0]
        design={s:np.unique(z.SDMVPSU) for s,z in demo.groupby('SDMVSTRA')}
        rng=np.random.default_rng(SEED+811);v={k:[] for k in preds[stage]}
        for b in range(1000):
            w=df.WTMEC2YR.to_numpy()*boot_multiplier(df,design,rng)
            for k,p in preds[stage].items():v[k].append([roc_auc_score(y_of(df),p,sample_weight=w),brier_score_loss(y_of(df),p,sample_weight=w)])
        for k in v:
            v[k]=np.array(v[k]);lo,hi=np.quantile(v[k],[.025,.975],axis=0)
            intervals.append({'stage':stage,'key':k,'auc_low':lo[0],'auc_high':hi[0],'brier_low':lo[1],'brier_high':hi[1]})
        specs=[]
        for panel in PANELS:
            for variant in VARIANTS:
                specs.append((f'{panel}_{variant}',f'{panel}_Logistic','vs_Logistic'))
            for variant in VARIANTS[1:]:specs.append((f'{panel}_{variant}',f'{panel}_FTStandard','architecture_change'))
        for variant in VARIANTS:specs.append((f'Expanded_{variant}',f'NonLabBP_{variant}','laboratory_increment'))
        for a,b,kind in specs:
            lo,hi=np.quantile(v[a]-v[b],[.025,.975],axis=0);ma=metric(df,preds[stage][a]);mb=metric(df,preds[stage][b])
            pairs.append({'stage':stage,'model':a,'reference':b,'comparison':kind,'delta_auc':ma['auc']-mb['auc'],'auc_low':lo[0],'auc_high':hi[0],'delta_brier':ma['brier']-mb['brier'],'brier_low':lo[1],'brier_high':hi[1]})
    pd.DataFrame(intervals).to_csv(OUT/'conditional_intervals.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
    versions={x:importlib.metadata.version(x) for x in ['torch','rtdl-revisiting-models','numpy','pandas','scikit-learn','scipy']}
    (OUT/'run_metadata.json').write_text(json.dumps({'completed':True,'evaluation_seconds':time.time()-evaluation_start,'experiment_elapsed_seconds':time.time()-planpath.stat().st_mtime,'versions':versions,'device':str(DEVICE),'conditional_bootstrap_only':True,'saved_final_models':9,'saved_outer_models':45,'total_inner_training_fits':324,'all_final_decisions_H_only':True},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['stage','key','auc','brier']].to_string(index=False),flush=True)

def smoke():
    start=time.time();d=pd.read_csv(BASE/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True)
    sample=d.iloc[:512].copy()
    # Preprocessor must not learn from prediction rows, and unknown/missing
    # categories must stay in the reserved category instead of becoming NaN.
    pp=Preprocess().fit(sample,Q)
    altered=sample.iloc[:8].copy();altered['RIDAGEYR']=999;altered['SmokingStatus']=999
    before=pp.mean.copy();a,c,m=pp.transform(altered)
    assert np.array_equal(before,pp.mean) and (c[:,-1]==0).all() and torch.isfinite(a).all()
    for variant in VARIANTS:
        pack=train(sample,Q,variant,{'width':32,'lr':.001,'epochs':3},SEED)
        p=predict(pack,altered)
        check=OUT/'smoke.pt';torch.save(pack,check)
        replay=predict(torch.load(check,weights_only=False),altered)
        assert np.array_equal(p,replay)
        assert np.isfinite(pack['losses']).all()
    print('Boundary/replay smoke checks passed; seconds',round(time.time()-start,2),flush=True)
    # Timing one actual candidate on H only. Does not choose models using G.
    pack=train(d,Q,'FTStandard',CONFIGS[0],SEED)
    print('Full H 40-epoch Q benchmark total seconds',round(time.time()-start,2),'loss',pack['losses'][0],pack['losses'][-1],flush=True)

if __name__=='__main__':
    if '--smoke' in sys.argv:smoke()
    else:main()
