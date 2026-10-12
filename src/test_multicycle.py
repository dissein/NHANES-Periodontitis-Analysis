from pathlib import Path
import sys,unittest,json,hashlib
import numpy as np
import pandas as pd
import joblib
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'src'))
import study_multicycle as s
from label_utils import clean,label,TEETH
from build_multicycle import smoking

def mock(sites):
    d={f'OHX{t:02d}{kind}{c}':0. for t in TEETH for kind in ['LA','PC'] for c in 'DSPA'}
    for tooth,c,al,pd_ in sites:d[f'OHX{tooth:02d}LA{c}']=al;d[f'OHX{tooth:02d}PC{c}']=pd_
    return pd.DataFrame([d])

class StudyTests(unittest.TestCase):
    def test_label_hierarchy(self):
        self.assertEqual(label(mock([(2,'D',6,5),(3,'S',6,2)])).iloc[0],'Severe')
        self.assertNotEqual(label(mock([(2,'D',6,5),(2,'S',6,2)])).iloc[0],'Severe')
        self.assertEqual(label(mock([(2,'D',0,5),(3,'S',0,5)])).iloc[0],'Moderate')
    def test_mild_standalone(self):self.assertEqual(label(mock([(2,'D',0,5)])).iloc[0],'Mild')
    def test_mild_site_not_tooth_count(self):self.assertEqual(label(mock([(2,'D',3,4),(2,'S',3,2),(3,'D',0,4)])).iloc[0],'Mild')
    def test_missing_label(self):
        d=mock([]);d[:]=99;self.assertIsNone(label(d).iloc[0]);d[:]=np.nan;self.assertIsNone(label(d).iloc[0])
    def test_special_missing(self):
        a=clean(pd.DataFrame({'v':[0.,5e-79,1.]}));self.assertEqual(a.v.iloc[0],0);self.assertTrue(pd.isna(a.v.iloc[1]))
    def test_smoking_skip(self):
        a=smoking(pd.Series([2,1,1,1,7,1]),pd.Series([np.nan,3,1,2,1,9]));np.testing.assert_allclose(a,[0,1,2,2,np.nan,np.nan],equal_nan=True)
    def test_panels_are_nested_no_leakage(self):
        self.assertTrue(set(s.Q)<=set(s.N)<=set(s.PANELS['NonLabBP'])<=set(s.FULL))
        self.assertTrue(set(s.N)<=set(s.PANELS['CBC'])<=set(s.FULL))
        forbidden={'NaturalTeeth','PairedSites','PairedTeeth','Cycle','SEQN','Periodontitis_Status','WTMEC2YR','SDMVPSU','SDMVSTRA'}
        for p in s.PANELS.values():self.assertFalse(set(p)&forbidden);self.assertEqual(len(p),len(set(p)))
    def test_threshold_target(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.3,.6,.8]);w=np.ones(4);cut=s.target_cut(y,p,w,.8)
        self.assertEqual(cut,.6);self.assertGreaterEqual(s.operating(y,p,w,cut)['sensitivity'],.8)
    def test_no_overlapping_cycles(self):
        H=pd.read_csv(OUT/'data/analysis_H.csv');G=pd.read_csv(OUT/'data/analysis_G.csv');self.assertFalse(set(H.SEQN)&set(G.SEQN));self.assertEqual(len(H),3617);self.assertEqual(len(G),3323)
    def test_full_design_bootstrap_three_PSU(self):
        G=pd.read_csv(OUT/'data/analysis_G.csv');rng=np.random.default_rng(77)
        for b in range(10):
            z=s.multiplier(G,rng);self.assertTrue(set(np.unique(z))<={0.,1.5,2.,3.})
            for _,a in G.assign(mult=z).groupby(['SDMVSTRA','SDMVPSU']):self.assertEqual(a.mult.nunique(),1)
    def test_calibration_interval_scope(self):
        a=pd.read_csv(OUT/'results/performance.csv');self.assertTrue(a.loc[a.variant!='raw','auc_low'].isna().all())
    def test_original_H_reconciliation(self):
        path=OUT.parent/'05_全面复核与修订_20261010/data/analysis_rebuilt.csv'
        if not path.exists():self.skipTest('Standalone copy: historical H file not present')
        a=pd.read_csv(path).sort_values('SEQN').reset_index(drop=True)
        b=pd.read_csv(OUT/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True);pd.testing.assert_frame_equal(a,b[a.columns],check_dtype=False)

def verify_models():
    H=pd.read_csv(OUT/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True);G=pd.read_csv(OUT/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
    records=[]
    for panel in s.PANELS:
      for name in s.SPECS:
        key=panel+'_'+name;folder=OUT/'results/models'/key;ir=pd.read_csv(folder/'internal_predictions.csv');er=pd.read_csv(folder/'external_predictions.csv');cols=s.PANELS[panel]
        for fold in range(1,6):
            pack=joblib.load(folder/f'fold{fold}.joblib');ix=np.flatnonzero(ir.fold.eq(fold));ids=H.SEQN.iloc[ix].tolist()
            assert ids==pack['test_ids'];assert not set(ids)&set(pack['train_ids'])
            tr=H.loc[H.SEQN.isin(pack['train_ids'])];te=H.iloc[ix];assert not set(s.group(tr))&set(s.group(te))
            p=s.predict(pack['fitted'],H[cols].iloc[ix],name);err=float(np.max(abs(p-ir.prob.iloc[ix].to_numpy())));assert err<1e-6
            y=tr.Periodontitis_Status.isin(['Moderate','Severe']).astype(int).to_numpy();w=tr.WTMEC2YR.to_numpy();cut=s.target_cut(y,pack['inner_probability'],w,.8)
            np.testing.assert_allclose(ir.cut80.iloc[ix],cut,rtol=1e-10,atol=1e-10)
            records.append({'key':key,'stage':f'fold{fold}','max_prediction_error':err})
        pack=joblib.load(folder/'final_model.joblib');assert not set(pack['train_ids'])&set(G.SEQN)
        p=s.predict(pack['fitted'],G[cols],name);err=float(np.max(abs(p-er.prob.to_numpy())));assert err<1e-6
        records.append({'key':key,'stage':'external','max_prediction_error':err})
    a=pd.DataFrame(records);a.to_csv(OUT/'results/model_replay_checks.csv',index=False)
    print('Replayed',len(a),'saved model assessments; max error',a.max_prediction_error.max())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StudyTests))
    if not result.wasSuccessful():sys.exit(1)
    verify_models()
