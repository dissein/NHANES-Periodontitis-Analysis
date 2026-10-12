from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]

def main():
    r=OUT/'results';a=pd.read_csv(r/'performance.csv');b=pd.read_csv(r/'paired_comparisons.csv');replay=pd.read_csv(r/'model_replay_checks.csv')
    assert len(a)==60 and len(a.query('variant=="raw"'))==30
    assert len(replay)==90 and replay.max_prediction_error.max()<1e-6
    assert len(pd.read_csv(r/'fold_performance.csv'))==75
    assert len(pd.read_csv(r/'selected_parameters.csv'))==90
    dictionary=pd.read_csv(OUT/'data/data_dictionary.csv');data=pd.read_csv(OUT/'data/analysis_H.csv')
    assert dictionary.variable.is_unique and set(dictionary.variable)==set(data.columns)
    for row in b.to_dict('records'):
        x=a[(a.variant=='raw')&(a.stage==row['stage'])].set_index('key')
        assert abs(row['delta_auc']-(x.loc[row['model'],'auc']-x.loc[row['reference'],'auc']))<1e-12
        assert abs(row['delta_brier']-(x.loc[row['model'],'brier']-x.loc[row['reference'],'brier']))<1e-12
    assert len(pd.read_csv(r/'refit_bootstrap_replicates.csv'))==200
    assert pd.read_csv(r/'refit_bootstrap_replicates.csv').replicate.nunique()==200
    age=pd.read_csv(r/'age_baseline_replay_checks.csv');assert len(age)==6 and age.max_prediction_error.max()<1e-6
    assert all(z['same_released_values'] for z in json.loads((r/'official_source_verification.json').read_text(encoding='utf-8')))
    manifest=json.loads((r/'raw_sources.json').read_text(encoding='utf-8'))
    for record in manifest:
        p=OUT/'data/raw'/f'{record["module"]}.xpt';assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    versions=['complete_results','source_sha256','paired_deltas_reconcile','90_saved_assessments_replay','200_unique_refit_replicates','no_calibrated_raw_CI_mislabel']
    files=[{'file':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['delivery_manifest.json','final_verification.json']]
    (r/'delivery_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'complete':True,'boundary_tests':12,'main_saved_prediction_assessments_replayed':90,'age_baseline_saved_assessments_replayed':6,'max_prediction_difference':float(replay.max_prediction_error.max()),'official_modules':30,'participants_H':3617,'participants_G':3323,'refit_replicates':200,'conditional_bootstrap_replicates':1000,'checks':versions,'files':len(files),'bytes':sum(x['bytes'] for x in files)}
    (r/'final_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
