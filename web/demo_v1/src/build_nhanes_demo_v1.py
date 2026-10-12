from pathlib import Path
import csv,json,math,hashlib,shutil,random,base64,io
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'web/demo_v1';OUT.mkdir(parents=True,exist_ok=True)
HERE=Path(__file__).resolve().parent
sources=[]
def read(rel):
    path=ROOT/rel;sources.append(path);rows=[]
    for row in csv.DictReader(path.open(encoding='utf-8-sig',newline='')):
        for k,v in row.items():
            if v=='':row[k]=None;continue
            try:
                val=float(v);row[k]=val if math.isfinite(val) else None
            except (ValueError,TypeError):pass
        rows.append(row)
    return rows
Q=['RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR','OHQ030','SmokingStatus']
N=Q+['BMXBMI','BMXWAIST']
CBC=['LBXEOPCT','LBXHGB','LBXLYPCT','LBXMCVSI','LBXMOPCT','LBXMPSI','LBXNEPCT','LBXPLTSI','LBXRDW','LBXWBCSI']
dictionary=read('data/data_dictionary.csv')
legacy=read('data/data_dictionary_legacy_features.csv');full=[r['variable'] for r in legacy if r['role']=='predictor']+['SmokingStatus']
labels={r['variable']:r['label'] or r['variable'] for r in dictionary}
labels.update({'RIDAGEYR':'年龄','RIAGENDR':'性别','RIDRETH3':'种族／族裔','DMDEDUC2':'教育程度','INDFMPIR':'收入贫困比','OHQ030':'距上次看牙时间','SmokingStatus':'吸烟状态','BMXBMI':'BMI','BMXWAIST':'腰围','BPXSY1':'首次收缩压','BPXDI1':'首次舒张压','LBXCOT':'血清可替宁','LBXWBCSI':'白细胞计数','LBXGH':'糖化血红蛋白','LBXTC':'总胆固醇','LBXSTR':'甘油三酯','LBXEOPCT':'嗜酸性粒细胞比例','LBXHGB':'血红蛋白','LBXLYPCT':'淋巴细胞比例','LBXMCVSI':'平均红细胞体积','LBXMOPCT':'单核细胞比例','LBXMPSI':'平均血小板体积','LBXNEPCT':'中性粒细胞比例','LBXPLTSI':'血小板计数','LBXRDW':'红细胞分布宽度','LBXSAL':'白蛋白','LBXSAPSI':'碱性磷酸酶','LBXSASSI':'天冬氨酸氨基转移酶','LBXSATSI':'丙氨酸氨基转移酶','LBXSBU':'尿素氮','LBXSC3SI':'碳酸氢盐','LBXSCA':'总钙','LBXSCK':'肌酸激酶','LBXSCLSI':'氯','LBXSCR':'肌酐','LBXSGB':'球蛋白','LBXSGTSI':'γ-谷氨酰转移酶','LBXSIR':'铁','LBXSKSI':'钾','LBXSLDSI':'乳酸脱氢酶','LBXSNASI':'钠','LBXSPH':'磷','LBXSTB':'总胆红素','LBXSUA':'尿酸','UACR':'尿白蛋白／肌酐比','LBDFOTSI':'血清总叶酸','LBXVIDMS':'总25-羟维生素D'})
perf=[r for r in read('results/performance.csv') if r['variant']=='raw']
pairs=read('results/paired_comparisons.csv')
importance=[r for r in read('explainability_20261011/importance.csv') if r['stage'] in ['H_OOF','G_external'] and r['method']!='path_dependent_SHAP']
deletions=read('explainability_20261011/deletion_comparisons.csv')
calibration=read('results/calibration_bins.csv')
subgroups=[r for r in read('results/subgroups.csv') if r['key']=='Questionnaire_Logistic' and r['subgroup']=='age']
rawshapes=read('explainability_20261011/ebm_shapes.csv');shapes={}
for r in rawshapes:
    key=r['panel']+'_'+r['variable']
    if key in shapes:continue
    sub=[z for z in rawshapes if z['panel']==r['panel'] and z['variable']==r['variable']]
    final=sorted([z for z in sub if z['fit']=='final_H'],key=lambda z:z['value']);xs=[z['value'] for z in final]
    folds={k:sorted([z for z in sub if z['fit']=='outer'+str(k)],key=lambda z:z['value']) for k in range(1,6)}
    assert all([z['value'] for z in f]==xs for f in folds.values())
    shapes[key]={'x':xs,'y':[z['delta_logodds_from_reference'] for z in final],
                 'low':[min(f[i]['delta_logodds_from_reference'] for f in folds.values()) for i in range(len(xs))],
                 'high':[max(f[i]['delta_logodds_from_reference'] for f in folds.values()) for i in range(len(xs))],
                 'reference':r['reference_value'],'cat':r['categorical']=='True'}
report=ROOT/'docs/完整研究稿_方法结果讨论.md';sources.append(report)
scatter={}
for panel in ['Questionnaire','Expanded']:
    for cycle in ['G','H']:
        stage='G_external' if cycle=='G' else 'H_OOF'
        ranked=sorted([r for r in importance if r['panel']==panel and r['algorithm']=='XGBoost' and r['stage']==stage],key=lambda r:-r['weighted_mean_abs_logodds'])
        variables=Q if panel=='Questionnaire' else list(dict.fromkeys([r['variable'] for r in ranked[:10]]+['RIDAGEYR','INDFMPIR','SmokingStatus','OHQ030','LBXCOT','LBXWBCSI','LBXGH']))
        raw=read(f'data/analysis_{cycle}.csv');byid={r['SEQN']:r for r in raw}
        contributions=read(f'explainability_20261011/explanations/{panel}_XGBoost_{"G" if cycle=="G" else "H_OOF"}_contributions.csv')
        assert len({r['SEQN'] for r in contributions})==len(contributions)
        assert all(r['SEQN'] in byid for r in contributions)
        indices=sorted(random.Random(20261011).sample(range(len(contributions)),min(900,len(contributions))))
        values={}
        for v in variables:
            points=[]
            for i in indices:
                r=contributions[i];original=byid[r['SEQN']][v]
                points.append([None if original is None else round(original,4),round(r[v],6)])
            values[v]=points
        scatter[panel+'_'+cycle]={'n':len(contributions),'shown':len(indices),'order':[r['variable'] for r in ranked[:7 if panel=='Questionnaire' else 10]],'values':values}
D={'performance':perf,'pairs':pairs,'importance':importance,'deletions':deletions,'calibration':calibration,'subgroups':subgroups,'shapes':shapes,'labels':labels,'questionnaire':Q,'panels':{'Questionnaire':Q,'NonLab':N,'NonLabBP':N+['BPXSY1','BPXDI1'],'CBC':N+CBC,'Expanded':full},'report':report.read_text(encoding='utf-8')}
D['scatter']=scatter
html=(HERE/'nhanes_demo_v1.html').read_text(encoding='utf-8').replace('__DATA__',json.dumps(D,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/'))
font_path=Path('C:/Windows/Fonts/NotoSansSC-VF.ttf')
if '__FONT__' in html and font_path.exists():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    opts=subset.Options();opts.flavor='woff'
    font=TTFont(font_path)
    sub=subset.Subsetter(options=opts);sub.populate(text=html);sub.subset(font)
    font.flavor='woff';buffer=io.BytesIO();font.save(buffer)
    html=html.replace('__FONT__',base64.b64encode(buffer.getvalue()).decode('ascii'))
html=html.replace('__FONT__','')
assert '<img' not in html.lower() and 'data:image/png' not in html and '<table>' in html
(OUT/'index.html').write_text(html,encoding='utf-8')
(OUT/'data_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('# NHANES demo\n\n双击 index.html 即可离线打开。可切换模型、信息面板和评估周期，查看模型表现、SHAP、EBM 曲线及年龄分组结果。\n\n[项目仓库](https://github.com/dissein/NHANES-Periodontitis-Analysis)\n\n重新生成：在仓库根目录运行 `python web/demo_v1/src/build_nhanes_demo_v1.py`。\n',encoding='utf-8')
(OUT/'src').mkdir(exist_ok=True)
for name in ['build_nhanes_demo_v1.py','nhanes_demo_v1.html']:
    dest=OUT/'src'/name
    if (HERE/name).resolve()!=dest.resolve():shutil.copy2(HERE/name,dest)
print(OUT/'index.html')
print('Bytes:',len(html.encode('utf-8')),'native charts:',len(shapes),'result rows:',len(perf))
