from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import transformer_trial as t

o=t.OUT
p=pd.read_csv(o/'performance.csv')
ci=pd.read_csv(o/'conditional_intervals.csv')
old=pd.read_csv(t.BASE/'results/performance.csv').query('variant=="raw" and panel in ["Questionnaire","NonLabBP","Expanded"]')
combined=pd.concat([p,old],ignore_index=True)
combined=combined.drop(columns=['auc_low','auc_high','brier_low','brier_high'],errors='ignore').merge(ci,on=['stage','key'],validate='one_to_one')
combined.to_csv(o/'all_model_comparison.csv',index=False)
labels={'Logistic':'Logistic','EBM':'EBM','XGBoost':'XGBoost','FTStandard':'标准 FT','FTPiecewise':'分段数值编码 FT','FTPiecewiseMissing':'分段编码 + 缺失标记 FT'}
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,5))
algs=list(labels)
for ax,panel,title in zip(axes,t.PANELS,['问卷 7 项','含血压 11 项','扩展 47 项']):
    sub=combined.query('stage=="External" and panel==@panel').set_index('algorithm').loc[algs]
    for j,(alg,r) in enumerate(sub.iterrows()):
        color={'Logistic':'#182b36','EBM':'#a4aaa8','XGBoost':'#a4aaa8','FTStandard':'#317b77','FTPiecewise':'#b56449','FTPiecewiseMissing':'#a58b48'}[alg]
        ax.errorbar(r.auc,5-j,xerr=[[r.auc-r.auc_low],[r.auc_high-r.auc]],fmt='o',color=color,capsize=3)
        ax.text(.862,5-j,f'{r.auc:.3f}',va='center',fontsize=9)
    ax.set_yticks(range(6),[labels[a] for a in algs[::-1]])
    ax.set_xlim(.60,.89);ax.set_ylim(-.7,5.7);ax.set_title(title);ax.set_xlabel('G 周期加权 AUC · 条件 95% 区间');ax.grid(axis='x',alpha=.18)
fig.suptitle('Transformer 首轮实验：跨周期验证',fontsize=17);fig.tight_layout();fig.savefig(o/'comparison_external.png',dpi=160);fig.savefig(o/'comparison_external.pdf');plt.close(fig)

folds=pd.read_csv(t.BASE/'data/outer_folds.csv').set_index('SEQN').fold
h=pd.read_csv(t.BASE/'data/analysis_H.csv').sort_values('SEQN').reset_index(drop=True)
g=pd.read_csv(t.BASE/'data/analysis_G.csv').sort_values('SEQN').reset_index(drop=True)
checks=[];lossrows=[];maximum=0
preserved=json.loads((o/'main_study_preservation.json').read_text())
assert all(hashlib.sha256((t.BASE/f).read_bytes()).hexdigest()==digest for f,digest in preserved.items())
for panel in t.PANELS:
 for variant in t.VARIANTS:
    key=panel+'_'+variant;folder=o/'models'/key
    saved=pd.read_csv(folder/'internal_predictions.csv').set_index('SEQN')
    for f in range(1,6):
        pack=torch.load(folder/f'fold{f}.pt',weights_only=False)
        tr=h[h.SEQN.isin(pack['train_ids'])];va=h[h.SEQN.isin(pack['validation_ids'])]
        assert not set(t.groups(tr))&set(t.groups(va))
        assert not set(pack['train_ids'])&set(pack['validation_ids'])
        assert (folds.reindex(va.SEQN)==f).all()
        replay=t.predict(pack,va);err=float(np.max(abs(replay-saved.prob.reindex(va.SEQN).to_numpy())))
        assert err<1e-6;maximum=max(maximum,err);checks.append({'key':key,'model':f'fold{f}','max_difference':err})
    pack=torch.load(folder/'final.pt',weights_only=False)
    assert set(pack['train_ids'])==set(h.SEQN) and not set(pack['train_ids'])&set(g.SEQN)
    saved=pd.read_csv(folder/'external_predictions.csv').set_index('SEQN')
    replay=t.predict(pack,g);err=float(np.max(abs(replay-saved.prob.reindex(g.SEQN).to_numpy())))
    assert err<1e-6;maximum=max(maximum,err);checks.append({'key':key,'model':'final','max_difference':err})
    lossrows.append({'key':key,'initial_training_loss':pack['losses'][0],'final_training_loss':pack['losses'][-1],'config':json.dumps(pack['config']),'parameters':pack['parameter_count']})
pd.DataFrame(checks).to_csv(o/'prediction_replay_checks.csv',index=False)
pd.DataFrame(lossrows).to_csv(o/'training_diagnostics.csv',index=False)
(o/'verification.json').write_text(json.dumps({'completed':True,'saved_model_replay_checks':len(checks),'maximum_prediction_difference':maximum,'no_outer_PSU_overlap':True,'no_G_subject_in_final_training':True,'same_outer_folds_as_original':True,'main_study_data_results_preserved':True,'conditional_intervals_not_full_refit_intervals':True},indent=2),encoding='utf-8')

def table(df,cols):
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(str(r[c]) for c in cols)+' |\n' for _,r in df.iterrows())
external=combined.query('stage=="External"').copy()
external['面板']=external.panel.map({'Questionnaire':'问卷7项','NonLabBP':'含血压11项','Expanded':'扩展47项'})
external['模型']=external.algorithm.map(labels);external['AUC']=external.auc.map(lambda x:f'{x:.4f}')
external['95%区间']=external.apply(lambda r:f'[{r.auc_low:.4f}, {r.auc_high:.4f}]',axis=1)
external['Brier']=external.brier.map(lambda x:f'{x:.4f}')
pairs=pd.read_csv(o/'paired_comparisons.csv');a=pairs.query('stage=="External" and comparison=="architecture_change"').copy()
a['模型']=a.model;a['对照']=a.reference;a['ΔAUC']=a.delta_auc.map(lambda x:f'{x:+.4f}');a['95%区间']=a.apply(lambda r:f'[{r.auc_low:+.4f}, {r.auc_high:+.4f}]',axis=1)
lab=pairs.query('stage=="External" and comparison=="laboratory_increment"').copy()
lab['模型']=lab.model;lab['ΔAUC']=lab.delta_auc.map(lambda x:f'{x:+.4f}');lab['95%区间']=lab.apply(lambda r:f'[{r.auc_low:+.4f}, {r.auc_high:+.4f}]',axis=1)
age_rows=[]
for panel in t.PANELS:
 for alg in labels:
    key=panel+'_'+alg
    file=(o/'models'/key/'external_predictions.csv') if alg in t.VARIANTS else (t.BASE/'results/models'/key/'external_predictions.csv')
    pred=pd.read_csv(file).set_index('SEQN').prob.reindex(g.SEQN).to_numpy()
    for level,mask in [('30–49',(g.RIDAGEYR>=30)&(g.RIDAGEYR<50)),('50–64',(g.RIDAGEYR>=50)&(g.RIDAGEYR<65)),('65+',(g.RIDAGEYR>=65))]:
        age_rows.append({'panel':panel,'algorithm':alg,'age':level,'n':int(mask.sum()),**t.metric(g.loc[mask],pred[mask.to_numpy()])})
ages=pd.DataFrame(age_rows);ages.to_csv(o/'age_subgroups_exploratory.csv',index=False)
age_table=ages.query('panel=="Questionnaire"').pivot(index='algorithm',columns='age',values='auc').reindex(list(labels)).reset_index()
age_table['模型']=age_table.algorithm.map(labels)
for c in ['30–49','50–64','65+']:age_table[c]=age_table[c].map(lambda x:f'{x:.3f}')
text='''# Transformer 首轮实验报告

日期：2026-10-11。此为原研究的探索性扩展，不替换原主结果。

## 做了什么

采用作者推荐的 rtdl_revisiting_models.FTTransformer 实现，设置小型网络（2 层、4 头）。对比标准数值线性编码、8 段分段线性数值编码、分段编码加数值缺失嵌入三个版本。类别变量均保留缺失/未知类别。

三个版本分别使用问卷7项、含血压11项、扩展47项面板，合计9个组合。每个外层训练集使用与主研究相同种子的3折PSU分组内层验证，在32维/学习率0.001和64维/学习率0.0005两种配置中选择；均固定40轮、batch 512、AdamW权重衰减0.001。调查权重进入二元交叉熵损失。未加类别平衡，不用外层或G数据早停。

与主研究相同5个外层折；所有G预测均在9个最终模型完成训练后生成。共324次内层拟合、45个保存的外层模型、9个保存的最终模型。数值预处理采用训练中位数插补、指定偏态变量log1p、训练均值/标准差缩放及标准化值截断[-10,10]。分段端点只来自训练数据。注意：标准FT版本指标准架构的小型配置，不是原论文的大型默认配置；与Logistic等的对比包含各自预处理流程的差异。

## 跨周期结果

'''+table(external,['面板','模型','AUC','95%区间','Brier'])+'''
## 自建改动相对标准FT

'''+table(a,['模型','对照','ΔAUC','95%区间'])+'''
## 同一Transformer版本的化验增量（47项减11项）

'''+table(lab,['模型','ΔAUC','95%区间'])+'''
## 年龄亚组补充检查（问卷7项，G周期AUC点估计）

'''+table(age_table,['模型','30–49','50–64','65+'])+'''
此为针对原研究老年人群表现问题的补充探索，仅列点估计，未计算亚组差值区间，不作显著性或稳定优势判断。全部三个面板见 age_subgroups_exploratory.csv。

## 如何理解

这是有限配置、单随机种子的首轮实验，不证明某种架构普遍优劣。95%区间为1000次条件抽样设计Bootstrap，保留分层/PSU结构，不包括重新训练的不确定性。多项比较未校正。查看G结果后，不应继续根据G选择配置并把它当全新验证；本实验开始前已有旧研究的G结果，所以新增分析明确属于探索。

训练曲线只保存训练损失，不等于验证学习曲线；40轮是否足够以及多随机种子稳定性仍需训练数据内研究。本次没有扩大参数搜索至原论文预算，没有验证新模型可用于临床。缺失标记可能学到检查流程相关信号，不作生物学机制解释。

## 复现与证据

运行 src/transformer_trial.py，然后 src/finalize_transformer_trial.py。模型权重、预处理参数、训练/验证受试者ID、候选得分和训练损失均保存。prediction_replay_checks.csv 验证54个模型的保存预测；all_model_comparison.csv含原模型对照；analysis_plan.json保存本轮方案；run_metadata.json记录软件版本与实际运行设备。旧项目文件没有被覆盖。

架构来源：https://github.com/yandex-research/rtdl-revisiting-models
数值编码依据：https://arxiv.org/abs/2203.05556
'''
(o/'实验报告.md').write_text(text,encoding='utf-8')
(o/'README_先读.md').write_text('''# Transformer 实验入口

- 实验报告.md：设计、结果和局限。
- comparison_external.png：跨周期结果图。
- all_model_comparison.csv：原模型与9个Transformer组合的统一比较。
- paired_comparisons.csv：算法与架构改动、化验增量的配对比较。
- models/：45个外层模型、9个最终模型及预测。
- prediction_replay_checks.csv、verification.json：保存预测回放与训练/验证隔离核查。
- src/：可复现代码。

运行方式：先运行 src/transformer_trial.py，再运行 src/finalize_transformer_trial.py。
GPU并行时可分别使用 --panel Questionnaire --train-only、--panel NonLabBP --train-only、--panel Expanded --train-only；三者全部完成后运行 --evaluate-only。已完成模型可从检查点继续。

本次是有限预算、单随机种子的探索性试验。标准FT使用原作者实现的小型配置；数值编码及缺失嵌入版本属于对标准架构的修改，不宣称新算法或已证明改进。
''',encoding='utf-8')
meta=json.loads((o/'run_metadata.json').read_text())
(o/'requirements_versions.txt').write_text('\n'.join(f'{k}=={v}' for k,v in meta['versions'].items())+'\n',encoding='utf-8')
hashes={str(f.relative_to(o)):hashlib.sha256(f.read_bytes()).hexdigest() for f in o.rglob('*') if f.is_file() and f.name!='manifest.json'}
(o/'manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified',len(checks),'saved model predictions; maximum difference',maximum)
print(external[['面板','模型','AUC','Brier']].to_string(index=False))
