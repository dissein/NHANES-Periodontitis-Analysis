from pathlib import Path
import sys,json,hashlib,base64,html,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'explainability_20261011'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import explainability_extension as e
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'figure.facecolor':'#faf8f2','axes.facecolor':'#faf8f2','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#faf8f2','pdf.fonttype':42})
COL={'Logistic':'#c36b50','EBM':'#327973','XGBoost':'#283f56'}
CATLABEL={'SmokingStatus':['从不吸烟','既往吸烟','当前吸烟'],'RIAGENDR':['男性','女性'],'RIDRETH3':['墨西哥裔','其他西班牙裔','非西班牙裔白人','非西班牙裔黑人','非西班牙裔亚裔','其他／多族裔'],'DMDEDUC2':['低于9年级','9–11年级','高中／同等','部分大学／副学士','大学毕业及以上'],'OHQ030':['≤6个月','6–12个月','1–2年','2–3年','3–5年','>5年','从未']}

def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def table(df):
    text='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    for row in df.itertuples(index=False,name=None):text+='| '+' | '.join(str(v) for v in row)+' |\n'
    return text
def interval(r,prefix):return f"{r[prefix]:.4f} [{r[prefix+'_low']:.4f}, {r[prefix+'_high']:.4f}]"
def name_variant(key):
    v=key.rsplit('_',1)[0]
    if v.startswith('Q_without_'):
        c=v[len('Q_without_'):];return '教育＋收入' if c=='SES' else e.LABELS[c]
    return {'BP_without_body_size':'BMI＋腰围','Expanded_without_smoking':'自报吸烟','Expanded_without_cotinine':'可替宁','Expanded_without_both_smoking':'吸烟＋可替宁','Expanded_without_CBC':'全部血常规','Expanded_without_glycemic_lipids':'HbA1c＋总胆固醇＋甘油三酯','Questionnaire':'BMI＋腰围＋血压','NonLab':'血压','NonLabBP':'全部36项化验'}.get(v,v)

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    imp=pd.read_csv(OUT/'importance.csv');pairs=pd.read_csv(OUT/'deletion_comparisons.csv');perf=pd.read_csv(OUT/'performance.csv');shapes=pd.read_csv(OUT/'ebm_shapes.csv');G=e.data('G');H=e.data('H')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    order=imp.query("panel=='Questionnaire' and algorithm=='XGBoost' and stage=='G_external' and method=='interventional_SHAP'").sort_values('weighted_mean_abs_logodds').variable.tolist()
    for ax,alg in zip(axes,['XGBoost','EBM']):
        r=imp.query("panel=='Questionnaire' and algorithm==@alg and stage=='G_external' and method!='path_dependent_SHAP'").set_index('variable').loc[order]
        x=r.weighted_mean_abs_logodds.to_numpy();ax.barh([e.LABELS[c] for c in order],x,color=COL[alg],height=.65)
        ax.errorbar(x,np.arange(len(order)),xerr=np.vstack([x-r.importance_low_conditional,r.importance_high_conditional-x]),fmt='none',ecolor='#555555',capsize=3)
        ax.set_title('XGBoost：汇总编码后的 SHAP' if alg=='XGBoost' else 'EBM：原生加性项贡献');ax.set_xlabel('调查加权平均绝对贡献（log-odds）');ax.grid(axis='x',alpha=.15)
    fig.suptitle('七项问卷信息：年龄领先，吸烟与看牙时间贡献突出',fontsize=15);fig.text(.5,.01,'误差线为固定模型下的调查设计重抽样区间；两种方法的数值不直接作算法优劣比较。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.94]);save(fig,'01_questionnaire_importance')

    fig,axes=plt.subplots(2,4,figsize=(15,8));variables=e.s.Q
    for ax,c in zip(axes.flat,variables):
        r=shapes.query("panel=='Questionnaire' and variable==@c");final=r.query("fit=='final_H'").sort_values('value');grid=final.value.to_numpy();z=final.delta_logodds_from_reference.to_numpy();folds=r.query("fit!='final_H'").pivot(index='value',columns='fit',values='delta_logodds_from_reference').reindex(grid)
        if c in e.s.u.CAT:
            ax.bar(np.arange(len(grid)),z,color=COL['EBM'],width=.65)
            ax.vlines(np.arange(len(grid)),folds.min(axis=1),folds.max(axis=1),color='#222222',lw=1)
            ax.set_xticks(np.arange(len(grid)),CATLABEL[c],rotation=38,ha='right',fontsize=8)
        else:
            ax.fill_between(grid,folds.min(axis=1),folds.max(axis=1),color=COL['EBM'],alpha=.15);ax.plot(grid,z,color=COL['EBM'],lw=2)
            rug=ax.twinx();rug.hist(H[c].dropna(),bins=15,weights=H.loc[H[c].notna(),'WTMEC2YR']/H.WTMEC2YR.sum(),color='#c2bdb0',alpha=.2);rug.set_ylim(0,max(rug.get_ylim()[1]*4,.1));rug.set_yticks([]);rug.set_zorder(-1)
        ax.axhline(0,color='#777777',lw=.6);ax.set_title(e.LABELS[c]);ax.set_ylabel('相对参考值的贡献（log-odds）',fontsize=8)
        ax.text(.02,.98,'参考：'+str(round(final.reference_value.iloc[0],2)),transform=ax.transAxes,va='top',fontsize=8,color='#666666')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.85,'绿色：全部 H 数据训练的模型\n细线／阴影：五个外层模型的范围\n灰色：H 的加权数据分布\n\n范围表示训练波动，不是置信区间。\n分类变量比较完整编码组合。\n年龄80岁包含80岁及以上。\n曲线不是因果效应或临床阈值。',va='top',linespacing=1.8)
    fig.suptitle('EBM：七项问卷变量如何影响模型判断',fontsize=16);fig.tight_layout(rect=[0,0,1,.95]);save(fig,'02_ebm_questionnaire_shapes')

    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True,sharex=True);variables=e.s.Q+['SES']
    for ax,alg in zip(axes,e.s.SPECS):
        r=pairs.query("cycle=='G' and comparison=='deletion' and reduced_model.str.startswith('Q_without_') and reduced_model.str.endswith(@alg)",engine='python').set_index('reduced_model').loc[['Q_without_'+c+'_'+alg for c in variables]]
        xx=r.auc_loss.to_numpy();ax.errorbar(xx,np.arange(len(variables)),xerr=np.vstack([xx-r.auc_loss_low,r.auc_loss_high-xx]),fmt='o',color=COL[alg],capsize=3)
        ax.axvline(0,color='#999999',ls='--');ax.set_yticks(np.arange(len(variables)),[e.LABELS.get(c,'教育＋收入') for c in variables]);ax.invert_yaxis();ax.set_xlabel('删除后 AUC 损失（完整 − 删除）');ax.set_title(alg);ax.grid(axis='x',alpha=.15)
    fig.suptitle('七项问卷逐项删除并重训：跨周期验证',fontsize=16);fig.text(.5,.01,'正值表示删除后区分能力下降；1000次配对调查设计重抽样，区间条件于已训练模型，未校正多重比较。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.94]);save(fig,'03_questionnaire_deletion')

    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True,sharex=True)
    variants=['Expanded_without_smoking','Expanded_without_cotinine','Expanded_without_both_smoking','Expanded_without_CBC','Expanded_without_glycemic_lipids','NonLabBP']
    for ax,alg in zip(axes,e.s.SPECS):
        r=pairs.query("cycle=='G' and comparison=='deletion'").set_index('reduced_model').loc[[v+'_'+alg for v in variants]];xx=r.auc_loss.to_numpy()
        ax.errorbar(xx,np.arange(len(variants)),xerr=np.vstack([xx-r.auc_loss_low,r.auc_loss_high-xx]),fmt='o',color=COL[alg],capsize=3);ax.axvline(0,color='#999999',ls='--');ax.set_yticks(np.arange(len(variants)),[name_variant(v+'_'+alg) for v in variants]);ax.invert_yaxis();ax.set_xlabel('删除后 AUC 损失');ax.set_title(alg);ax.grid(axis='x',alpha=.15)
    fig.suptitle('扩展模型：有贡献的化验，是否提供不可替代的信息？',fontsize=16);fig.text(.5,.01,'每项均与同算法的47变量完整模型比较；全部化验删除复用原11变量面板。区间为条件配对区间。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.94]);save(fig,'04_expanded_deletion')

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,c in zip(axes,['LBXCOT','LBXWBCSI','LBXGH']):
        r=shapes.query("panel=='Expanded' and variable==@c");f=r.query("fit=='final_H'").sort_values('value');z=r.query("fit!='final_H'").pivot(index='value',columns='fit',values='delta_logodds_from_reference').reindex(f.value)
        ax.fill_between(f.value,z.min(axis=1),z.max(axis=1),alpha=.15,color=COL['EBM']);ax.plot(f.value,f.delta_logodds_from_reference,color=COL['EBM'],lw=2);ax.axhline(0,color='#777777',lw=.6);ax.set_title(e.LABELS[c]);ax.set_xlabel({'LBXCOT':'血清浓度（ng/mL）','LBXWBCSI':'计数（1000 cells/μL）','LBXGH':'HbA1c（%）'}[c]);ax.set_ylabel('相对训练参考值的贡献（log-odds）')
        if c=='LBXCOT':ax.set_xscale('symlog',linthresh=1)
    fig.suptitle('实验室指标的 EBM 曲线：关系形状与训练波动',fontsize=14);fig.text(.5,.01,'曲线覆盖H的1%–99%分位数；阴影是五个外层模型的范围，不是因果效应或95%置信区间。',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,.93]);save(fig,'05_ebm_laboratory_shapes')

    # SHAP dependence displays show original variables and retain missingness.
    vals=pd.read_csv(OUT/'explanations/Questionnaire_XGBoost_G_contributions.csv');fig,axes=plt.subplots(2,2,figsize=(12,8));rng=np.random.default_rng(20261011)
    for ax,c in zip(axes.flat,['RIDAGEYR','INDFMPIR','SmokingStatus','OHQ030']):
        valid=G[c].notna();x=G.loc[valid,c].to_numpy();z=vals.loc[valid,c].to_numpy()
        if c in e.s.u.CAT:x=x+rng.normal(0,.055,len(x))
        ax.scatter(x,z,s=5,alpha=.17,color=COL['XGBoost'],rasterized=True);ax.axhline(0,color='#777777',lw=.6);ax.set_title(e.LABELS[c]);ax.set_ylabel('汇总编码后的SHAP（log-odds）')
        if c in e.s.u.CAT:ax.set_xticks(e.s.u.CAT[c],CATLABEL[c],rotation=25,ha='right')
    fig.suptitle('XGBoost：同一变量对不同人的判断贡献',fontsize=15);fig.text(.5,.01,'每点一人（未按权重放大）；正值将模型判断推向中重度，负值相反。纵向差异体现个体背景，不能当作病因。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,.95]);save(fig,'06_shap_dependence')

    qrows=[]
    for c in e.s.Q+['SES']:
        row={'删除的信息':e.LABELS.get(c,'教育＋收入')}
        for alg in e.s.SPECS:
            r=pairs.loc[(pairs.cycle=='G')&(pairs.reduced_model=='Q_without_'+c+'_'+alg)].iloc[0]
            row[alg+' AUC损失 [95%区间]']=interval(r,'auc_loss')
        qrows.append(row)
    qtable=pd.DataFrame(qrows)
    labrows=[]
    for v in variants:
        for alg in e.s.SPECS:
            r=pairs.loc[(pairs.cycle=='G')&(pairs.reduced_model==v+'_'+alg)&(pairs.comparison=='deletion')].iloc[0]
            labrows.append({'删除的信息':name_variant(v+'_'+alg),'模型':alg,'AUC损失 [95%区间]':interval(r,'auc_loss'),'Brier增加 [95%区间]':interval(r,'brier_increase')})
    labtable=pd.DataFrame(labrows)
    qtable.to_csv(OUT/'table_questionnaire_deletion.csv',index=False,encoding='utf-8-sig');labtable.to_csv(OUT/'table_expanded_deletion.csv',index=False,encoding='utf-8-sig')
    missing=[]
    for cycle,d in [('H',H),('G',G)]:
        for c in e.s.FULL:missing.append({'cycle':cycle,'variable':c,'label':e.LABELS[c],'missing_n':int(d[c].isna().sum()),'weighted_missing_fraction':float(d.loc[d[c].isna(),'WTMEC2YR'].sum()/d.WTMEC2YR.sum())})
    pd.DataFrame(missing).to_csv(OUT/'missingness.csv',index=False)
    top=[]
    for panel in ['Questionnaire','Expanded']:
        for alg in ['XGBoost','EBM']:
            rr=imp.query("panel==@panel and algorithm==@alg and stage=='G_external' and method!='path_dependent_SHAP'").sort_values('rank').head(8)
            top.append({'面板':panel,'方法':alg,'贡献排序（前列）':'、'.join(e.LABELS[c] for c in rr.variable)})
    def get(key):return pairs.loc[(pairs.cycle=='G')&(pairs.reduced_model==key)&(pairs.comparison=='deletion')].iloc[0]
    # Specific conclusions are written from the actual results, not a promised
    # mechanism. All algorithms shown, including contradictory increments.
    findings=[]
    for title,key in [('删除年龄','Q_without_RIDAGEYR_Logistic'),('删除吸烟','Q_without_SmokingStatus_Logistic'),('删除看牙时间','Q_without_OHQ030_Logistic'),('删除教育＋收入','Q_without_SES_Logistic')]:
        r=get(key);findings.append(f'{title}：Logistic 外部 AUC 损失 {interval(r,"auc_loss")}，Brier 增加 {interval(r,"brier_increase")}。')
    smoke=[]
    for alg in e.s.SPECS:
        rs=get('Expanded_without_smoking_'+alg);rc=get('Expanded_without_cotinine_'+alg);rb=get('Expanded_without_both_smoking_'+alg)
        smoke.append(f'{alg}：只删自报吸烟损失 {interval(rs,"auc_loss")}；只删可替宁损失 {interval(rc,"auc_loss")}；同时删两者损失 {interval(rb,"auc_loss")}。')
    report=f'''# NHANES 可解释性补充研究：SHAP、EBM 曲线与删除变量重训

完成日期：2026-10-11。此为原跨周期研究的探索性补充，不替代原研究设计。

## 这次分析得到什么

1. **年龄与吸烟是最稳定的问卷信号。**删除年龄后，三个模型的G验证AUC损失分别约0.062–0.070；删除吸烟损失约0.015–0.021，条件配对区间均在0以上。看牙时间的损失约0.006–0.010，但XGBoost区间跨0，不能说三种模型都有明确独立收益。
2. **教育与收入应该同时看。**收入在问卷XGBoost的SHAP排序第4，却在删除并重新调参后使G的AUC提高约0.0087；Logistic和EBM单删收入的区间跨0。然而，同时删教育与收入时，三个模型的AUC都下降，条件区间均在0以上。这支持把社会经济信息作为一组研究，不能仅按贡献排序决定保留哪一项。
3. **自报吸烟与可替宁提供部分可替代的信息。**在47变量模型中，单独删除任一项的损失较小，联合删除损失约0.011–0.018，三个模型的条件区间均在0以上；在另一项缺失时，保留这一项有预测价值。XGBoost单删自报吸烟仍有小幅损失，说明不是完全互换。这个结果更接近“烟草暴露信息的重要性”，不能认定可替宁检测具有稳定的额外采集价值。
4. **有可见的化验曲线，不等于有稳定增量。**血常规整组及三项糖脂指标删除后，没有一致的区分能力损失。EBM删可替宁或糖脂小组时AUC反而小幅提高，属于模型和周期特定的探索结果；不能推出这些指标与牙周炎无关。

所有“区间在0以上／以下”均指固定模型下、未校正多重比较的探索性区间。不能根据已看过的G结果反向挑出一个六变量模型，再把同一个G当作其全新验证；任何精简后的正式模型都需要新的验证。

## 研究问题与设计

主要问题：七项问卷信息中的哪些内容承载了预测信号？实验室指标是否提供了已有信息无法替代的增量？研究当前可观察牙周检查成人的中重度牙周炎识别，不研究未来发病或病因。

开发为2013–2014 H周期（n=3617），验证为2011–2012 G周期（n=3323）。结局、纳入域、权重及PSU分组沿用主研究。14种删除方案 × Logistic、EBM、XGBoost，共42套新流程；每套5个外层模型和1个全部H模型，共252个保存模型。每次外层训练及最终训练均独立进行3折PSU分组内部选择，两组候选参数与原研究相同。包含内部候选训练共1764次拟合。删除年龄时完整移除Logistic年龄样条，未保留隐藏年龄输入。

方案在本次训练前写入analysis_plan.json；G周期先前已被分析，且变量优先级参考了既往结果，因此不是前瞻性预注册或全新盲测。所有新模型只在H训练和调参；训练脚本不读取G。原数据、原模型及原结果文件的SHA256保持一致。

## 解释方法

XGBoost采用TreeSHAP解释原始log-odds，主要设定为interventional。每个模型的背景集从对应H训练样本按调查权重有放回抽128次；背景不含该外层验证样本，也不含G。另对G做tree_path_dependent敏感性分析。分类哑变量和缺失指示项的归因相加回原始变量，因此报告的是“汇总编码后的归因”；这不是重新估计的原变量组联盟Shapley值。编码层的interventional设定可能产生不现实的分类组合，需结合路径设定稳定性及成组删除判断。

EBM保持interactions=0，直接提取加性项eval_terms，并按原始变量汇总。连续曲线覆盖H中1%–99%分位数；分类图比较完整合法编码。图上贡献以相同的H中位数／众数为参考，单位log-odds，不能解释成风险百分比。五个外层模型曲线的范围展示训练波动，不是95%置信区间。年龄80岁为公开数据顶端编码，包含80岁及以上。

H解释来自各外层留出样本，G解释来自全部H训练的最终模型。全局贡献使用调查加权平均绝对贡献。贡献总和加截距必须重建模型判断，逐模型检查加性及概率重建误差。

删除方案重新插补、重新变换、重新内部选择并训练。保留相同样本，不因删除变量改变纳入人数。对H外层预测和G验证预测实施1000次配对Rao-Wu重抽样，从各周期全部正MEC权重设计中抽PSU，比较AUC损失（完整−删除）及Brier增加（删除−完整）。正值表示删除后变差。区间条件于已训练模型，不包含完整训练和背景抽样不确定性，未校正多重比较；不能据此作等效性或临床重要性结论。

## 七项问卷信息的贡献

{table(pd.DataFrame(top))}

![问卷贡献](figures/01_questionnaire_importance.png)

![SHAP依赖图](figures/06_shap_dependence.png)

重要性是模型输出的归因强度，不是删除后的性能损失。EBM和SHAP的绝对数值及排名不要求一致，也不能跨算法直接把更大的贡献数值当作更好的模型。

## 删除问卷变量后的验证结果

{table(qtable)}

{chr(10).join('- '+x for x in findings)}

![问卷删除重训](figures/03_questionnaire_deletion.png)

## EBM 关系形状

![问卷EBM曲线](figures/02_ebm_questionnaire_shapes.png)

曲线说明模型学到的关系，不能将视觉转折点命名为临床阈值。看牙时间同时可能反映医疗可及性、预防行为和有症状后的就诊；种族／族裔可能承载社会结构信息，不能解释为天生生物学机制。教育与收入分别及联合删除用于检验信息重叠，而非估计社会因素的因果效应。

## 自报吸烟与可替宁：互补还是冗余

{chr(10).join('- '+x for x in smoke)}

额外对“无可替宁时删除吸烟”和“无自报吸烟时删除可替宁”做直接配对比较，结果见deletion_comparisons.csv的conditional_deletion。联合删除不能用两个单独损失简单相加，也不能把非相加性称为已证实的生物学交互。

可替宁的加权缺失比例为H约2.85%、G约4.17%；自报吸烟约0.03%、0.04%。归因包含训练插补及缺失指示项的贡献，因此不能把全部可替宁归因都解释成浓度的生物学作用。全部变量的缺失信息见missingness.csv。

## 实验室信息的删除重训

{table(labtable)}

![化验删除](figures/04_expanded_deletion.png)

![化验EBM曲线](figures/05_ebm_laboratory_shapes.png)

血常规作为10项完整检测面板删除。代谢小组仅指HbA1c、总胆固醇、甘油三酯三项，不能概括成删除全部代谢信息。可替宁、白细胞及HbA1c展示曲线；白细胞没有单独删除重训，不能把整组删除结果归到白细胞一项。

“删除全部36项化验”复用原11变量NonLabBP模型；“删除血压”复用原9变量NonLab模型；“删除全部体测和血压”复用原7变量模型。它们接受同样的嵌套训练协议，但有不同维度、编码顺序与随机列采样；差异应理解为完整建模流程的变化。

## 稳定性与证据边界

{table(pd.read_csv(OUT/'importance_stability.csv').query("comparison in ['H_OOF_vs_G','SHAP_interventional_vs_path']").round(3))}

秩相关描述排序稳定性，未进行显著性检验；不能证明每个人的方向均相同。固定模型条件区间不包含模型训练的不确定性。两周期都来自美国NHANES相邻时期，不能替代中国人群验证。

本次分析将“模型使用的信息”与“删除后不可替代的信息”分开：某项指标可以有较高归因贡献，但因其他变量提供相似信息，删除后性能变化有限。不能从贡献高推断病因，也不能从增量区间跨0推断指标无用。

## 复现与文件

入口：README_先读.md；完整结果：performance.csv、deletion_comparisons.csv、importance.csv、importance_stability.csv、ebm_shapes.csv；受试者级归因：explanations/；模型：models/；检查：verification.json、explanation_checks.json、prediction_replay_checks.csv、preservation_after.json；源码：src/。图提供PNG及PDF。

复现顺序：explainability_extension.py plan；train --algorithm Logistic／XGBoost／EBM；explain；evaluate；deliver_explainability.py。训练断点从独立文件恢复，完成结果不重复训练。

## 方法来源

- [SHAP TreeExplainer 官方文档](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)：输出尺度、背景及特征依赖设定。
- [InterpretML EBM 官方API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)：加性项、eval_terms与模型结构。
- 调查设计与结局来源沿用主研究及保存的CDC代码簿。
'''
    (OUT/'完整分析报告.md').write_text(report,encoding='utf-8')
    checks=json.loads((OUT/'verification.json').read_text());checks['explanation_model_checks']=len(json.loads((OUT/'explanation_checks.json').read_text()));checks['figure_count']=6;e.writejson(OUT/'verification.json',checks)
    brief='已完成42套删除重训流程、252个保存模型，并补齐SHAP和EBM曲线。七项问卷中的年龄、吸烟、看牙时间需要结合删除结果解释；实验室指标的归因强度不等同于其独立预测增量。具体效应及区间见本报告，主研究结论仍以原跨周期模型比较为依据。'
    (OUT/'README_先读.md').write_text('# 可解释性补充研究\n\n'+brief+'\n\n- [完整分析报告](完整分析报告.md)\n- [结果网页](可解释性结果.html)\n- 原研究原始数据、模型和结果保留不变。\n- 全部分析是探索性的预测解释，不是病因分析。\n',encoding='utf-8')
    # Portable HTML uses only aggregate tables and embedded exported figures.
    sections=[]
    for title,body,img in [('问卷信息：模型在使用什么',pd.DataFrame(top).to_html(index=False,escape=True),'01_questionnaire_importance'),('关系形状：EBM 如何判断','<p>相对训练参考值的加性贡献。曲线范围表示训练波动，不是置信区间或临床阈值。</p>','02_ebm_questionnaire_shapes'),('删除后重训：信息能否被替代',qtable.to_html(index=False,escape=True),'03_questionnaire_deletion'),('化验是否补充已有信息',labtable.to_html(index=False,escape=True),'04_expanded_deletion'),('化验曲线','<p>白细胞仅进行了血常规成组删除，未单独删除重训。</p>','05_ebm_laboratory_shapes'),('个体判断的 SHAP 贡献','<p>正值推向中重度，负值相反；每点代表一人，图中点大小未按调查权重调整。</p>','06_shap_dependence')]:
        b64=base64.b64encode((OUT/'figures'/f'{img}.png').read_bytes()).decode();sections.append(f'<section><h2>{title}</h2><div class="scroll">{body}</div><img src="data:image/png;base64,{b64}" alt="{title}"></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NHANES · 信息贡献与可替代性</title><style>body{margin:0;background:#faf8f2;color:#243b50;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1160px;margin:auto;padding:48px 24px}header{border-bottom:1px solid #d9d5cb;padding-bottom:32px}small{letter-spacing:2px;color:#327973}h1{font-size:clamp(32px,5vw,60px);line-height:1.2}h2{font-size:28px}section{margin-top:56px}img{width:100%;height:auto;margin-top:24px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:12px;text-align:left;border-bottom:1px solid #d9d5cb;white-space:nowrap}th{color:#327973}.scroll{overflow:auto}a{color:#327973}.note{padding:20px;background:#eeeadd;border-left:4px solid #c36b50}footer{margin-top:48px;font-size:13px;color:#666}</style><main><header><small>NHANES / EXPLAINABILITY / 2026.10.11</small><h1>模型用到了什么，<br>又离不开什么？</h1><p>SHAP ＋ EBM 曲线 ＋ 删除变量重训</p><p>42套新流程 · 252个保存模型 · 相同PSU分组嵌套验证 · 1000次配对调查设计重抽样</p><p><a href="完整分析报告.md">完整报告与方法</a> · <a href="../web/NHANES_研究展示.html">主研究结果</a></p></header><div class="note"><b>阅读方式</b><p>贡献说明模型如何使用信息；删除重训检验其可替代性。以下均为探索性预测解释。区间条件于已训练模型，未校正多重比较，不代表因果效应。</p></div>'''+''.join(sections)+'<footer>开发 H 2013–2014 · 验证 G 2011–2012 · 保留原研究数据和模型。全部图表由本次计算结果生成，页面可离线打开。</footer></main></html>'
    insight='<section><h2>本次分析的三个重点</h2><p><b>年龄与吸烟：</b>三个模型删除后均变差，是最稳定的问卷信号。</p><p><b>吸烟与可替宁：</b>单删损失较小，联合删除损失更大，说明信息有较大重叠。</p><p><b>贡献不等于收益：</b>收入的SHAP贡献较高，但删除重训没有显示稳定增量；教育与收入联合删除则会损失信息。</p></section>'
    page=page.replace('<div class="note">',insight+'<div class="note">',1)
    (OUT/'可解释性结果.html').write_text(page,encoding='utf-8')
    (OUT/'src').mkdir(exist_ok=True)
    for f in ['explainability_extension.py','deliver_explainability.py']:shutil.copy2(Path(__file__).resolve().parent/f,OUT/'src'/f)
    (OUT/'src/requirements_lock.txt').write_text('\n'.join(k+'=='+v for k,v in json.loads((OUT/'runtime_versions.json').read_text()).items())+'\n',encoding='utf-8')
    # Extend the manuscript with an auditable supplement link and real results.
    marker='\n\n## 可解释性补充研究（2026-10-11）\n'
    p=ROOT/'docs/完整研究稿_方法结果讨论.md';text=p.read_text(encoding='utf-8').split(marker)[0]
    paragraph='本次新增SHAP、原生EBM曲线及42套删除重训流程。'+''.join(findings)+'\n\n三个算法删除年龄与吸烟后均出现预测损失。扩展模型单独删除自报吸烟或可替宁的损失较小，联合删除后的AUC损失约0.011–0.018，提示部分信息可替代；不证明完全互换或生物学交互。收入在问卷XGBoost中SHAP贡献较高，删除并重调参后G的AUC却提高约0.0087；教育与收入联合删除则在三个模型中都出现损失，说明排序不能直接决定变量取舍。不能根据G结果挑选精简模型后继续把G当作全新验证。\n\n解释性结果反映预测贡献和信息可替代性，不能用于因果推断。实验室成组删除、自报吸烟与可替宁对照及区间见[完整补充报告](../explainability_20261011/完整分析报告.md)。本次设计在既往G结果已知后确定，属于探索性补充；删除重训区间条件于固定模型，未校正多重比较。\n'
    p.write_text(text+marker+paragraph,encoding='utf-8')
    for p in [ROOT/'README_先读这个.md',ROOT.parent/'当前版本入口_20261011.md']:
        if p.exists():
            mark='\n\n### 2026-10-11 可解释性补充\n';prefix=p.read_text(encoding='utf-8').split(mark)[0];target='explainability_20261011/完整分析报告.md' if p.parent==ROOT else '06_跨周期完整研究_20261010/explainability_20261011/完整分析报告.md';p.write_text(prefix+mark+f'已完成SHAP、EBM曲线及删除变量重训：[完整报告]({target})。\n',encoding='utf-8')
    # Durable link in both the existing generated site and its builder.
    link='<aside style="max-width:1160px;margin:32px auto;padding:20px;background:#eeeadd;color:#243b50"><b>新增：变量贡献与信息可替代性</b><p>SHAP、EBM曲线及删除变量重训已完成。<a href="../explainability_20261011/可解释性结果.html">查看可解释性补充结果</a></p></aside>'
    for p in [ROOT/'web/NHANES_研究展示.html',ROOT/'src/build_showcase.py']:
        text=p.read_text(encoding='utf-8')
        if '查看可解释性补充结果' not in text:text=text.replace('<footer',link+'<footer',1);p.write_text(text,encoding='utf-8')
    e.writejson(OUT/'manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print('DELIVERED',OUT,flush=True)

if __name__=='__main__':main()
