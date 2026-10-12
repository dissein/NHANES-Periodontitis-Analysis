from pathlib import Path
import sys,json,hashlib,shutil,datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE.parent if HERE.name=='src' else Path(__file__).resolve().parents[1]

def table(a,cols=None,digits=3):
    a=a[cols].copy() if cols else a.copy()
    for c in a:
        if a[c].dtype.kind=='f':a[c]=a[c].map(lambda x:f'{x:.{digits}f}' if np.isfinite(x) else '')
    return '| '+' | '.join(a.columns)+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in a.to_numpy())

def savefig(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def main():
    r=OUT/'results';docs=OUT/'docs';figs=OUT/'figures'
    performance=pd.read_csv(r/'performance.csv');raw=performance.query('variant=="raw"').copy();pairs=pd.read_csv(r/'paired_comparisons.csv')
    refit=pd.read_csv(r/'refit_uncertainty.csv');ops=pd.read_csv(r/'operating_points.csv');prev=pd.read_csv(r/'prevalence.csv');reverse=pd.read_csv(r/'reverse_transport.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    colors={'Logistic':'#1D6996','EBM':'#38A6A5','XGBoost':'#D78136'};order=['Questionnaire','NonLab','NonLabBP','CBC','Expanded']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,stage in zip(axes,['Internal','External']):
        for j,m in enumerate(colors):
            a=raw.query('stage==@stage & algorithm==@m').set_index('panel').loc[order]
            x=np.arange(5)+(j-1)*.17;ax.errorbar(x,a.auc,yerr=[a.auc-a.auc_low,a.auc_high-a.auc],fmt='o-',capsize=3,color=colors[m],label=m)
        ax.set_xticks(range(5),['7-question','9 non-lab','11 + BP','19 + CBC','47 expanded'],rotation=20);ax.set_title('H nested internal validation' if stage=='Internal' else 'H-trained, G cycle validation');ax.set_ylim(.735,.835);ax.set_ylabel('Survey-weighted AUC');ax.legend(fontsize=8)
    fig.suptitle('Information burden and algorithm complexity');fig.tight_layout();savefig(fig,'01_panel_performance')
    fig,ax=plt.subplots(figsize=(8,4))
    a=pairs[(pairs.stage=='External')&pairs.reference.str.startswith('NonLabBP')&(pairs.comparison=='information_increment')]
    x=np.arange(len(a));ax.errorbar(a.delta_auc,x,xerr=[a.delta_auc-a.auc_delta_low,a.auc_delta_high-a.delta_auc],fmt='o',capsize=4,color='#1D6996');ax.axvline(0,color='black',lw=1);ax.set_yticks(x,a.model);ax.set_xlabel('Expanded minus non-lab + BP AUC');ax.set_title('Laboratory increment: paired conditional design intervals');fig.tight_layout();savefig(fig,'02_laboratory_increment')
    bins=pd.read_csv(r/'calibration_bins.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True)
    for ax,cycle in zip(axes,['H','G']):
        for key,color in [('Questionnaire_Logistic','#1D6996'),('Expanded_Logistic','#38A6A5'),('Expanded_EBM','#D78136')]:
            a=bins[(bins.cycle==cycle)&(bins.key==key)&(bins.n>=15)];ax.plot(a.predicted,a.observed,'o-',label=key,color=color)
        ax.plot([0,1],[0,1],'--',color='grey');ax.set_title('Internal OOF H' if cycle=='H' else 'Independent cycle G');ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed weighted prevalence');ax.legend(fontsize=7)
    fig.tight_layout();savefig(fig,'03_calibration_shift')
    a=ops[(ops.stage=='External')&(ops.threshold_rule=='sensitivity80')&ops.key.isin(['Questionnaire_Logistic','NonLabBP_Logistic','Expanded_Logistic','Expanded_EBM'])];fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(len(a))
    bottom=np.zeros(len(a))
    for c,label,color in [('detected_per1000','Detected cases','#1D6996'),('false_positive_per1000','Flagged non-cases','#DD9A3D'),('missed_per1000','Missed cases','#A34C5E')]:ax.bar(x,a[c],bottom=bottom,label=label,color=color);bottom+=a[c].to_numpy()
    ax.bar(x,1000-bottom,bottom=bottom,label='Unflagged non-cases',color='#CBD2D8')
    ax.set_xticks(x,a.key,rotation=15,fontsize=8);ax.set_ylabel('Per 1,000 in the observable G-cycle domain');ax.set_title('Development-selected 80% sensitivity threshold');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout();savefig(fig,'04_threshold_workload')
    imp=pd.read_csv(r/'external_permutation.csv');fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,key in zip(axes,['NonLab_Logistic','Expanded_EBM']):
        a=imp[imp.key==key].nlargest(10,'auc_decrease').sort_values('auc_decrease');ax.barh(a.variable,a.auc_decrease,color='#1D6996');ax.set_title(key);ax.set_xlabel('AUC drop after held-out permutation')
    fig.tight_layout();savefig(fig,'05_external_contributions')
    sub=pd.read_csv(r/'subgroups.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,typ in zip(axes,['sex','age']):
        a=sub[(sub.cycle=='G')&(sub.key=='Questionnaire_Logistic')&(sub.subgroup==typ)].copy()
        if typ=='sex':a['level']=a.level.map({'1.0':'Male','2.0':'Female'})
        else:a=a.set_index('level').loc[['30-49','50-64','65+']].reset_index()
        x=np.arange(len(a));ax.errorbar(a.auc,x,xerr=[a.auc-a.auc_low,a.auc_high-a.auc],fmt='o',capsize=3);ax.set_yticks(x,a.level);ax.set_xlim(.57,.9);ax.set_title(typ+' strata: 7-variable Logistic');ax.set_xlabel('AUC with conditional design interval')
    fig.tight_layout();savefig(fig,'06_subgroup_discrimination')
    maincols=['key','n_features','auc','auc_low','auc_high','brier','calibration_intercept','calibration_slope']
    mains=raw[raw.stage=='External'];delta=pairs[(pairs.stage=='External')&pairs.reference.str.startswith('NonLabBP')&(pairs.comparison=='information_increment')]
    op=a=ops[(ops.stage=='External')&(ops.threshold_rule=='sensitivity80')&ops.key.isin(['Questionnaire_Logistic','NonLabBP_Logistic','Expanded_Logistic'])]
    a=raw.set_index(['stage','key']);q=a.loc[('External','Questionnaire_Logistic')];n=a.loc[('External','NonLabBP_Logistic')];f=a.loc[('External','Expanded_Logistic')]
    ci=refit[(refit.comparison=='Expanded_minus_NonLabBP')&(refit.metric=='auc')].iloc[0]
    methods='''## 研究方法

### 设计、立意与分析定位
研究题目：**牙周炎现患识别中化验信息与算法复杂度的增量价值：NHANES双周期调查加权研究**。
英文题目：**Incremental value of laboratory information and model complexity for prevalent periodontitis classification: a survey-weighted, cross-cycle NHANES study**。

主问题是在同一批受试者、相同验证规则和相同算法下，增加化验信息是否改善中重度牙周炎的区分能力与概率表现。次问题包括测量负担、算法增量、跨周期校准、亚组表现及检查工作量。分析为探索性重新设计；原H周期曾用于旧分析，因此本研究不是预注册的全新研究。方案在新周期预测性能检查前形成，另留存增加11项含血压面板以分离化验增量的修订记录。不得将观察后选定的论文主线称为前瞻性注册假设。

### 数据、样本与分析域
2013—2014（H）用于主要开发与嵌套内部验证；2011—2012（G）用于独立受试者的较早周期验证。主分析要求年龄≥30、MEC权重>0、OHDPDSTS=1且有可分类邻面测量。H纳入3,617人，G纳入3,323人，总计6,940名不同受试者。无牙、因医疗原因不能探诊、部分检查及无可用测量者均没有被擅自当作健康对照。纳入/排除人群的加权特征另表列出。

G是较早周期，主方向验证评估向既往周期迁移的能力，不证明对未来人群的预测。补充G开发、H验证的反向分析使用相同特征面板和候选算法，但它是同一研究内的探索性方向检查，不能计为第二个完全独立的新验证队列。本研究适用域是有可观察牙周测量的样本；MEC权重不能自动消除牙周检查缺失、医疗排除和无牙带来的选择问题。

### 结局定义
按CDC/AAP 2012监测标准，以28颗非第三磨牙的四个邻面位置构造标签。重度：≥2个不同牙齿存在CAL≥6 mm的邻面，同时至少1个邻面PD≥5 mm；中度：≥2个不同牙齿存在CAL≥4 mm的邻面，或≥2个不同牙齿存在PD≥5 mm的邻面；轻度：≥2个邻面CAL≥3 mm且不同牙齿的≥2个邻面PD≥4 mm，或至少1个邻面PD≥5 mm。按重度、中度、轻度顺序赋值，其余有可用测量者为无牙周炎。不可评估编码99及SAS特殊缺失不计为阳性测量。主结局为中重度对其余类别；不使用2018临床分期术语。

### 变量与采集层级
7项问卷：年龄、性别、种族/族裔、教育、家庭收入贫困比、最近牙科就诊时间、自报吸烟状态。9项非化验面板再加BMI和腰围；11项面板再加首次收缩压和舒张压；19项CBC面板为9项基础加10个血常规变量；47项扩展面板为原46个特征加自报吸烟，包含上述11项和36项化验信息。血压单独分层，故47对11的主比较可以隔离加入化验信息的收益。

吸烟状态由SMQ020与SMQ040构造：未吸满100支为0，吸满100支且目前不吸为1，每日或部分日吸为2；拒答、不知道和不能由跳题逻辑确定者保留缺失。UACR直接使用官方URDACT，单位mg/g。G的血清可替宁源文件为COTNAL_G，H为COT_H；叶酸用当前发布的修正后LBDFOTSI，维生素D用LBXVIDMS的nmol/L值。所有30个源模块和官方代码簿已留存；H本地源数据另与CDC当前下载核对值相同。

天然牙数、测量覆盖、检查状态、受试者编号、调查分层和PSU只用于样本审计/敏感性/设计，绝不作为预测变量。变量缺失比例及跨周期对应关系列在字典中。

### 建模与验证
同一套按SDMVSTRA+SDMVPSU分组的外层5折。每个外层训练样本内使用3折分组验证选择有限候选；同一PSU不跨训练和评估。数值中位数插补及缺失指示、分类变量缺失编码/独热编码、标准化均只在训练样本拟合。不做类别平衡，模型损失使用按均值归一的WTMEC2YR权重。

算法为年龄样条的岭Logistic、无交互的可解释提升机EBM、XGBoost。Logistic年龄采用训练样本拟合的4结点三次B样条，指定偏态化验变量做log1p变换；这样避免把明显的年龄非线性全部归功于复杂算法。Logistic的C候选为0.1/1；EBM的max_leaves为2/3，固定interactions=0；XGBoost候选为depth2/min_child_weight10和depth3/min_child_weight15，固定250树等设置见源码。候选以平均内层AUC选择，不能据G结果重调。有限搜索不是各算法可达到的性能上限。

外部模型在完整H上以3折分组选择参数，随后在H拟合；G仅用冻结模型计算预测。另将内层OOF拟合的Logistic概率校准作为次要版本，主结果仍报告原始概率。80%、90%敏感度目标阈值和Youden阈值全部从训练内层OOF选取，G或外层评估标签不参与阈值选择。

### 指标与不确定性
报告调查加权AUC、AP、Brier、联合拟合的校准截距/斜率、训练内阈值下的敏感度/特异度/PPV/NPV。校准截距与斜率是评价参数，不是用G重新校准后再声称验证成功。另报告每1,000人检出、误报、漏检及每检出1例需要检查的人数；这些是加权可观察样本的数学工作量，未证明真实服务成本或临床效益。

患病比例采用完整MEC抽样设计中的分析域Taylor线性化及logit区间。模型AUC/Brier及配对差采用1,000次Rao–Wu重标度PSU重抽样：每层m个PSU抽取m−1个并赋m/(m−1)倍权重。G部分层含3个PSU，不能全部按2个处理。这里的主区间以既有预测为条件。

为补充开发过程不确定性，针对11/47项Logistic主比较，完成200次独立开发与验证双周期设计重抽样；每次重新拟合插补、年龄样条、内层调参、阈值和最终模型。该区间包含此次有限开发流程与验证抽样的不确定性，不覆盖所有算法/所有亚组，也不消除研究问题探索与候选空间选择的影响。200次为有限Monte Carlo估计。

### 补充分析
1. 同一冻结中重度评分对任意/重度标签的AUC，不能把它当作各结局专属的概率。
2. 性别、年龄、种族/族裔和收入亚组的表现与条件区间，不声称完成公平性验证。
3. 排除最近就诊、种族或自报吸烟的Logistic重新开发分析。
4. 至少20颗天然牙、至少90%邻面配对测量覆盖、非化验/扩展完整病例的共同样本限制分析；病例构成发生变化，AUC变化不等于测量质量导致的因果效应。
5. 单独的完整病例调查加权未惩罚Logistic关联模型：年龄每10岁、PIR每单位、BMI每5 kg/m²，类别参照预先明确；采用全设计PSU sandwich和设计自由度t区间。报告现患优势比而非风险或因果效应，不以探索性多重比较挑选显著机制。
6. G评估集原始变量置换贡献及假设阈值的决策曲线。相关变量的置换不能分离因果贡献；决策曲线没有真实干预、成本和接受度证据。
'''
    result='## 实际结果\n\n### 样本及现患比例\n\n'+table(prev.query('outcome in ["Any","ModSev","Severe"]'))+'\n\n这些比例仅针对本研究的可观察牙周检查域。H/G中重度比例分别为34.28%/41.13%，周期间病例构成不同。\n\n### 独立周期的主性能\n\n'+table(mains,maincols)+'\n\n### 相同算法下的化验增量：47项减11项\n\n'+table(delta,['model','delta_auc','auc_delta_low','auc_delta_high','delta_brier','brier_delta_low','brier_delta_high'])+f'\n\nLogistic的AUC增量为{f.auc-n.auc:.4f}。加入开发过程重训后，配对差95%区间为[{ci.low:.4f}, {ci.high:.4f}]，仍跨越0。数据没有建立化验面板的稳定增量优势；这不是预设界值的等效性或非劣效性试验，也不证明全部化验没有价值。\n\n### 阈值与检查负担\n\n'+table(op,['key','sensitivity','specificity','ppv','flagged_fraction','detected_per1000','false_positive_per1000','missed_per1000','examinations_per_detected_case'])+'\n\n阈值由H的内层预测确定。7项问卷Logistic在G检出约330例/1,000人，同时标记约198名无中重度者并漏掉约82例；需要标记约528人。它是潜在优先检查策略的估算，不能称为已经验证的筛查方案。\n\n### 双向迁移\n\n'+table(reverse,['key','auc','brier','calibration_intercept','calibration_slope'])+'\n\nH→G时各面板校准截距普遍为正，G→H普遍为负，体现概率迁移偏差。保持AUC不意味着概率可直接跨周期使用。不能将观察到的偏差完全归因于患病率变化，其他协变量与选择构成也可能参与。\n\n### 敏感性与解释\n\n'+table(pd.read_csv(r/'feature_ablations_external.csv'),['variant','n_features','auc','brier'])+'\n\n限制天然牙数、测量覆盖和完整病例时，AUC与比例发生变化；完整数值见quality_sensitivity.csv。这里报告的是固定预测在不同分析子集的表现，不是插补方法之间的随机对照，也不能声称全部敏感性结果不变。关联、亚组和置换结果均为补充证据，不能建立病理机制或因果网络。\n'
    age=sub[(sub.cycle=='G')&(sub.key=='Questionnaire_Logistic')&(sub.subgroup=='age')]
    result+='\n\n### 年龄亚组：整体AUC不能代替老年表现\n\n'+table(age,['level','n','n_cases','auc','auc_low','auc_high','sensitivity','specificity','ppv'])+'\n\n7项问卷模型在65岁及以上亚组的AUC明显低于整体结果，不能把整体约0.80推广成对所有年龄层都有效。此为探索性分层表现，不是预设交互作用检验。\n'
    if (r/'age_only_baseline.csv').exists():result+='\n\n### 年龄样条单变量基线\n\n'+table(pd.read_csv(r/'age_only_baseline.csv'),['stage','auc','brier','questionnaire_minus_age_auc','delta_low','delta_high'])+'\n\n年龄样条基线使用相同训练/验证划分重新开发；7项问卷与其比较的区间仍为既有预测条件下的设计区间。该补充分析在初步主结果后增加，不能表述为预注册分析。\n'
    discussion='''## 讨论与结论

本研究的重要结果不是某种算法夺得最高AUC，而是在相同受试者和调查设计下，7项问卷与9/11项非化验信息已达到约0.80的独立周期区分能力。扩展化验面板的收益随算法和周期变化，没有形成稳定的增量证据。简单模型的年龄函数经过适当处理后，复杂算法没有表现出普遍优势。

这支持将研究重点放在信息采集负担、可靠基线、概率校准和外部迁移，而不是以“更多指标、更深模型”作为创新本身。65岁及以上亚组的区分能力明显不足，整体约0.80不能用于宣称老年筛查有效。自报吸烟、社会经济与牙科就诊信息能够承载疾病识别信号，但它们也与既往疾病、医疗可及性及社会结构有关，不能当作纯粹病因。种族/族裔变量可能反映多重结构性差异，绝不能据此推断生物本质或用来限制就诊。

局限包括：两个相邻美国调查周期，不是中国人群；主验证方向向较早周期；横断面现患分类，不是未来发病预测；无牙和不能检查者没有可靠结局；H周期已有探索历史，论文叙事不是前瞻性预注册；输入缺失采用训练内单次插补；算法搜索空间有限；没有实际费用、转诊路径、医生/患者接受度或前瞻性临床效益数据；亚组区间为条件重抽样；仅两套Logistic流程完成开发重训不确定性分析；并未进行预先规定界值的等效性检验。

结论：**在NHANES可观察牙周检查成人中，少量问卷/非化验信息对当前中重度牙周炎具有一定区分能力。加入化验信息和模型复杂度未建立稳定的增量优势，而跨周期概率偏差仍明显。后续转化应优先研究目标人群校准与实际检查策略，而非直接增加指标或部署诊断工具。**

## 伦理、开放性与作者声明

使用公开去标识化NHANES资料，原调查伦理与同意信息见CDC官方文档。此处未代替北京大学或拟投稿单位作出本地伦理豁免认定。未接触私人患者记录。没有已获导师指导、资助、作者贡献或投稿的虚构陈述。AI协助代码实现、审计与文字整理；正式署名研究者需要核查代码、方法和结论，并按目标期刊要求披露。当前作者/通讯作者、指导教师、资助及本地伦理判断留待真实确认。
'''
    refs='''## 方法与相关研究来源

1. [CDC/AAP 2012监测定义，Eke等](https://stacks.cdc.gov/view/cdc/56160)。
2. [NHANES G牙周代码簿](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/OHXPER_G.htm)、[H牙周代码簿](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/OHXPER_H.htm)。完整30个代码簿见official_codebooks。
3. [G吸烟问卷](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SMQ_G.htm)、[G牙科就诊问卷](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/OHQ_G.htm)。
4. [G可替宁](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/COTNAL_G.htm)、[G叶酸修正](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/FOLFMS_G.htm)、[H叶酸修正](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/FOLFMS_H.htm)。
5. Collins GS等. TRIPOD+AI. BMJ. 2024;385:e078378. [原文](https://www.bmj.com/content/385/bmj-2023-078378)。用于报告框架，不代表本项目获得认证。
6. Moons KGM等. PROBAST+AI. BMJ. 2025;388:e082505. [原文](https://www.bmj.com/content/388/bmj-2024-082505)。用于自审偏倚与适用性。
7. [NHANES权重](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)、[复杂设计方差及分析域](https://wwwn.cdc.gov/nchs/nhanes/tutorials/varianceestimation.aspx)。
8. Suh B等. Explainable Deep Learning Approaches for Risk Screening of Periodontitis. J Dent Res. 2025;104(1):45–53. [原研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11667192/)。NHANES可解释模型及其他数据验证已有人研究，不能宣称本研究首创这一技术组合。
9. Zhu ZX等. Development of a Machine Learning Tool for Home-Based Assessment of Periodontitis. [medRxiv预印本，2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952591/)。检索页面标注为预印本，不能当作已同行评审成果。
10. Choi S等. Screening for Periodontitis Using Blood Biomarkers and Demographic Data: A Machine Learning Study. [原研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13188171/)。用于说明已有血液指标研究背景，未与本研究跨论文直接比较数值。

此为针对相关性及方法的文献核查，不是系统综述。正式投稿需完成目标期刊格式和更全面检索。没有以未知文献补充因果结论。
'''
    report='# NHANES跨周期完整研究稿\n\n完成日期：2026-10-11。正式结果以本目录CSV及保存模型为准，旧05版保留用于溯源。\n\n'+methods+'\n'+result+'\n'+discussion+'\n'+refs
    (docs/'完整研究稿_方法结果讨论.md').write_text(report,encoding='utf-8')
    (docs/'文献与立意核查.md').write_text('# 文献与立意核查\n\n'+refs+'\n## 立意选择\n\n主线采用化验和算法的增量价值＋跨周期校准。备选角度为问卷优先检查的负担、采集层级、医疗可及性相关预测信号。没有新增缺少数据支持的分子机制、饮食因果或未来死亡预测，也没有为了显著结果反复寻找暴露组合。\n',encoding='utf-8')
    abstract=f'''# English structured abstract

**Title:** Incremental value of laboratory information and model complexity for prevalent periodontitis classification: a survey-weighted, cross-cycle NHANES study

**Objective:** To examine whether laboratory information and nonlinear algorithms provide incremental discrimination beyond questionnaire and non-laboratory information for current moderate/severe periodontitis.

**Methods:** This exploratory analysis included 3,617 participants from NHANES 2013–2014 for development and 3,323 independent participants from 2011–2012 for earlier-cycle validation. CDC/AAP 2012 surveillance definitions were applied to full-mouth interproximal measurements. Five predefined information panels (7, 9, 11, 19 and 47 variables) were evaluated using age-spline ridge logistic regression, additive explainable boosting and XGBoost. Development used PSU-grouped nested cross-validation; the validation cycle was not used for tuning or threshold selection. Survey-weighted discrimination, Brier scores, calibration and threshold-specific examination workload were evaluated. Conditional paired design intervals used 1,000 resamples; 200 independent two-cycle resamples additionally repeated development for the 11- versus 47-variable logistic comparison.

**Results:** Weighted moderate/severe prevalence was 34.28% in the development observable domain and 41.13% in validation. The 7-variable questionnaire logistic model achieved validation AUC {q.auc:.3f} (conditional 95% interval {q.auc_low:.3f}–{q.auc_high:.3f}), compared with {n.auc:.3f} for the 11-variable non-laboratory/BP model and {f.auc:.3f} for the 47-variable model. The full-procedure paired interval for the logistic AUC increment was {ci.low:.3f} to {ci.high:.3f}. Expanded-panel EBM achieved AUC 0.792 and XGBoost 0.797. Calibration intercepts were positive when transported to the earlier cycle and negative in reverse-direction analyses. At the development-selected 80% sensitivity threshold, the questionnaire model flagged approximately 528 per 1,000, detecting 330 cases, flagging 198 non-cases, and missing 82 cases.

**Conclusions:** Laboratory information and algorithm complexity did not establish consistent incremental performance over parsimonious information in these observable NHANES domains. Cross-cycle discrimination was accompanied by probability miscalibration. These exploratory findings prioritize transportability and information burden; they do not establish equivalence, prospective disease prediction, clinical utility or applicability in China.
'''
    (docs/'English_Abstract.md').write_text(abstract,encoding='utf-8')
    summary=f'''# 导师汇报：这次项目实际完成了什么

我将原来单周期多算法比较，重新整理为“化验信息和复杂算法是否值得增加”的双周期研究。

- 原周期3,617人，新周期3,323人，总计6,940名独立受试者。
- 5个采集层级×3种算法，PSU分组嵌套验证，所有开发决策在H完成。
- 7项问卷Logistic在独立周期AUC {q.auc:.3f}；47项完整Logistic为{f.auc:.3f}。
- 同算法47项相对11项的化验增量很小；200次重新开发的差值区间[{ci.low:.3f}, {ci.high:.3f}]跨0。
- 新周期出现系统性概率偏低；反向迁移也有概率偏差，说明AUC与概率准确性需要分开。
- 65岁及以上亚组AUC约0.646，不能用整体结果宣称老年人群同样有效。
- 已完成原始变量溯源、调查权重、插补与阈值泄露防范、双向周期检查、亚组、缺失/测量覆盖限制、消融、解释及重训不确定性分析。

希望老师指导：这条信息采集负担与校准主线是否值得继续；如何找到不同来源人群进一步验证；怎样预先定义临床检查策略和实际效益。当前项目的模型与结论是探索性证据，不把它包装成已发表论文或可以临床部署的工具。

## 面试时需要能回答

1. 为什么以中重度为主结局？为何不直接用2018分期？
2. 7/9/11/19/47项具体是什么？为什么单独拆出血压？
3. 插补、样条、调参和阈值如何保证只看训练样本？
4. 为什么G有3个PSU的抽样层？区间怎样计算？
5. 为什么AUC约0.80仍不能称为筛查工具？
6. 为什么旧版最好算法和新版结论不同？新增吸烟、基线年龄函数、验证周期与主问题均改变。
7. 为什么“无明确增量”不等于“化验无用”或“两个面板等效”？
8. AI承担实现与辅助审查，自己如何检查定义、代码和数字？
'''
    (docs/'导师汇报摘要.md').write_text(summary,encoding='utf-8')
    checklist='''# 方法与报告自审

| 项目 | 已完成证据 | 仍有的限制 |
| --- | --- | --- |
| 数据来源 | 30个XPT、30个官方代码簿、SHA256与H官方下载值核对 | 限两个美国周期 |
| 入组与结局 | sample_flow、eligibility_audit、12项边界测试 | 不能观测无牙/医疗排除者的同一结局 |
| 数据划分 | PSU分组5×3、保存训练/评估ID | H有历史探索暴露 |
| 特征与处理 | 5个面板、源变量字典、训练内插补/样条 | 单次插补、有限候选 |
| 基线公平性 | Logistic允许年龄非线性，自报吸烟共同纳入 | 不是算法性能上限比较 |
| 增量 | 同受试者/同算法配对AUC与Brier | 非预设等效性设计 |
| 跨周期 | H→G15模型、G→H6模型 | 主验证向较早周期，反向为补充 |
| 概率与阈值 | 校准、内层阈值、PPV与工作量 | 无临床实际成本与前瞻性效益 |
| 不确定性 | 1000次条件设计bootstrap、200次双周期开发重训 | 重训仅两个Logistic面板，Monte Carlo有限 |
| 敏感性 | 3消融、4限制子集、亚组 | 多重探索不作确认性假设检验 |
| 可复现性 | 90模型评估重放、锁定依赖、run_all.py | 环境硬件可影响细微数值 |
| 伦理和作者 | 公开去标识数据，声明AI辅助 | 本地伦理判断、指导教师、作者、资助未虚构 |

此表参照TRIPOD+AI和PROBAST+AI的关注点组织，不是官方逐条认证、低偏倚评级或投稿保证。
'''
    (docs/'严谨性自审与证据索引.md').write_text(checklist,encoding='utf-8')
    readme=r'''# NHANES当前完整研究版

完成于2026-10-11，目录日期沿用开始日期20261010。

优先阅读：

1. `docs/导师汇报摘要.md`：项目立意与真实结论。
2. `docs/完整研究稿_方法结果讨论.md`：方法、结果、讨论、局限和参考来源。
3. `docs/English_Abstract.md`：英文摘要草稿。
4. `docs/严谨性自审与证据索引.md`：完成内容和证据位置。

## 主要证据

- `data/raw`：两周期30个官方XPT，原历史文件另外保留。
- `data/analysis_H.csv`、`analysis_G.csv`：主要分析表，附加牙数与测量覆盖只用于审计。
- `results/performance.csv`：5面板×3算法×内/跨周期×原始/训练内校准；区间仅对应原始概率。
- `results/paired_comparisons.csv`：同算法采集信息增量与同面板算法增量。
- `results/refit_uncertainty.csv`：主Logistic对比的200次双周期重新开发区间；逐次记录另存。
- `results/operating_points.csv`：从训练选择的80/90%与Youden阈值结果。
- `results/reverse_transport.csv`、`subgroups.csv`、`quality_sensitivity.csv`、`feature_ablations_external.csv`：补充迁移、亚组、限制、消融。
- `results/associations_nonlab.csv`：独立探索性调整POR。
- `results/models`：75个外层训练模型、15个H最终模型、对应预测和元数据。
- `results/model_replay_checks.csv`：全部90个评估的预测重放检查。
- `figures`：6张主要研究图，PNG/PDF。

## 复现

Python 3.12；在独立环境安装`src/requirements_lock.txt`。

```powershell
python -m pip install -r .\src\requirements_lock.txt
python .\src\run_all.py
```

代码只需要本目录现有原始XPT，不依赖Codex或在线AI。原数据重建的历史一致性检查可选依赖父目录05版；独立搬运时自动跳过历史检查并记录。在线官方刷新校验可以单独运行`src/verify_official_sources.py`。模型开发与bootstrap设有缓存，重复执行使用已保存结果。要做全新重跑，可复制本目录的`src`和`data/raw`到新的目录，避免复用旧缓存。不要将缓存目录与已修改的方案混用。

## 当前结论

7项问卷的简单模型跨周期AUC约0.801；47项完整模型没有建立稳定的化验增量。跨周期概率仍失准。当前是可复现的完整探索性研究，不是等效性试验、临床验证、未来风险预测或投稿承诺。05版为上一轮审计，当前新增结果以06版为准。
'''
    (OUT/'README_先读这个.md').write_text(readme,encoding='utf-8')
    print('Manuscript, abstract, summary, audit index and six figures created')

if __name__=='__main__':main()
