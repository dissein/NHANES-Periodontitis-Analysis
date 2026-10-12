# NHANES 牙周炎分析

[English](README.md) · [交互 demo](https://dissein.github.io/NHANES-Periodontitis-Analysis/)

研究化验信息和算法复杂度对中重度牙周炎识别的增量价值。使用 NHANES 2013–2014 开发模型、2011–2012 跨周期验证，共纳入 6,940 名成人。

## 结论

- 七项问卷 Logistic 模型的跨周期加权 AUC 为 0.801。增加化验指标或使用更复杂的算法，未见稳定提升。
- 年龄与吸烟提供了较稳定的信息，自报吸烟与可替宁存在部分信息重叠。
- 65 岁及以上人群的 AUC 约为 0.646；跨周期应用还需校准预测概率。

## 方法

按 CDC/AAP 2012 监测定义确定结局，比较 7、9、11、19、47 项信息面板与 Logistic、EBM、XGBoost。使用调查权重和 PSU 分组嵌套验证，通过配对比较区分同算法下的信息增量与同面板内的算法增量。

化验的主要比较为 47 项相对 11 项，单独拆出血压。结合 SHAP、EBM 曲线和删除变量后重训，分析模型使用的信息及其重叠。

## 文件

| 目录 | 内容 |
|---|---|
| `src/` | 数据构建、训练、验证与检查 |
| `data/` | 变量字典与公开 NHANES 数据整理得到的分析表 |
| `results/` | 模型比较、区间估计、已保存模型与预测 |
| `explainability_20261011/` | SHAP、EBM 曲线与删除变量分析 |
| `figures/` | 研究图 |
| `docs/` | 正文、补充材料与英文摘要 |
| `web/demo_v1/` | 离线交互 demo 与源文件 |

原有单独 XGBoost 流程保留在 Git 历史中。

## 运行

使用 Python 3.12，依赖版本见 `requirements.txt`。

```bash
python -m pip install -r requirements.txt
python src/run_all.py
```

首次运行会下载官方 XPT 文件至 `data/raw/`。已有模型结果会作为缓存使用；完整重训可在新目录中放入 `src/` 和 `data/` 中的变量字典后运行。

可解释性分析：

```bash
python explainability_20261011/src/explainability_extension.py
```

双击 `web/demo_v1/index.html` 即可离线查看模型表现、信息贡献和适用边界。重新生成页面：

```bash
python web/demo_v1/src/build_nhanes_demo_v1.py
```

Windows 下可嵌入已安装的思源黑体；其他环境使用系统字体。

## 数据来源

NHANES 官方公开数据：[2011–2012](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011)、[2013–2014](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013)。模块地址与文件校验值见 `results/raw_sources.json`。

分析对象为当前中重度牙周炎，不是未来发病风险。代码采用 [MIT 许可](LICENSE)。
