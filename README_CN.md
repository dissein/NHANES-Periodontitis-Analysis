# 基于 XGBoost 和 NHANES 数据的牙周炎风险预测分析框架

本项目基于美国国家健康和营养检查调查（NHANES）的大样本数据，构建了一套基于 XGBoost 算法的牙周炎患病风险预测流程。该分析框架在统计学上保证了研究的严谨性，不仅整合了复杂的抽样权重处理，还实现了自动化的超参数寻优以及基于 SHAP 归因理论的模型可解释性分析。

## 主要功能
- **全流程预处理**：支持缺失值自动补全、类别特征编码，并深度适配复杂的抽样权重（Survey Weights）。
- **高效参数寻优**：集成 Optuna 框架进行贝叶斯优化，确保在海量搜索空间中高效定位模型的最优参数组合。
- **统计学严谨性**：在模型调优与训练的全过程中全程引入调查权重，确保分析结果对总体人群具有代表性。
- **深度模型解释**：利用 SHAP 理论量化各特征对预测结果的影响力，提供清晰的全局重要性排行与局部特征依赖分析。
- **临床阈值识别**：利用单变量样条插值（Univariate Spline）技术精确定位 SHAP 贡献零点，辅助发现具有临床参考价值的特征阈值（Cutoffs）。

## 项目结构
- `code/`: 核心源代码（包含数据加载、模型训练、参数寻优及 SHAP 分析等模块）。
- `LICENSE`: MIT 开源许可协议。
- `.gitignore`: 预设的 Git 忽略规则，确保本地大型数据集和缓存文件不被追踪。

## 快速开始
1. **准备数据**: 将 `data.csv` 放置在项目根目录下。
2. **环境配置**:
   ```bash
   pip install -r requirements.txt
   ```
3. **启动分析**:
   ```bash
   python code/main.py
   ```

## 环境要求
- Python 3.8+
- 核心库：XGBoost, Optuna, SHAP, Pandas, NumPy, Matplotlib, Scipy

## 作者
- **dissein** <dissein@outlook.com>

## 许可证
本项目采用 [MIT License](LICENSE) 开源协议。
