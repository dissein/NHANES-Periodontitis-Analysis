# 基于 XGBoost 的牙周炎风险预测模型分析

本项目采用机器学习方法（XGBoost），利用美国国家健康和营养检查调查（NHANES）数据集，对牙周炎状态进行建模与风险预测。该管道采用了统计学稳健的方法，包括复杂的抽样权重处理、超参数自动化调优以及基于 SHAP 的模型解释。

## 主要功能
- **自动化预处理**：自动处理缺失值、类别变量编码并应用调查权重。
- **超参数优化**：使用 Optuna 进行贝叶斯优化，寻找最佳模型参数。
- **统计稳健性**：在训练和调优过程中引用 NHANES 调查权重。
- **模型解释性**：集成 SHAP (SHapley Additive exPlanations) 分析全局特征重要性和局部特征依赖。
- **临床阈值提取**：通过样条插值分析连续变量的 SHAP 零点，提取临床风险阈值（Cutoffs）。

## 项目结构
- `code/`: 核心源代码模块（数据加载、训练、调优、SHAP 分析等）。
- `LICENSE`: MIT 开源许可协议。
- `.gitignore`: Git 忽略规则，确保本地大数据集和中间图片不被上传。

## 快速开始
1. **准备数据**: 将 `data.csv` 放置在项目根目录下。
2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```
3. **运行分析**:
   ```bash
   python code/main.py
   ```

## 环境要求
- Python 3.8+
- XGBoost, Optuna, SHAP, Pandas, NumPy, Matplotlib, Scipy

## 作者
- **dissein** <dissein@outlook.com>

## 许可证
本项目采用 [MIT License](LICENSE) 开源协议。
