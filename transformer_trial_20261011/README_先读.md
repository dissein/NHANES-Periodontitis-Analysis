# Transformer 实验入口

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
