# GATS 论文复现与深度理解笔记

## 项目目标
- 完整复现 NeurIPS 2022 论文《What Makes Graph Neural Networks Miscalibrated?》
- 深入理解 GCN、GATS 的代码实现与数学原理
- 建立从“能跑通”到“能设计”的研究能力

## 核心结论
- GATS 在保持准确率的同时，将 ECE 从 13.83% 降至 3.43%
- 校准的本质是让模型的“置信度”匹配“真实准确率”
- 注意力机制是 GATS 成功的关键：它让模型自动判断该信任哪些邻居

## 项目结构
GATS-Implementation-Notes/
│
├── README.md                           # 仓库首页：项目概述、学习目标、核心结论
│
├── 01-Code-Anatomy/                    # 代码解剖
│   ├── 01-data-flow.md                 # 数据如何从原始文件流动到模型
│   ├── 02-model-architecture.md        # GCN/GAT 模型结构详解
│   ├── 03-gats-core.md                 # GATS 核心算法逐行解读
│   └── 04-evaluation-metrics.md        # ECE、NLL、Brier 的计算方式
│
├── 02-Math-Notes/                      # 数学原理
│   ├── 01-gcn-math.md                  # GCN 公式推导与直觉
│   ├── 02-gats-math.md                 # GATS 公式逐项拆解
│   ├── 03-calibration-theory.md        # 校准的数学定义与 ECE 本质
│   └── 04-attention-mechanism.md       # 注意力机制的数学形式
│
├── 03-Experiments/                     # 实验记录
│   ├── 01-results-summary.md           # 你跑出的结果汇总
│   ├── 02-reliability-diagrams/        # 可靠性图 (代码 + 图片)
│   └── 03-ablation-analysis.md         # 消融实验理解 (如果有)
│
├── 04-Code-Snippets/                   # 可运行的代码片段
│   ├── gcn_minimal.py                  # 最小 GCN 实现
│   ├── gats_minimal.py                 # 最小 GATS 实现
│   └── ece_calculation.py              # ECE 计算
│
├── 05-Insights/                        # 总结与启发
│   ├── 01-key-takeaways.md             # 核心收获
│   └── 02-open-questions.md            # 未解问题与新想法
│
└── references.md                       # 参考文献

## 运行环境
- PyTorch 2.13.0+cu126
- PyTorch Geometric 2.3.0
- Cora 数据集
