# GNN 校准复现

复现两篇论文的核心结果：
- GCL (MM 2022)
- DC(GNN) (AAAI 2024)

## 运行结果

| 方法 | 准确率 | ECE |
|------|--------|-----|
| 未校准 | 81.5% | 0.135 |
| GCL | 81.9% | 0.039 |

## 数学公式

GCL 损失函数：
$$ \mathcal{L}_{GCL} = -\sum_{y}(1+\gamma \hat{p}_y)p_y\log \hat{p}_y $$

## 代码

```python
# 你的核心代码可以贴在这里
print("Hello GNN")# gnn-calibration
