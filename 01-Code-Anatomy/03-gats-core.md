# GATS 核心代码逐行解读

## 核心思想
GATS 为每个节点学习一个专属的温度参数 T_i，然后用 T_i 缩放该节点的 logits。

## 代码逐行解读

### 1. 初始化阶段
```python
class GATS(nn.Module):
    def __init__(self, model, edge_index, num_nodes, train_mask, num_classes, dist_to_train, args):
        super().__init__()
        self.model = model           # 骨干 GNN (GCN/GAT)
        self.edge_index = edge_index # 图的边列表
        self.num_nodes = num_nodes   # 节点总数
        self.train_mask = train_mask # 训练节点掩码
        self.num_classes = num_classes
        self.dist_to_train = dist_to_train
