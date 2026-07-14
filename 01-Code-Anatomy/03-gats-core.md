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
#文件使用：
#布尔掩码:
         array = np.array([0,1,2,3])
         boolean_mask=array[t,f,f,f,t]
         select_mask = array[boolean_mask]
         print(select_mask)-->0,3
#提取当前 fold 的训练和验证索引
    # k_fold_indices[fold] = (train_indices, val_indices, test_indices)
    train_indices = data['k_fold_indices'][fold][0]
    val_indices = data['k_fold_indices'][fold][1]
    test_indices = data['test_indices']
#GCN模型：
     ————init__(self,in_channels,hidden_channels,out_channels,drop_rate=0.5,num_layers=2)
     super.__init__()
     self.convs=torch.nn.ModlesLists()
     #第一层输入——隐藏
      self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中间层: 隐藏 → 隐藏 (如果 num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 最后一层: 隐藏 → 输出
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.drop_rate = drop_rate
  #前向传播：
         参数:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
        
        返回:
            logits: 原始输出 [num_nodes, out_channels]
        """
        # 遍历所有卷积层
        for i, conv in enumerate(self.convs):
            # 执行图卷积: 聚合邻居信息 + 线性变换
            # 数学: D^{-1/2} A D^{-1/2} H W
            x = conv(x, edge_index)
            
            # 如果不是最后一层，应用 ReLU 和 Dropout
            if i < len(self.convs) - 1:
                x = F.relu(x)                    # 非线性激活
                x = F.dropout(x, p=self.drop_rate, training=self.training)  # 正则化
        
        # 返回 logits (未经过 softmax，留给后续校准处理)
        return x
   #Gat校准：T_i = 1/H Σ softplus( ω δĉ_i + Σ α_{i,j} γ_j τ_j^h ) + T_0
我对于自己家种树做了8次调研，每次线听取自己意见，然后去了解邻居的建议，理我近的邻居更了解我家的情况权重高，意见与我相似的邻居权重页高，我汇总了邻居和自己的意见加上自己和其他邻居的意见的误差乘以权重最后加上总误差得到了一次温度建议
        # ===== Step 1: 骨干模型前向传播 =====
        # logits: [num_nodes, num_classes]
        logits = self.model(x, edge_index)
        
        # ===== Step 2: 计算置信度 =====
        # probs: [num_nodes, num_classes]
        probs = F.softmax(logits, dim=1)
        # confidences: [num_nodes] 每个节点的最大概率
        confidences, _ = probs.max(dim=1)
        
        # ===== Step 3: 计算相对置信水平 δĉ_i =====
        # 公式: δĉ_i = ĉ_i - (1/|N(i)|) * Σ_{j∈N(i)} ĉ_j
        neighbor_conf = self._aggregate_neighbor(confidences, edge_index)
        delta_conf = confidences - neighbor_conf  # [num_nodes]
        
        # ===== Step 4: 计算节点温度贡献 τ_i^h =====
        # 公式 (9): τ_i^h = (θ^h)^T · 排序后的 logits
        # 对 logits 排序: 让模型关注相对分布，而非绝对类别
        sorted_logits, _ = logits.sort(dim=1)  # [num_nodes, num_classes]
        tau = []
        for head in self.linear_layers:
            # 每个头输出一个标量贡献
            tau_h = head(sorted_logits).squeeze(-1)  # [num_nodes]
            tau.append(tau_h)
        tau = torch.stack(tau, dim=1)  # [num_nodes, heads]
        
        # ===== Step 5: 计算注意力系数 α_{i,j} =====
        # 公式 (10): 基于特征相似性计算邻居重要性
        alpha = self._compute_attention(logits, edge_index)  # [num_edges, heads]
        
        # ===== Step 6: 计算距离缩放因子 γ =====
        # 公式 (7): 训练节点用 γ_t，邻居用 γ_n，其余为 1
        gamma = self._compute_gamma()  # [num_nodes]
        
        # ===== Step 7: 聚合邻居贡献 =====
        # 公式 (8) 的求和部分: Σ α_{i,j} · γ_j · τ_j^h
        neighbor_agg = self._aggregate_weighted(tau, alpha, gamma, edge_index)
        # neighbor_agg: [num_nodes, heads]
        
        # ===== Step 8: 计算节点温度 T_i =====
        # 公式 (8): T_i = 1/H Σ softplus( ω δĉ_i + neighbor_agg ) + T0
        temp = self.omega * delta_conf.unsqueeze(1) + neighbor_agg  # [num_nodes, heads]
        temp = F.softplus(temp)  # 确保温度为正
        temp = temp.mean(dim=1) + self.T0  # 多头平均 + 全局偏置
        # temp: [num_nodes]
        
        # ===== Step 9: 应用温度缩放 =====
        # 校准后: p_i = softmax(z_i / T_i)
        # 温度大于 1 使分布更平滑，小于 1 使分布更尖锐
        calibrated_logits = logits / temp.unsqueeze(1)  # [num_nodes, num_classes]
        
        return calibrated_logits
    
    def _aggregate_neighbor(self, confidences, edge_index):
        """计算每个节点的邻居平均置信度"""
        # 实现: 使用 scatter_add 聚合邻居信息
        # 这里仅作为骨架，实际实现需要处理边索引
        # 简化版本: 假设已经实现
        pass
    
    def _compute_attention(self, logits, edge_index):
        """计算注意力系数 α"""
        # 公式 (10): 基于缩放后的 logits 内积
        pass
    
    def _compute_gamma(self):
        """计算距离缩放因子 γ"""
        # 公式 (7): 训练节点 γ_t，邻居 γ_n，其他 1
        gamma = torch.ones(self.num_nodes, device=self.T0.device)
        gamma[self.train_mask] = self.gamma_t
        # 这里需要计算训练节点的一跳邻居
        # 简化版本: 假设已经实现
        return gamma
    
    def _aggregate_weighted(self, tau, alpha, gamma, edge_index):
        """聚合邻居的加权贡献"""
        # 实现: 根据 edge_index 和 alpha 加权聚合
        pass
ece检验校准：ECE = Σ_{m=1}^{M} (|B_m|/n) |acc(B_m) - conf(B_m)|
def compute_ece(confidences, accuracies, n_bins=15):
    """
    计算期望校准误差 (Expected Calibration Error)
    
    参数:
        confidences: [n_samples] 每个样本的预测置信度 (max probability)
        accuracies: [n_samples] 每个样本是否预测正确 (0 或 1)
        n_bins: 分桶数量 (论文常用 15)
    
    返回:
        ece: 标量，校准误差
    
    数学原理:
        1. 将 [0,1] 区间分成 n_bins 个等宽桶
        2. 每个桶内计算: 平均置信度 conf 和 平均准确率 acc
        3. 如果模型完美校准，每个桶的 acc == conf
        4. ECE 是加权平均偏差
    """
    # 1. 创建桶边界: [0, 0.0667, 0.1333, ..., 1.0]
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    n_samples = len(confidences)
    
    for i in range(n_bins):
        # 2. 找出落在当前桶内的样本
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        num_in_bin = in_bin.sum().item()
        
        if num_in_bin > 0:
            # 3. 计算桶内平均置信度
            conf_in_bin = confidences[in_bin].mean().item()
            # 4. 计算桶内平均准确率
            acc_in_bin = accuracies[in_bin].float().mean().item()
            # 5. 计算偏差并加权
            ece += (num_in_bin / n_samples) * abs(acc_in_bin - conf_in_bin)
    
    return ece

def compute_nll(log_probs, targets):
    """
    计算负对数似然 (Negative Log-Likelihood)
    
    数学: NLL = - (1/n) * Σ log(p_i[target_i])
    衡量概率预测的质量，越低越好
    """
    return -log_probs.gather(dim=1, index=targets.unsqueeze(1)).mean().item()

def compute_brier(probs, targets):
    """
    计算 Brier Score
    
    数学: Brier = (1/n) * Σ Σ (p_{i,k} - 1_{y_i=k})^2
    衡量概率预测的均方误差，越低越好
    """
    n_classes = probs.shape[1]
    one_hot = torch.eye(n_classes)[targets].to(probs.device)
    return ((probs - one_hot) ** 2).mean().item()
       
             
