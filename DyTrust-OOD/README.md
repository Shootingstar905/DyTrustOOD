# DyTrust-OOD

动态信任预测的核心模型实现。保留环境分块编码、单向 GRU、时间稳定性选择、条件 CVAE、表示替换、Hodge–Laguerre 结构中介，以及联合训练损失。

本仓库仅提供模型库，不附带数据集、数据预处理、负采样、合成 OOD 特征生成、完整实验运行器、测试或持续集成文件。数据与历史结构输入由调用方提供。

## 安装

需要 Python 3.10 或以上版本。运行时依赖仅为 PyTorch；应按实际 CPU/CUDA 环境安装合适的 PyTorch，然后在仓库根目录执行：

```bash
python -m pip install -e .
```

项目与仓库名称为 `DyTrust-OOD`，Python 导入名称为 `dytrust_ood`。连字符不能用于普通 Python 包导入语句。

```python
from dytrust_ood import DyTrustOOD, ModelConfig, build_hodge

config = ModelConfig(nfeat=8, length=11)
model = DyTrustOOD(config)
```

`nfeat` 是输入特征宽度，需要按自己的输入修改。`length` 是 CVAE 时间条件的容量，不会自动划分训练、验证、测试数据。隐藏宽度为 `n_factors * delta_d`，默认是 `8 * 16 = 128`。

## 目录

```text
DyTrust-OOD/
├── dytrust_ood/
│   ├── __init__.py
│   ├── config.py       # 仅模型配置
│   ├── encoders.py     # 环境图编码、GRU、稳定性选择、CVAE
│   ├── hodge.py        # 稀疏边界算子、Laguerre 滤波、图聚合
│   ├── mediation.py    # 结构中介模块
│   ├── model.py        # 完整模型、表示替换、预测与损失接口
│   └── losses.py       # BCE、干预损失均值与方差
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 输入约定

### 历史快照 `snapshots`

传入长度为 `T` 的列表，`1 <= T <= config.length`。各快照使用相同的节点编号及节点轴大小 `N`，第 `t` 项包含：

| 键 | 张量 | 含义 |
|---|---|---|
| `x` | 浮点 `[N, F]` | 截至该历史时刻可获得的节点特征，`F=config.nfeat` |
| `edge_index` | `torch.long [2, E_t]` | 有向历史信任边，源节点向目标节点传递消息 |
| `active_nodes` | `torch.bool [N]` | 该时刻节点是否已可观察 |

所有历史张量及待预测边须与模型在相同设备上；浮点输入须与模型浮点类型一致。模型不会移动或过滤调用方的数据。历史边不能连接未出现的节点。预测未来时刻时，不能把未来快照或未来标签衍生的特征放入历史列表。

### 结构输入 `structures`

默认开启结构分支，须为每个历史快照提供一个结构字典，包含：

| 键 | 张量／对象 | 含义 |
|---|---|---|
| `edge_index` | `torch.long [2, M_t]` | 与 `hodge.edge_index` 完全一致的规范边及顺序 |
| `edge_features` | 浮点 `[M_t, 3]` | 三列依次为内积、Pearson 相关性、历史度序列 DTW 相似度 |
| `hodge` | `HodgeOperators` | 与该快照对应的稀疏边界算子 |

`structural_features="stable"` 时，前两列在模型前向中根据稳定表示重新计算；第三列仍须调用方提供真实的、仅依赖历史的 DTW 相似度。本库不包含 DTW 特征计算或缓存预处理，不会用零值或随机特征暗中替代它。

`build_hodge` 是模型的数学算子构造函数：接受已经准备好的历史边张量，生成无向简单骨架上的 `B1`、`B2`，不读取任何数据文件。保留它是为了实现 full Hodge：

```python
hodge = build_hodge(
    history_edge_index,
    num_nodes=num_nodes,
    max_triangles=config.max_triangles,
    dtype=next(model.parameters()).dtype,
)
# 按 hodge.edge_index 的顺序准备或重排对应的 edge_features。
structure = {
    "edge_index": hodge.edge_index,
    "edge_features": edge_features,
    "hodge": hodge,
}
```

算子作用于无向结构骨架，但监督信任关系保留方向。`build_hodge` 返回的边序是权威顺序；不能直接把原有向边特征不经对齐就传入。调用方负责保证这些结构张量确实来自对应历史快照，模型不能从数值本身证明其时间来源。

`ModelConfig(no_frontdoor=True)` 可关闭结构分支，此时 `structures=None`。这是不同的模型变体，不能当作完整模型的等价输入方式。

### 监督用户对

`pairs` 为 `torch.long [2, B]`，保留源／目标方向。`labels` 为浮点 `[B]`，二元任务采用 0 或 1。模型不创建正负样本，也不替调用方决定冷启动候选、训练划分或指标口径。

## 预测与训练

以下代码使用调用方已经准备好的 `snapshots`、`structures`、`pairs` 和 `labels`，不是内置数据示例。

```python
import torch

model.eval()
with torch.no_grad():
    probabilities = model(snapshots, pairs, structures, logits=False)
    # 默认 logits=True 时返回未经过 sigmoid 的分数，形状同样是 [B]。

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-7)
model.train()
optimizer.zero_grad(set_to_none=True)
loss, terms = model.compute_loss(snapshots, pairs, labels, structures)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
optimizer.step()
```

`compute_loss` 计算单个预测目标的联合目标：

\[
\mathcal L=\alpha\mathcal L_{\mathrm{CVAE}}
+\beta\frac1J\sum_{j=1}^{J}(\ell_j-\bar\ell)^2+\bar\ell,
\qquad \bar\ell=\frac1J\sum_{j=1}^{J}\ell_j.
\]

`terms` 包含 `cvae`、`variance`、`prediction` 三项未加权损失。各次替换共享原历史、监督边、标签、用户选择和环境池。方差分母为 `J`；`J=1` 时方差为零。默认前向预测不执行随机替换；使用 `factor_selection="random_matched"` 时，块选择本身仍然具有随机性。

上述例子展示一次参数更新。跨多个目标的梯度累积、早停、选模和评估由调用方实现；它不是自动复现原实验训练协议的脚本。所有可学习组件都属于 `model`，无需额外建立遗漏模块的优化器。

## 核心默认设置

| 配置 | 默认值 |
|---|---|
| 环境数／块宽／总宽 | `n_factors=8`，`delta_d=16`，总宽 128 |
| 图编码／时序编码 | 单层环境卷积，单层单向 GRU |
| Dropout | `0.1` 仅作用于多层图编码的层间；默认单层有效值为 0 |
| CVAE | 潜变量 16；隐藏宽度为 `max(32, 2*delta_d, 2*d_for_cvae)` |
| 观测池／生成池 | 上限 4,096／512；CVAE 批大小上限 1,024 |
| 干预 | `J=5`，用户比例 0.6，生成池抽样概率 0.5 |
| 损失权重 | `alpha=0.01`，`beta=0.01` |
| Hodge | full，3 个 Laguerre 项，标量系数，默认不进行谱缩放 |
| 结构与梯度 | `equivariant`／`stable`／`detached` |
| 边分块 | `edge_batch_size=65536`，仅控制解码计算分块 |

配置仅保留模型计算需要的参数；完整字段在 `config.py`。模型默认与精简前版本保持一致，包括未缩放 Hodge 的默认选择。可显式开启 `hodge_normalize=True`，但需在论文配置中报告。`max_triangles` 超出时会报错，不会静默删掉三角形。

## 保存模型

```python
from dataclasses import asdict

torch.save({"config": asdict(config), "state_dict": model.state_dict()}, "model.pt")
state = torch.load("model.pt", map_location="cpu", weights_only=True)
restored = DyTrustOOD(ModelConfig(**state["config"]))
restored.load_state_dict(state["state_dict"], strict=True)
restored.eval()
```

旧实验运行器及其 checkpoint 自动恢复入口已移除；这里保存和恢复的是模型配置及纯参数字典。

## 方法与发布范围

时间稳定性是表示选择启发式，不保证识别因果因素；结构模块是因果启发的预测结构，不是经典前门因果识别。方向一致读出、稳定表示构造边属性和梯度隔离沿用上一版已经明确的实现选择。本次精简不新增实验结果，也不代表论文表格已经复现。

本包未代填作者身份或软件许可证；公开发布前由权利人确认代码来源、必要署名及授权条款。
