# OA-STID 包结构说明（Eg=β · limnbr k=3 · nobehav）

本文说明本目录**每个文件夹装什么**、**每个 `.py` 干什么**，以及**模型 / 模块代码落在哪里**。  
配置身份：`fusion=concat`（Eg=β 拼进 Z）、`sage_full_neighbor=0`、`sage_k=3`、`use_behav_emb=0`（nobehav 族）。

---

## 1. 一句话：代码在哪

| 你要找的东西 | 位置 |
|---|---|
| **主模型** `OASTIDTransfer`（时序骨干 + 可选语义/BSTS/空间 Z + 预测头） | `stage2/oastid_full.py`（marshal 嵌入的完整逻辑） |
| **构图 / 换 Orbit GNN** `build_graph`、`_maybe_swap_orbit_gnn`、`build_oastid_model` | 同上 |
| **空间编码器 GraphSAGE**（3× SAGEConv） | `lib/orbit_graphsage.py` |
| **BSTS 统计编码器** `BSTSEncoder` | `lib/bsts_stats.py` |
| **道路语义文本 → Qwen 嵌入** | `lib/road_semantic.py`（权重走软链 `Qwen3-Embedding-0.6B`） |
| **行为原型 π（本包训练关掉，但模块仍被主文件 import）** | `lib/behav_type.py`、`lib/behav_zeroshot.py` |
| **Huber 训练循环**（Stage1 / Stage2 ERM·MAML·MLDG） | `stage2/oastid_full_huber.py` |
| **本包正式入口**（接线 limnbr + freezesem + nobeta 等） | `stage2/oastid_full_huber_egbeta_limnbr_entry.py` |
| **指标 MAE/RMSE/Masked-MAPE** | `lib/metrics.py`（Cross 评测经 `oastid_full` 的 `eval_target`） |
| **权重与 Cross 指标 JSON** | `output/oastid/`、`output/cross_domain/` |

> **可读性说明**：`oastid_full.py`、`bsts_stats.py`、`road_semantic.py`、`behav_*.py` 是从原 `.pyc` 用 **marshal 字节码 + `exec`** 恢复的壳文件——运行行为与原编译模块一致，但**不是**漂亮明文源码。真正可读的训练改动主要在 `oastid_full_huber.py` 与各 `*_patch.py`。

---

## 2. 目录总览

```
oastid_egbeta_limnbr_k3_full/
├── stage2/          # 训练管线 + 主模型（入口在这里）
├── lib/             # 数据、指标、GNN、语义/BSTS/行为、运行时补丁
├── scripts/         # Slurm 预检小工具
├── output/          # 训练 ckpt + Cross 指标（运行产物）
├── logs/            # sbatch / submitter 日志（运行产物）
├── data -> …        # 数据集软链（包外）
├── Qwen3-Embedding-0.6B -> …  # 语义模型软链（包外）
├── *.sh / compare_*.py / README.md / PACKAGE.md / env 文件
```

**代码不 import 父仓 `iclr27-main`。** 仅资源软链 + Conda Python。

---

## 3. 调用链（训练 / 评测）

```
sbatch run_oastid_nobehav_egbeta_limnbr_{ablation|eval_ckpt}_slurm.sh
        │
        ▼
stage2/oastid_full_huber_egbeta_limnbr_entry.py     ← 本包唯一推荐入口
        │  安装补丁：nospace / nobeta / skip_stage3 / seed / require_gpu / sage limnbr
        ▼
stage2/oastid_full_huber_freezesem.py               ← Stage2 冻结 sem_proj 外环梯度
        ▼
stage2/oastid_full_huber.py                         ← Huber Stage1/2 循环；复用 base.*
        ▼
stage2/oastid_full.py                               ← OASTIDTransfer、build_*、eval_target、main
        │
        ├── lib/orbit_graphsage.py                  ← beta_net 槽位上的 GraphSAGE
        ├── lib/bsts_stats.py / road_semantic.py    ← 侧信息
        ├── lib/behav_type.py / behav_zeroshot.py   ← π（本 ablation use_behav_emb=0）
        ├── lib/dataloader.py / metrics.py / …
        └── lib/sage_nosubgraph_patch.py 等         ← 运行时改 forward / Stage2 采样
```

**有 Z（非 nobeta）**：Stage1 → Stage2 ERM（`meta_iters=50`，随机子图）→（推荐）`--eval_ckpt` Cross。  
**无 Z（`*nobeta*`）**：`no_beta_net=1`，`skip_stage2=1`（s1only）→ eval_ckpt Cross。

---

## 4. `stage2/` — 管线与模型

| 文件 | 作用 |
|---|---|
| **`oastid_full.py`** | **核心仓库**：模型类、构图、Stage1/2/Cross 原版逻辑（MAE 训练版骨架）。含 `OASTIDTransfer`、`OASTIDAttnTransfer`、`BETAET`、`MultiLayerPerceptron`、`IdentityQueryCrossAttention`；`build_oastid_model` / `build_graph` / `train_source` / `meta_train*` / `eval_target` / `main` / `parse_args` 等。 |
| **`oastid_full_huber.py`** | 在**不改** `oastid_full.py` 的前提下，把训练损失换成 Huber（Stage1 原尺度 δ≈23；Stage2 归一化尺度 δ≈0.15）；val/Cross 仍用 MAE。从 `base`（即 `oastid_full`）绑定数据与模型 helper，重写 `train_source` / `meta_train*` 等。 |
| **`oastid_full_huber_freezesem.py`** | 给 Stage2（含 MAML 路径）装 hook：`sem_proj` 仍参与计算图，但**外环 meta-grad 置零**，使 Cross 用的语义投影保持 Stage1。 |
| **`oastid_full_huber_egbeta_limnbr_entry.py`** | **正式 CLI 入口**。只接线本包需要的补丁与 CLI（`no_beta_net` / `skip_stage3` / `spatial_avwgcn=0`），再把 Stage2 ERM 包一层 freezesem + limnbr GraphSAGE。训练脚本 `ENTRY=` 指向此文件。 |

### 4.1 模型里各块在代码中的对应关系

| 模块概念 | 实现位置 |
|---|---|
| 时序嵌入 / 日历 / 融合预测 | `OASTIDTransfer`（`stage2/oastid_full.py`） |
| 空间角色 Z：`GraphSAGE(A, F) → LN` 再与 β 等融合 | `lib/orbit_graphsage.py` 的 `GraphSAGE` / `OrbitGraphSAGE`；由 `_maybe_swap_orbit_gnn` 塞进 `beta_net` |
| **Eg=β**：节点/边属性进空间支路 | 同在 `OASTIDTransfer.forward` + Orbit 适配；本包 `spatial_avwgcn=0`（不用 AVWGCN 替换） |
| BSTS 四组统计 → 投影 | `lib/bsts_stats.py` · `BSTSEncoder` |
| Qwen 路段文本嵌入 → `sem_proj` | `lib/road_semantic.py` · `load_or_encode_road_sem`；投影层在 `OASTIDTransfer` |
| 行为嵌入 π | `lib/behav_type.py`（源域）、`lib/behav_zeroshot.py`（跨域检索）；本包 `use_behav_emb=0` |
| **删 Z（nobeta）** | `lib/nobeta_orbit_patch.py`：拆掉 `beta_net`/`ln_beta`，forward 不走空间支路 |
| **limnbr k=3** | `lib/sage_nosubgraph_patch.py`：`--sage_full_neighbor 0` + `sage_k=3`；Stage2 随机子图采样 |

---

## 5. `lib/` — 库与补丁

分三类：**数据/指标**、**特征与 GNN**、**运行时补丁**（`install_*` 在 import 时改类或函数）。

### 5.1 数据与指标

| 文件 | 作用 |
|---|---|
| `load_dataset.py` | 从包内 `data/` 读 npz 等时空序列（`load_st_dataset`）。 |
| `add_window.py` | 滑窗构造 `X`/`Y`（`Add_Window_Horizon`）。 |
| `normalization.py` | 标准化 / MinMax 等 scaler。 |
| `dataloader.py` | 组装 DataLoader：读数据 → 归一化 → 滑窗 → batch。 |
| `metrics.py` | `MAE_torch` / RMSE / MAPE 等（含 mask）。 |
| `ydzt_sampler_random.py` | Stage2 子图节点数采样 `build_sizes`（**已改为吃全局 `np.random`**，保证可复现）。 |

### 5.2 特征与 GNN（模型侧模块）

| 文件 | 作用 |
|---|---|
| **`orbit_graphsage.py`** | **空间模块明文源码**：原版 3 层 SAGEConv + `sample_neighbors`；`OrbitGraphSAGE` 适配 `beta_net(node_feat, edge_index, edge_weight)`。 |
| **`bsts_stats.py`** | BSTS：时序 / 邻居 / 秩 / 谱统计与 `BSTSEncoder`（marshal）。 |
| **`road_semantic.py`** | 路段 meta → 文本 → Qwen embedding 缓存（marshal）。 |
| **`behav_type.py`** | 源域行为特征 φ、聚类、π 表（marshal）。本包训练不启用 behav emb，但 `oastid_full` 仍会 import。 |
| **`behav_zeroshot.py`** | 目标域零样本 π 检索 / 合成（marshal）。 |

### 5.3 运行时补丁（本包入口会装）

| 文件 | 作用 |
|---|---|
| `sage_nosubgraph_patch.py` | **本消融关键补丁**：限邻居 SAGE、Stage2 随机子图 ERM、相关 CLI；并挂上 `fullgraph_pe_patch`。 |
| `fullgraph_pe_patch.py` | Stage2 子图训练时可选切片全图 Laplacian PE（`--fullgraph_pe_slice`）。 |
| `nobeta_orbit_patch.py` | `--no_beta_net 1`：彻底去掉空间 Z 支路（nobeta 族）。 |
| `nospace_orbit_patch.py` | `node_dim=0` 时避免 `UninitializedParameter.numel()` 崩溃。 |
| `skip_stage3_patch.py` | 训练结束跳过 inline Stage3；真正 Cross 用 `--eval_ckpt`（若 `eval_ckpt` 已设则仍跑 Cross）。 |
| `repro_seed.py` | `seed_everything` + 给 `parse_args` 打补丁。 |
| `require_gpu_patch.py` | 请求 CUDA 却无 GPU 时快速失败。 |

---

## 6. 根目录脚本与其它文件

| 文件 | 作用 |
|---|---|
| `run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh` | **训练** sbatch：设 `ENTRY`、消融 flag、`SAGE_K`、`META_ITERS`、写 `output/oastid/...`。 |
| `run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh` | **评测** sbatch：加载 `best.pt`，`--eval_ckpt`，写 `output/cross_domain/.../*_metrics.json`。 |
| `run_egbeta_limnbr_full_{train,eval}_slurm.sh` | 只跑 Full=`nobehav` 的薄封装。 |
| `submit_egbeta_limnbr_repro.sh` | 8 变体 × 3 源域：先 train 再 eval；可选 `GOLD_ROOT` 对比。 |
| `submit_oastid_nobehav_egbeta_limnbr_ablation.sh` | 另一套提交/轮询脚本（同网格）。 |
| `compare_egbeta_limnbr_repro.py` | 本包 Cross 指标 vs 外部金标目录逐路由对比。 |
| `scripts/oastid_require_gpu.sh` | 作业起步前检查 CUDA。 |
| `environment.yaml` / `requirements.txt` | 环境依赖声明。 |
| `README.md` | 短用法；本文为完整结构说明。 |

---

## 7. `output/` 与 `logs/`（运行产物，非源码）

### `output/oastid/`

每个源域一次训练一个目录，例如：

`largest_{sd|gla|gba}_..._egbeta_limnbr_{s2i50|s1only}_abl_{variant}/`

常见文件：`best.pt`、`stage1.pt`、训练日志侧写的配置等。

### `output/cross_domain/`

跨域评测目录，例如：

`oastid_largest_{src}_to_largest_{tgt}_..._abl_{variant}/`  
以及带 `_eval_ckpt` 后缀的正式评测目录。

内含 `largest_{tgt}_metrics.json`（MAE / RMSE / Masked-MAPE 等）。

### `logs/`

- `submit_*.log` / `*.pid`：提交器  
- `oastid_nobehav_egbeta_limnbr_*.{out,err}`：各 job 标准输出/错误  
- `ablation_per_route_summary.txt`：人工汇总表（若生成过）

---

## 8. 包外软链

| 路径 | 指向 | 用途 |
|---|---|---|
| `data` | OAGNN 数据目录 | 流量 / 图 / meta |
| `Qwen3-Embedding-0.6B` | 父仓或共享权重 | `use_sem_emb=1` 时编码路段文本 |

---

## 9. 消融变体 ↔ 代码开关（本包网格）

| 变体名（脚本参数） | 含义 | 主要开关 |
|---|---|---|
| `nobehav` | Full = Q+BSTS+Z，无 behav | `use_behav_emb=0` |
| `nosem_nobehav` | −Qwen | `use_sem_emb=0` |
| `nobsts_nobehav` | −BSTS | 关 BSTS 相关 flag |
| `nobsts_nosem_nobehav` | −Q−BSTS | 上两者 |
| `nobeta_nobehav` | −Z，Stage1-only | `no_beta_net=1`，`skip_stage2` / s1only |
| `nosem_nobeta_nobehav` 等 | −Q/−B 与 −Z 组合 | 同上叠加 |

具体 CLI 拼装见 `run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh`。

---

## 10. 想改模型时改哪里

1. **换损失 / Stage1·2 训练日程** → `stage2/oastid_full_huber.py`（明文）。  
2. **换邻居数、子图大小、Stage2 是否全图** → `lib/sage_nosubgraph_patch.py` + 训练脚本环境变量。  
3. **改 GraphSAGE 结构（层数/采样）** → `lib/orbit_graphsage.py`（明文）。  
4. **改主干融合、预测头、数据管线** → 需动 `stage2/oastid_full.py`（目前为 marshal；改逻辑成本高，优先打 patch）。  
5. **本包接线方式** → 只改 `oastid_full_huber_egbeta_limnbr_entry.py`，避免再引入已删的 behav/AVWGCN 栈。

---

## 11. 快速跑法（备忘）

```bash
cd /path/to/oastid_egbeta_limnbr_k3_full
sbatch --chdir=$PWD --qos=normal run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh nobehav sd
sbatch --chdir=$PWD --qos=normal run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh nobehav sd
# 整表
bash submit_egbeta_limnbr_repro.sh
```
