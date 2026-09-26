# OA-STID · Eg=β + limnbr k=3 · nobehav 消融包（自包含）

**完整目录 / 每个 `.py` 职责 / 模型代码位置 → 见 [`PACKAGE.md`](PACKAGE.md)。**

## 可删已删
- `cache/`、`.recover_tools/`（恢复期残留）
- behav/AVWGCN/road-text/zerospace 等 slim 入口未接线补丁；`*_source.py` / 旧 `*_behav_ext` 入口；根目录误落的 `slurm-*.out`；`__pycache__`

## 包外依赖（仅此两类资源软链；代码不 import 父仓）
- `data` → 数据集
- `Qwen3-Embedding-0.6B` → 语义编码（`use_sem_emb=1` 时）
- Conda Python（环境，非项目代码）

## 入口 / 跑法
- 入口：`stage2/oastid_full_huber_egbeta_limnbr_entry.py`
- 训练：`sbatch --chdir=$PWD --qos=normal run_oastid_nobehav_egbeta_limnbr_ablation_slurm.sh nobehav sd`
- 评估：`sbatch --chdir=$PWD --qos=normal run_oastid_nobehav_egbeta_limnbr_eval_ckpt_slurm.sh nobehav sd`
- 整表：`bash submit_egbeta_limnbr_repro.sh`（可选 `GOLD_ROOT=...` 对照外部金标）
# TransSTNI
# TransSTNI
