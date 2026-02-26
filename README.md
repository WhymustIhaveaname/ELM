# ELM — Extreme Learning Machines for PDEs

JAX 复现 Dong 组的 locELM 系列工作。

## Quickstart

```bash
cd ELM
uv sync
source .venv/bin/activate
python examples/helm1d.py
```

## 复现报告：locELM 1D Helmholtz

**论文**: S. Dong & Z. Li, *Local Extreme Learning Machines and Domain Decomposition for Solving Linear and Nonlinear PDEs*, CMAME 2021 (arXiv:2012.02895)

**方程**: u''(x) - 10u(x) = f(x), x ∈ [0, 8]，Dirichlet BC，解析解已知。配置：N_e=4 子域, Q=100 配点/子域, R_m=3.0, tanh 激活, scipy.linalg.lstsq (LAPACK gelsd)。变化 M（每个子域的基函数个数 = 隐藏层节点数）：

| M | 我们 max_err | 论文 max_err | 比值 | 我们 rms_err | 论文 rms_err | 比值 |
|---|---|---|---|---|---|---|
| 75 | 8.96e-9 | 4.02e-8 | 0.22x | 2.07e-9 | 5.71e-9 | 0.36x |
| 100 | 1.07e-9 | 1.56e-9 | 0.68x | 2.27e-10 | 2.25e-10 | 1.01x |
| 125 | 1.71e-10 | 1.42e-10 | 1.20x | 4.26e-11 | 2.55e-11 | 1.67x |

全部在论文值的 2 倍以内，收敛趋势一致。JAX 与 TF 的 PRNG 不同导致随机权重不同，故精确数字不完全吻合；关键发现是 `scipy.linalg.lstsq` 比 `jnp.linalg.lstsq` 精度高约一个量级。

## 复现报告：locELM 2D Helmholtz

**方程**: ∇²u - 10u = f(x,y), (x,y) ∈ [0, 3.6]²，Dirichlet BC，解析解为可分离形式。配置：2×2 子域, R_m=1.5, tanh 激活, C^1 连续性在子域边界线上施加。

| Q | M | 我们 max_err | 论文 max_err | 比值 | 我们 rms_err | 论文 rms_err | 比值 |
|---|---|---|---|---|---|---|---|
| 20×20 | 300 | 6.32e-4 | 7.28e-4 | 0.87x | 6.17e-5 | 5.28e-5 | 1.17x |
| 25×25 | 400 | 1.47e-5 | 2.01e-5 | 0.73x | 1.51e-6 | 1.41e-6 | 1.07x |

全部在 2 倍以内。运行: `python examples/helm2d.py`

## Scaling Law 研究（2026-02-26）

### Parameter scaling law（`examples/scaling_law.py`）

固定 Q=100，变化 NM ∈ {128, 256, 512, 1024}，考察 RMSE ∝ P^α（P=3NM）：

| N | α | R |
|---|---|---|
| 1 (vanilla) | -1.63 | -0.786（曲线不单调，N=1 在大 M 下欠定） |
| 2 | -5.09 | -0.976 |
| 4 (paper) | -6.68 | -0.888 |

### Q 饱和实验（`examples/q_saturation.py`）

固定 NM=1024，扫描 Q ∈ {25…6400}，寻找 RMSE 饱和点：

| N | M | 饱和 Q | 饱和 RMSE |
|---|---|---|---|
| 1 | 1024 | 不存在（条件数极差，始终震荡） | ~0.01–1 |
| 2 | 512 | ~100–200 | ~1.5e-8 |
| 4 | 256 | ~100–200 | ~2e-11（近机器精度） |

**结论**：Q 过大（>800）反而因矩阵条件数变差而精度略降。推荐 **Q=400**（≈ 2×饱和点）用于后续 parameter scaling law 实验。
