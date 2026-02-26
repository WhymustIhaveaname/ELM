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

**方程**: u''(x) - 10u(x) = f(x), x ∈ [0, 8]，Dirichlet BC，解析解已知。配置：N_e=4 子域, Q=100 配点/子域, R_m=3.0, tanh 激活, scipy.linalg.lstsq (LAPACK gelsd)。

| M | 我们 max_err | 论文 max_err | 比值 | 我们 rms_err | 论文 rms_err | 比值 |
|---|---|---|---|---|---|---|
| 75 | 8.96e-9 | 4.02e-8 | 0.22x | 2.07e-9 | 5.71e-9 | 0.36x |
| 100 | 1.07e-9 | 1.56e-9 | 0.68x | 2.27e-10 | 2.25e-10 | 1.01x |
| 125 | 1.71e-10 | 1.42e-10 | 1.20x | 4.26e-11 | 2.55e-11 | 1.67x |

全部在论文值的 2 倍以内，收敛趋势一致。JAX 与 TF 的 PRNG 不同导致随机权重不同，故精确数字不完全吻合；关键发现是 `scipy.linalg.lstsq` 比 `jnp.linalg.lstsq` 精度高约一个量级。
