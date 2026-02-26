# Dong 组 ELM 系列工作笔记

三篇文章来自 Purdue 的 Suchuan Dong 组，围绕同一个核心方法——**Extreme Learning Machine (ELM) 求解 PDE**——在三个方向上展开。

## 核心方法：Randomized Neural Network + Least Squares

ELM 的核心思想极其简单：用一个浅层前馈神经网络表示 PDE 的解，但**隐藏层的权重/偏置随机赋值后固定不动**，只训练输出层的线性权重。这意味着隐藏层的输出 $V_j(\mathbf{x})$ 本质上是一组**随机但固定的非线性基函数**，解被表示为这些基函数的线性组合：

$$u(\mathbf{x}) = \sum_{j=1}^{M} w_j V_j(\mathbf{x})$$

训练过程不走反向传播，而是将 PDE 残差、边界条件、初始条件在配点上离散化后，直接通过**线性最小二乘**（线性 PDE）或**非线性最小二乘**（非线性 PDE）求解输出层权重。导数通过自动微分计算。

这个设计带来两个关键优势：
- **极快的训练速度**：归结为一个（非线性）最小二乘问题，而非梯度下降迭代
- **指数收敛**：误差随配点数或训练参数数量指数下降，可逼近机器精度

随机权重的最大幅值 $R_m$（即权重从 $[-R_m, R_m]$ 均匀采样）是一个重要的超参数，中等大小的 $R_m$ 通常效果最好。

---

## 文章一：locELM — 区域分解 + 局部 ELM（正问题）

> S. Dong & Z. Li. *Local Extreme Learning Machines and Domain Decomposition for Solving Linear and Nonlinear PDEs.* CMAME, 2021. (arXiv:2012.02895)

**动机**：单个全局 ELM 在大域或复杂解上能力有限。

**方法**：将计算域 $\Omega$ 分成 $N_e$ 个不重叠的子域，每个子域分配一个独立的浅层局部神经网络（locELM），子域间通过 $C^k$ 连续性条件耦合（$k$ 由 PDE 阶数决定，如二阶 PDE 需要 $C^1$ 连续）。

关键要素：
- **线性 PDE**：所有子域的 PDE 残差 + 边界条件 + $C^k$ 连续性条件拼成一个线性方程组，线性最小二乘一步求解
- **非线性 PDE**：方程组变为非线性，提出两种求解策略：
  - NLSQ-perturb：非线性最小二乘 + 随机扰动重启机制，避免陷入局部极小
  - Newton-LLSQ：Newton 迭代 + 每步用线性最小二乘求解 Jacobian 系统（更快但精度略低）
- **Block time-marching**：长时间模拟时将时间轴分块，逐块求解，每块内部是一个中等大小的时空域上的 locELM 问题，用前一块末时刻的解作为下一块的初始条件

**结果**：
- 误差随自由度指数下降
- 对比 DGM / PINN：精度和速度**高出若干个量级**
- 对比经典 FEM：精度和计算成本**可比甚至更优**
- 固定总自由度时，增加子域数可大幅降低训练时间而精度不变

---

## 文章二：逆问题 — 用 ELM 求解参数化 PDE 的逆问题

> S. Dong & Y. Wang. *A Method for Computing Inverse Parametric PDE Problems with Random-Weight Neural Networks.* JCP, 2023. (arXiv:2210.04338)

**动机**：实际应用中 PDE 的系数（如扩散系数、波速）往往未知，需要从稀疏测量数据中反演。

**问题设定**：参数化 PDE 形如

$$\alpha_1 \mathcal{L}_1(u) + \alpha_2 \mathcal{L}_2(u) + \cdots + \mathcal{F}(u) = f$$

给定稀疏的测量数据 $\mathcal{M}u(\xi) = S(\xi)$，同时求解未知参数 $\boldsymbol{\alpha}$ 和场解 $u(\mathbf{x})$。

**方法**：继续使用 locELM 表示解场，但现在系统中有两组未知量——网络可训练参数 $\boldsymbol{\beta}$ 和逆参数 $\boldsymbol{\alpha}$。提出三种算法：

1. **NLLSQ**：将 $\boldsymbol{\alpha}$ 和 $\boldsymbol{\beta}$ 打包在一起，直接用 NLLSQ-perturb 求解整个系统
2. **VarPro-F1**（Variable Projection 形式一）：通过变量投影消去 $\boldsymbol{\alpha}$，得到只关于 $\boldsymbol{\beta}$ 的约化问题；先用 NLLSQ-perturb 解 $\boldsymbol{\beta}$，再用线性最小二乘回代求 $\boldsymbol{\alpha}$
3. **VarPro-F2**（形式二，与 F1 互逆）：消去 $\boldsymbol{\beta}$，得到只关于 $\boldsymbol{\alpha}$ 的约化问题；先解 $\boldsymbol{\alpha}$，再回代求 $\boldsymbol{\beta}$。适用于对 $u$ 线性的 PDE

**结果**：
- 无噪声数据：逆参数和场解的误差随配点数/训练参数数指数下降，可达机器精度附近
- 有噪声数据：精度下降但仍然相当准确，通过缩放测量残差系数 $\lambda_{\text{mea}}$ 可改善
- 对比 PINN：精度和训练时间均有明显优势

---

## 文章三：高维 PDE — 将 ELM 推广到高维

> Y. Wang & S. Dong. *An Extreme Learning Machine-Based Method for Computational PDEs in Higher Dimensions.* CMAME, 2024. (arXiv:2309.07049)

**动机**：传统网格方法在高维（$d > 3$）下因维度灾难而不可行。ELM 的随机基函数具有与维度无关的逼近收敛率 $O(1/\sqrt{n})$（Igelnik & Pao 1995 的理论结果），这为高维提供了理论支撑——相比之下，确定性基函数的逼近率为 $O(1/n^{1/d})$，不可避免地受维度灾难影响。

**方法一：ELM + 随机配点**

与低维 ELM 的两个关键区别：
- 配点从均匀网格**改为随机采样**（高维下网格点数量指数增长，不可行）
- 使用 locELM 时，区域分解**至多沿 $\mathcal{M}$ 个方向**进行（文中取 $\mathcal{M}=2$），避免子域数量指数增长

内部配点数量 $N_{\text{in}}$ 对精度的影响极小（这与低维时的表现截然不同），边界配点数量 $N_{\text{bc}}$ 才是关键。

**方法二：ELM/A-TFC（近似泛函连接理论）**

TFC（Theory of Functional Connections）提供了一种系统性方法，通过构造"约束表达式"自动满足边界条件，但其项数随维度指数增长。本文提出 A-TFC：保留 TFC 层次分解中的主要项，截断高阶项，使项数不再指数增长。A-TFC 约束表达式中的自由函数由 ELM 表示，训练方式与方法一类似。

**结果**：
- 两种方法均可求解高维线性/非线性稳态/动态 PDE，低维时误差接近机器精度
- 维度增高时精度下降但仍可控
- ELM 和 ELM/A-TFC 精度相当，A-TFC 在低维时略好但计算量更大
- 对比 PINN：精度和训练时间均**显著优于** PINN

---

## 同组其他相关工作

### 超参数与实现改进

> S. Dong & J. Yang. *On Computing the Hyperparameter of ELMs: Algorithm and Application to Computational PDEs, and Comparison with Classical and High-Order Finite Elements.* CMAME, 2022. (arXiv:2110.14121, 被引 67)

解决 $R_m$ 选取难题。具体做法：将 $[-R_m, R_m]$ 的随机系数实现为固定随机向量 $\boldsymbol{\xi} \in [-1,1]$ 乘以缩放因子 $R_m$，然后最小化残差范数 $\|\mathbf{r}(R_m)\|$ 来寻找最优 $R_{m0}$，优化算法为**差分进化**（种群 6–10，搜索范围 $[0.01, 5]$，约 50 代）。区分两种配置：

- **Single-$R_m$-ELM**：所有隐藏层共用一个 $R_m$。$R_{m0}$ 对配点数几乎不敏感，对训练参数数量弱依赖，随隐藏层数增加而减小
- **Multi-$R_m$-ELM**：每层独立 $R_m$，精度更高但优化成本更大

另一关键改进：用 TensorFlow **ForwardAccumulator**（前向模式自动微分）替代 GradientTape（反向模式）计算 $V_j$ 的导数——因为 ELM 末隐藏层节点数远大于输入维数，前向模式更高效。还指出应避免基于法方程的 LLSQ（如 TF 默认 lstsq），改用 LAPACK（scipy）减轻病态问题。

对比 FEM 结果：ELM 远超经典二阶 FEM；与高阶 FEM 存在交叉点——小规模时两者接近（高阶 FEM 略优），大规模时 ELM 明显更优；时变 PDE + BTM 场景下 ELM 几乎全面胜出。

### 网络架构改良：HLConcELM

> N. Ni & S. Dong. *Numerical Computation of PDEs by Hidden-Layer Concatenated ELM.* CMAME, 2023. (arXiv:2204.11375, 被引 26)

常规 ELM 的致命缺陷：精度完全取决于**最后一层**隐藏层的宽度。即使前面的层很宽（如 300 节点），只要最后一层窄（如 30 节点），精度就会崩溃（误差可达 $10^2$），因为只有最后一层直接连到输出，前面各层的自由度被浪费。

HLConcELM 在输出层前加入**逻辑拼接层**，将所有隐藏层的输出拼接后送入输出层：$u(\mathbf{x}) = \sum_{i=1}^{L-1}\sum_{j=1}^{M_i}\beta_{ij}\phi_j^{(i)}(\mathbf{x})$。这样所有隐藏节点都参与线性组合。隐藏层权重仍随机固定，每层用独立的 $R_i$（hidden magnitude vector $\mathbf{R}$），由差分进化优化。

**表示能力定理**：对 HLConcFNN，增加隐藏层或增加某层节点数时表示能力**单调不减**（传统 FNN 无此保证——加层后精度可能反而下降）。实验对比：架构 $[2,300,30,1]$ 时，传统 ELM 最大误差 $\sim 10^2$，HLConcELM 可达 $\sim 10^{-7}$。

### 训练策略改良：VarPro + ANN

> S. Dong & J. Yang. *Numerical Approximation of PDEs by a Variable Projection Method with ANNs.* JCP, 2022. (arXiv:2201.09989, 被引 22)

与 ELM 互补的路线：**不固定隐藏层权重**，而是用 VarPro 消去输出层线性参数 $\boldsymbol{\beta}$，得到只关于隐藏层参数 $\boldsymbol{\theta}$ 的降维问题，再用 Gauss-Newton + trust region 求解。Jacobian 采用 Kaufman 简化（只保留第一项）减少计算量。

对非线性 PDE，问题不可分离，不能直接用 VarPro。关键创新：用 Newton 迭代线性化时，**写成关于 $u^{k+1}$ 的形式**（而非增量 $v = u^{k+1} - u^k$），线性化后可直接套用 VarPro。实验表明这两种线性化形式的精度差距可达**两个量级以上**。

精度**显著高于** ELM，但训练时间也远高于 ELM（对流方程 $20 \times 20$ 配点：VarPro 约 78.2s vs ELM 约 0.32s）。将非线性最小二乘迭代次数设为 0 时，VarPro 退化为 ELM。VarPro 对 $R_m$ 的敏感性低于 ELM。

### 学习时间积分算法

> S. Dong & N. Ni. *Learning the Exact Time Integration Algorithm for Initial Value Problems by Randomized Neural Networks.* 2025. (arXiv:2502.10949)

将 ODE 初值问题 $dy/dt = f(y,t)$ 的精确时间推进映射 $\psi(y_0, t_0, h)$ 建模为一个满足 PDE 的"算法函数"，用 ELM 以物理信息方式求解。解的形式为 $\psi = F + \xi^{s+1}\varphi$，其中 $F$ 是已知近似（如 Euler、中点法），$\varphi$ 由 ELM 逼近。训练完成后，网络即为一个可复用的时间积分器，支持任意初值和步长。

提出显式（NN-Exp-S0/S1/S2）和隐式（NN-Imp-S1/S2）两类形式。显式优于隐式；NN-Exp-S0 在多数问题上表现最好。若 $f$ 对 $t$ 或 $y$ 的某分量有周期性，则 $\psi$ 有周期/平移关系，可缩小训练域。Stiff 问题用区域分解 + 准自适应步长策略处理。子域上的局部 NN 可并行训练。

误差近指数下降；对比 scipy 的 DOP853/RK45/Radau/BDF，NN-Exp-S0 在多数基准问题上**性能更优**。目前缺少真正的自适应步长策略，是后续方向。

### PINN 理论分析（同组）

> Y. Qian, Y. Zhang, Y. Huang & S. Dong. *Error Analysis of PINN for Approximating Dynamic PDEs of Second Order in Time.* JCP, 2024. (arXiv:2303.12245, 被引 32)

针对**双曲型 PDE**（波动方程、Sine-Gordon 方程、线弹性动力学）的 PINN 误差分析——此前理论集中在椭圆/抛物型。使用两层隐藏层 + $\tanh$ 的 FNN，基于 De Ryck 等的 Sobolev 逼近结果。主要结论：解场、时间导数、梯度的近似误差可由**训练 loss** 和**求积点数**有效界定。理论分析揭示标准 PINN loss 缺少若干关键残差项（初值梯度残差、边界条件时间导数残差等），加入后得到改进的 PINN 变体。

> Y. Qian, Y. Zhang & S. Dong. *Error Analysis and Numerical Algorithm for PDE Approximation with HLConcPINN.* 2024. (arXiv:2406.06350)

将 HLConcFNN 与 PINN 结合，配合**改进的块时间推进（ExBTM）**，建立抛物型（热方程、Burgers）和双曲型（波动、非线性 Klein-Gordon）PDE 的误差界。核心突破：前两层隐藏层用 $\tanh$，其余层可用**任意常用光滑激活函数**，网络深度可任意（$\geq 2$ 隐藏层），打破了此前理论必须 2 层 + $\tanh$ 的限制。ExBTM 相比原始 BTM 更便于理论分析——解决了多时间块时初值正则性难以控制的问题。

### 泛函连接元方法

> J. Yang & S. Dong. *A Functionally Connected Element Method for Solving Boundary Value Problems.* CMAME, 2025. (arXiv:2403.06393)

基于泛函连接理论（TFC）构造分片函数的一般形式 $u = g - \mathcal{A}g + \mathcal{A}G$，使其在子域边界**内禀满足** $C^0$ 或 $C^1$ 连续性（不像 locELM 需要额外施加连续性条件作为约束）。提供三种模式：$C^1$ FCE（自动满足 $C^0$ 和 $C^1$）、$C^0$ FCE（自动满足 $C^0$，$C^1$ 由最小二乘施加）、FCE-NC（全部由最小二乘施加）。自由函数可用 Legendre 多项式或准随机正弦基表示，误差随基函数数量呈指数收敛（谱型精度）。

独特优势：对含**相对边界条件**（解或导数之间的线性/非线性约束，如 $u(a) = 2u(b)$）的问题，FCE 可精确满足，传统有限元/谱元处理困难。目前仅适用于规则剖分（单元边界与坐标线对齐），扩展到一般剖分是后续工作。

---

## 系列工作全景

```
                        locELM (2021) ← 奠基
                        区域分解 + 随机 NN + 最小二乘
                       /       |        \
       超参数 R_m (2022)   HLConcELM (2022)   VarPro+ANN (2022)
       自动调参           拼接隐藏层           训练隐藏层
              \            |            /
               ————————————————————————
                    |             |
            逆问题 (2023)    高维 PDE (2024)
            VarPro 消参      随机配点 + A-TFC
                    |             |
            学习时间积分 (2025)  FCE 方法 (2025)
            ELM 学积分器      内禀连续性
```

**理论线** (Qian, Zhang & Dong)：PINN 误差分析 (2023) → HLConcPINN 理论 (2024)

核心优势一脉相承：
- 不依赖梯度下降，训练极快（ELM 典型训练在秒级，VarPro 更重但仍远快于 PINN）
- 指数收敛，精度远超 PINN/DGM
- 性能可比甚至超越经典 FEM 和高阶 FEM（大规模时尤甚）

**精度–成本谱**：ELM（快/低精度上限）→ HLConcELM（快/高精度）→ VarPro（慢/最高精度），三者可按需选择

---

## 代码可用性

**该组无公开代码仓库。** 10 篇论文的 tex 源文件中均未提供 GitHub 等代码链接；GitHub 上也未找到官方实现。唯一的声明来自 PINN 误差分析一文（2303.12245）："Data will be made available on reasonable request."，需联系作者（sdong@purdue.edu）索取。

各论文提及的技术栈：

| 工作 | 框架 | 关键依赖 |
|------|------|----------|
| locELM, R_m, HLConcELM, VarPro, 逆问题, 高维 PDE | TensorFlow + Keras | scipy（`scipy.linalg.lstsq` 解最小二乘，`scipy.optimize.least_squares` 做非线性优化）；TensorFlow-Probability（生成随机系数）；前向自动微分用 TF 的 `ForwardAccumulator` |
| 时间积分 (2502.10949) | TensorFlow + Keras（推测） | tex 中实现细节被注释掉（`% comments on code...`），未明确写出 |
| PINN 误差分析 (2303.12245) | **PyTorch** | Adam + L-BFGS 优化器 |
| HLConcPINN (2406.06350) | **PyTorch** | Adam + L-BFGS（30000 iterations） |
| FCE (2403.06393) | 未明确提及 | — |
