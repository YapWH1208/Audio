同一句话在不一样的语速、语调等等原因在语音波形图的表达都有所不同。本章使用了隐马尔可夫模型（Hidden Markov Model， HMM）去解决这个问题。


# HMM的基本概念
## 马尔科夫链
在一个随机过程中，如果每个事件的发生概率仅依赖上一个事件，则称该过程为马尔科夫过程。

假设随机序列在任意时刻可以处于状态 $\{s_1,s_2,...,s_N\}$ ，且已有随机序列 $\{q_1,q_2,...,q_{t-1},q_t\}$ ，则产生新的事件 $q_{t+1}$ 的概率为$$P(q_{t+1}|q_t,q_{t-1},...,q_1)=P(q_{t+1}|q_t)$$换句话说，马尔科夫过程只能基于当前事件，预测下一个事件，而与之前或未来的事件无关。时间和事件都是离散的马尔科夫过程，称为马尔科夫链。

---

## 双重随机过程
HMM包含隐含状态（Hidden State），隐含状态和观察事件并不是一一对应的关系，因此，它所描述的问题比马尔科夫模型更复杂。

本质上，HMM描述了双重随机过程，包括：
- 马尔科夫链：状态转移的随机性
- 依存于状态的观察事件的随机性

---

## HMM定义
$N$：模型中的状态数目
$M$：每个状态可能输出的观察符号的数目
$A=\{a_{ij}\}$：状态转移概率分布
$B=\{b_j(k)\}$：观察符号概率分布
$\pi=\{\pi_i\}$：初始状态概率分布
以上参数可简化表示如下：$$\lambda=(\pi,A,B)$$
当给定模型 $\lambda=(\pi,A,B)$ 后，就可将该模型看成一个符号生成器，由它生成观察值序列 $O=o_1,o_2,...,o_T$。生成过程如下：
1. 初始状态概率分布为 $\pi$，随机选择一个初始状态 $q_1=s_i$。
2. 令 $t=1$。
3. 基于状态 $s_i$ 的符号概率分布为 $b_i(k)$，随机产生一个输出符号 $o_t=V_k$。
4. 基于状态 $s_i$ 的状态转移概率分布为 $a_{ij}$，随机转移至一个新的状态 $q_{t+1}=s_j$。
5. 令 $t=t+1$，若 $t\leqslant T$，则返回步骤 $3$，否则结束过程。

---

## HMM的三个基本问题
- 模型评估问题：如何求概率 $P(O|\lambda)$ ？
- 最佳路径问题：如何求隐含状态序列 $Q=q_1,q_2,...,q_T$ ？
- 模型训练问题：如何求模型参数 $\pi,A,B$ ？

### 模型评估问题
给定模型 $\lambda=(\pi,A,B)$ 以及观察值序列 $O=o_1,o_2,...,o_T$ 时，计算模型 $\lambda$ 对观察值序列 $O$ 的 $P(O|\lambda)$ 概率。

> 一般来说 $P(O|\lambda)$ 越高，模型的评价就越高


#### 穷举法：
1. 对长度为 $T$ 的观察值序列 $O$，找出所有可能产生该观察值序列 $O$ 的状态转移序列 $Q^j=q^j_1,q^j_2,q^j_3,...,q^j_T(j=1,2,...,J)$。
2. 分别计算 $Q^j$ 与观察值序列 $O$ 的联合概率 $P(O,Q^j|\lambda)$
3. 取各联合概率 $P(O,Q^j|\lambda)$ 的和，即$$P(O|\lambda)=\sum^J_{j=1}P(O,Q^j|\lambda)$$将 $P(O,Q^j|\lambda)$ 通过全概率公式（Law of Total Probability）进一步表示为$$P(O,Q^j|\lambda)=P(Q^j|\lambda)P(O|Q^j,\lambda)$$分别计算右边两项：$$P(Q^j|\lambda)=P(q^j_1)P(q^j_2|q^j_1)P(q^j_3|q^j_2)...P(q^j_T|q^j_{T-1})=a^j_{0,1}a^j_{1,2}a^j_{2,3}...a^j_{T-1,T}$$$$P(O|Q^j,\lambda)=P(o_1|q^j_1)P(o_2|q^j_2)...P(o_T|q^j_T)=b^j_1(o_1)b^j_2(o_2)b^j_3(o_3)...b^j_T(o_T)$$最后得到：$$P(O,Q^j|\lambda)=a^j_{0,1}b^j_1(o_1)a^j_{1,2}b^j_2(o_2)...a^j_{T-1,T}b^j_T(o_T)$$$$P(O|\lambda)=\sum^J_{j=1}P(O,Q^j|\lambda)=\sum^J_{j=1}\prod^T_{t=1}a^j_{t-1,t}b^j_t(o_t)$$


#### 前向-后向算法
##### 前向算法
前向算法按输出观察值序列的时间，从前向后递推计算输出概率。此算法用 $a_t(j)$ 表示已经输出的观察值 $o_1,o_2,...,o_t$，并且达到状态 $s_j$ 的概率为 $$\alpha_t(j)=P(o_1,o_2,...,o_t,q_t=s_j|\lambda)$$
具体算法步骤：
1. 初始化$$\alpha_1(i)=\pi_ib_i(o_1),\quad1\leqslant i\leqslant N$$
2. 迭代计算$$\alpha_{t+1}(j)=[\sum^N_{i=1}\alpha_t(i)a_{ij}]b_j(o_{t+1}),\quad1\leqslant t\leqslant T-1,1\leqslant j\leqslant N$$
3. 终止计算$$P(O|\lambda)=\sum^N_{i=1}\alpha_t(i)$$
> [【机器学习】动画讲解马尔科夫链（六）：前向算法_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1yX4y127ub/?vd_source=82cc9f8195ff57b14f4f1d470824ef31)


##### 后向算法
后向算法由后向前推算输出概率。如果输出结束时的状态为 $s_N$，时刻 $t$ 的状态为 $s_i$，则输出观察值序列 $o_t,o_{t+1},...,o_T$ 的概率 $\beta_t(i)$ 表示为$$\beta_t(i)=P(o_t,o_{t+1},...,o_T,q_t=s_i,q_T=s_N,\lambda)$$
具体算法步骤：
1. 初始化$$\beta_T(i)=1,\quad1\leqslant i\leqslant N$$
2. 迭代计算 $$\beta_t(i)=\sum^N_{j=1}a_{ij}b_j(o_{t+1})\beta_{t+1}(j),\quad1\leqslant t\leqslant T-1,1\leqslant j\leqslant N$$
3. HMM的前向、后向概率估计$$P(O|\lambda)=\sum^N_{i=1}\sum^N_{j=1}\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j),\quad1\leqslant t\leqslant T-1$$$$P(O|\lambda)=\sum^N_{i=1}\alpha_t(i)\beta_t(i)=\sum^N_{i=1}\alpha_T(i),\quad1\leqslant t\leqslant T-1$$结合前向-后向算法的定义，可用 $\alpha_t(i)$ 和 $\beta_t(i)$ 组合来计算 $P(O|\lambda)$，这样计算的好处是能够把不同时刻的中间结果保存下来，避免不必要的重复计算。



### 最佳路径问题
使用 Viterbi 算法求出概率最大的可能性得到最佳路径

Viterbi 算法描述如下：
1. 定义最佳状态序列 $Q^*=q_1^*,q_2^*,...,q_T^*$，$\varphi_t(j)$ 记录局部最佳状态序列。
2. 定义 $\delta_t(i)$ 为在截止时刻 $t$，依照转台转移序列 $1_1,q_2,...,q_t$，产生观察值序列 $o_1,o_2,...,o_t$ 的最大概率，且最终状态为 $s_i$。$$\delta_t(i)=\max_{q_1,q_2,...,q_{t-1}}P(q_1,q_2,...,q_{t-1},q_t=s_i|\lambda)$$
> $\varphi$ : 状态序列号，$\delta$ ：概率

Viterbi 算法步骤：
1. 初始化$$\delta_0(1)=1,\delta_0(j)=0\quad(j\neq1)$$$$\varphi_1(j)=q_1$$
2. 递推$$\delta_t(j)=b_j(o_t)\max_{1\leqslant i\leqslant N}\delta_{t-1}(i)a_{ij},\quad 1\leqslant t\leqslant T,1\leqslant i\leqslant N$$其中，$b$ 表示当前概率
$$\varphi_t(j)=\arg\max_{1\leqslant i\leqslant N}\delta_{t-1}(i)a_{ij}$$
3. 终止计算$$P_\max(S,O|\lambda)=\max_{1\leqslant i\leqslant N}\delta_T(i)$$$$\delta_T(N)=\arg\max_{1\leqslant i\leqslant N}\delta_{T-1}(i)a_{ij}$$
算法终止时，$\delta_t()$ 记录的数据便是最佳状态序列 $Q^*$。

### 模型训练问题
模型训练问题可定义为：给定一个观察值序列 $O=o_1,o_2,...,o_T$，确定 $\lambda=(\pi,A,B)$ 使得 $P(O|\lambda)$ 最大，用公式表示为 $$\bar \lambda=\arg\max_\lambda P(O|\lambda)$$
但没有一种方法能直接估计最佳的 $\lambda$。因此要寻找替代的方法，即根据观察值序列选取初模型 $\lambda=(\pi,A,B)$，然后求得一组新参数 $\bar\lambda=(\pi,\bar A,\bar B)$，保证有 $P=(O|\bar \lambda)>P(O|\lambda)$。重复这个过程，逐步改进模型参数，直到 $P(O|\bar\lambda)$ 收敛。

基于状态序列 $Q$，有概率公式：$$P(O,Q|\lambda)=\pi_{q_0}\prod^T_{t=1}a_{q_{t-1}q_t}b_{q_t}(o_t)$$
取对数得到：$$\log P(O,Q|\lambda)=\log\pi_{q_0}+\sum^T_{t=1}a_{q_{t-1}q_t}+\sum^T_{t=1}b_{q_t}(o_t)$$
根据 Bayes 公式和 Jensen 不等式，经过一系列转化，可定义辅助函数：$$\begin{split}
Q(\lambda,\bar\lambda)&=\sum_QP(O,Q|\lambda)\log P(O,Q|\lambda)\\
&\begin{align}=&\sum^N_{i=1}P(O,q_0=i|\bar\lambda)\log\pi_i+\\
&\sum^N_{i=1}\sum^N_{j=1}\sum^T_{t=1}P(O,q_{t=1}=i,q_t=j|\bar\lambda)\log a_{ij}+\\
&\sum^N_{i=1}\sum^T_{t=1}P(O|q_t=i|\bar\lambda)\log b_i(o_t)
\end{align}\end{split}$$
模型参数 $\pi_i,a_{ij},b_i(o_t)$ 均符合如下函数形式：$$F(x)=\sum_ic_i\log x_i$$
并有条件限制 $\sum_ix_i=1$。当$$x_i=\frac{c_i}{\sum_kc_k}$$时，该函数可获得全局最优解。因此我们可以得到模型参数的重新估计公式：$$a_{ij}=\frac{\sum^{T-1}_{t=1}\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{\sum^{T-1}_{t=1}\alpha_t(i)\beta_t(i)}=\frac{\sum^{T-1}_{t=1}\xi_t(i,j)}{\sum^{T-1}_{t=1}\gamma_t(j)}$$$$b_j(k)=\frac{\sum^T_{\begin{split}&t=1\\&\text{s.t.}o_t=v_k\end{split}}\alpha_t(i)\beta_t(i)}{\sum^T_{t=1}\alpha_t(i)\beta_t(i)}=\frac{\sum^T_{\begin{split}&t=1\\&\text{s.t.}o_t=v_k\end{split}}\gamma_t(j)}{\sum^T_{t=1}\gamma_t(j)}$$
其中，$a_{ij}$ 是状态 $i$ 到 $j$ 的转移概率，$b_j(k)$ 是状态 $j$ 产生观察值 $v_k$ 的概率。

我们可以直观地认为，$a_{ij}$ 的重估计公式是所有时刻从所有时刻从状态 $s_i$ 转移到状态 $s_j$ 的概率和除以所有时刻处于状态 $s_i$ 的概率和，$b_j(k)$ 的重估计公式是所有时刻状态 $s_j$ 产生观察值 $v_k$ 的概率和除以所有时刻处于状态 $s_j$ 的概率和。其中，$\xi_t(i,j)$ 为给定训练序列 $O$ 和模型 $\lambda$ 时，HMM 在 $t$ 时刻处于状态 $s_i$，在 $t+1$ 时刻处于状态 $s_j$ 的概率，即$$\xi_t(i,j)=P(q_t=s_i,q_{t+1}=s_j|O,\lambda)$$根据前向-后向算法，可推导出：$$\xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{\sum^N_{i=1}\sum^N_{j=1}\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O|\lambda)}$$进一步定义 $\gamma_t(i)$ 为在 $t$ 时刻时处于状态 $s_i$ 的概率：$$\gamma_t(i)=\sum^N_{j=1}\xi_t(i,j)=\sum^N_{j=1}\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O|\lambda)}=\frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}$$
HMM 的经典训练方法是基于最大似然准则，采用 Baum-Welch 算法，对每个模型的参数针对其所属的观察值序列进行优化训练，最大化模型对观察值的似然概率，训练过程不断迭代，直至所有模型的平均似然概率提升达到收敛

Baum-Welch算法步骤：
1. 初始化
	$\pi$ 和 $A$ 的初值对结果影响不大，只要满足约束条件，就可随机选取或均值选取。$B$ 的初值对参数重估计影响较大，选取算法复杂。
2. E-step
	基于模型参数，计算 $\gamma_t(i)$ 和 $\xi_t(i,j)$。
3. M-step
	由重估计公式重新计算 $a_{ij}$ 和 $b_j(k)$，最大化辅助函数。
4. 迭代
	重复第 $2$ 步的操作，直到 $a_{ij}$ 和 $b_j(k)$ 收敛为止，即 $P(O|\lambda)$ 趋于稳定，不再明显增大。

