在语音识别中，HMM 的每个状态都可以对应多帧观察值，这些观察值是特征序列，多样化且不限取值范围，因此，观察值概率的分布不是离散的，而是连续的，也适用高斯混合模型（Gaussian Mixture Model, GMM）来建模。

---

# 概率统计
连续变量的概率密度表示为$$p(x)=\int p(x,y)dy$$$$p(x,y)=p(y|x)p(x)$$
数学期望包括：
- 离散变量的期望$$E[f]=\sum_xp(x)f(x)$$
- 连续变量的期望$$E[f]=\int p(x)f(x)dx$$
- 条件期望$$E_x[f|y]=\sum_xp(x|y)f(x)$$
方差包括：
- 变量 $x$ 的方差$$\text{var}[x]=E[x^2]-E[x]^2$$
- 变量 $x$ 和 $y$ 的方差$$\text{cov}[x,y]=E_{x,y}[\{x-E[x]\}\{y-E[y]\}]=E_{x,y}[xy]-E[x]E[y]$$
- 两个矢量 $x$ 和 $y$ 的方差$$\text{cov}[x,y]=E_{x,y}[\{x-E[x]\}\{y^T-E[y^T]\}]=E_{x,y}[xy^T]-E[x]E[y^T]$$
概率统计需要用数学分布来表示，如伯努利分布，它是一种离散分布，有两种可能的结果。$$P_n=\begin{cases}1-p&n=0\\p&n=1\end{cases}$$二项式分布即重复 $n$ 次独立的伯努利试验。当试验次数 $n$ 为 $1$ 时，二项式分布就是伯努利分布。$$P(X=x)=f(x|n,p)=(^n_x)p^x(1-p)^{n-x}$$
---

# 高斯分布
自然界中有很多信号都符合高斯分布，又称正态分布，其函数表示如下：$$N(x,\mu,\sigma^2)=\frac1{(2\pi\sigma^2)^\frac12}\exp\{-\frac1{2\sigma^2}(x-\mu)^2\}$$
其中，$\mu$ 是均值，$\sigma^2$ 是方差。

高斯分布有对应的期望和方差，其数学计算过程如下：
- 期望$$E[x]=\int^\infty_{-\infty}N(x|\mu,\sigma^2)xdx=\mu$$$$E[x^2]=\int^\infty_{-\infty}N(x|\mu,\sigma^2)x^2dx=\mu^2+\sigma^2$$
- 方差$$\text{var}[x]=E[x^2]-E[x]^2=\sigma^2$$

针对二维向量 $(x_1,x_2)$，其联合概率计算如下：$$\begin{split}(x_1,x_2)
&=p(x_1)p(x_2)=N(x_1|\mu_1,\sigma_1^2)N(x_2|\mu_2,\sigma_2^2)\\
&=\frac1{(2\pi\sigma_1^2)^\frac12}\exp\{-\frac1{2\sigma_1^2}(x_1-\mu_1)^2\}+ \frac1{(2\pi\sigma_2^2)^\frac12}\exp\{-\frac1{2\sigma_2^2}(x_2-\mu_2)^2\}\\
&=\frac1{2\pi(\sigma_1^2\sigma_2^2)^\frac12}\exp(-\frac12(\frac{(x_1-\mu_1)^2}{\sigma^2_1}+\frac{(x_2-\mu_2)^2}{\sigma^2_2}))
\end{split}$$
进一步扩展到任意 $D$ 维，$x = \{x_1,x_2,...,x_D\}$：$$\begin{split} N(x|\mu,\sigma^2)
&=\frac1{(2\pi)^{\frac D2}}\frac1{|\Sigma|^\frac12}\exp\{-\frac12(x-\mu)\sum^{-1}(x-\mu)^T\}\\
&=\frac1{{2\pi}^\frac D2}\frac1{(\prod^D_{d=1}\sigma^2_d)^\frac12}\exp\{-\frac12\sum^D_{d=1}\frac{(x_d-\mu_d)^2}{\sigma^2_d}\}
\end{split}$$
其中，$D$ 维向量 $\mu$ 是均值，$\Sigma$ 是 $D\times D$ 协方差矩阵，$|\Sigma|$ 是 $\Sigma$ 的行列式。

给定 $N$ 个样本的观察值序列 $x=(x_1,x_2,...,x_N)$，可以计算出所有样本的联合概率：$$p(x|\mu,\sigma^2)=\prod^N_{n=1}N(x_n|\mu,\sigma^2)$$为了简化计算，便于见算计处理（防止精度溢出），一般采用对数概率（似然率）：$$\begin{split}\ln p(x|\mu,\sigma^2)
&=\sum^N_{n=1}\ln(N(x_n|\mu,\sigma^2))\\
&=\sum^N_{n=1}\ln(\frac1{(2\pi)^\frac D2}\frac1{|\Sigma|^\frac12}\exp\{-\frac12\sum^D_{d=1}\frac{(x_{nd}-\mu_d)^2}{\sigma^2_d}\})\\
&=\sum^N_{n=1}\frac12(-D\ln(2\pi)-\ln|\Sigma|-\sum^D_{d=1}\frac{(x_{nd}-\mu_d)^2}{\sigma^2_d}\})
\end{split}$$
高斯函数的均值和方差可以通过最大似然估计得到。用对数据概率 $p(x|\mu,\sigma^2)$ 对参数求偏导可以得到：$$\mu_{ML}=\frac1N\sum^N_{n=1}x_n$$$$\sigma^2_{ML}=\frac1N\sum^N_{n=1}(x_n-\mu_{ML})^2$$

---

# GMM
复杂的数据分布难以用一个高斯函数来表示，更多的是采用多个高斯函数组合来表示，从而形成高斯混合函数（GMM）。

K 阶 GMM 使用 K 个单高斯分布的线性组合来描述。令 $\lambda=\{\mu,\sum\}$，则 K 阶 GMM 的概率密度函数为 $$p(x|\lambda)=\sum^K_{k=1}p(x,k|\lambda)=\sum^K_{k=1}p(k)p(x,|k,\lambda)=\sum^K_{k=1}c_kN(x|\mu_k,\Sigma_k)$$其中，$c_k$ 是第 $k$ 个函数函数的权重， $\sum^K_{k=1}c_k=1$ 表示所有高斯函数的权重和为 $1$。第 $k$高斯函数可表示为 $$N(x|\mu_k,\Sigma_k)=\frac1{(2\pi)^{\frac D2}|\Sigma_k|^\frac12}\exp\{-\frac{(x-\mu_k)^T\Sigma^{-1}_k(x-\mu_k)}2\}$$因此，GMM 包含三种参数，分别为混合权重 $c_k$、均值 $\mu_k$ 和方差 $\Sigma_k$ 。这些参数需要训练，训练主要分为两步：
1. 初始化，即构造初始模型
2. 重估计，即通过 EM 迭代算法精细化初始模型

## 初始化
训练 GMM 的参数需要大量的数据（特征向量），这些数据一般没有分类标签，记不清楚其属于哪个高斯分布。因此得用聚类模型进行分类，常用的有 K-means、LBG等。其中，K-means 算法流程如下：
1. 初始化
	把训练数据（特征向量）平均分为 $K$ 组，计算每组高斯函数句的均值 $\mu_k$。
2. 最近邻分类
	针对每个特征向量 $x_n$，通过计算欧氏距离，寻找与之最靠近的第 $k$ 个高斯分布，并把该特征向量分配给这个高斯分布。
3. 更新中心点
	通过求平均值，更新每个分布的中心点，得到对应高斯函数的均值。
4. 迭代
	重复步骤 $2$ 和 $3$，直到整体的平均距离低于预设的阈值。

## 重估计
为了得到 GMM 的最大期望（EM）重估计公式，根据权重系数 $c_k$ 的限制，加入拉格朗日算子：$$\ln p(x|c,\mu,\Sigma)+\lambda(\sum^K_{k=1}c_k-1)=\sum^N_{n=1}\ln\{\sum^K_{k=1}c_kN(x_n|\mu_k,\Sigma_k)\}+\lambda(\sum^K_{k=1}c_k-1)$$
分别对 $\mu_k$、$\Sigma_k$、$c_k$ 求最大似然函数。

对 $\mu_k$ 求偏导并令倒数为 $0$，得到：$$-\sum^N_{n=1}\frac{c_kN(x_n|\mu_k,\Sigma_k)}{\sum^K_{k=1}c_kN(x_n|\mu_k,\Sigma_k)}\Sigma_k(x_n-\mu_k)=0$$
两边同除以 $\Sigma_k$，重新整理，得到：$$\mu_k=\frac{\sum^N_{n=1}\gamma(n,k)x_n}{\sum^N_{n=1}\gamma(n,k)}$$其中$$\gamma(n,k)=\frac{c_kN(x_n|\mu_k,\Sigma_k)}{\sum^K_{k=1}c_kN(x_n|\mu_k,\Sigma_k)}$$对 $\Sigma_k$ 求偏导并令导数为 $0$，得到：$$\Sigma_k=\frac{\sum^N_{n=1}\gamma(n,k)(x_n-\mu_k)(x_n-\mu_k)^T}{\sum^N_{n=1}\gamma(n,k)}$$对 $c_k$ 求偏导并令导数为 $0$，有：$$\sum^N_{n=1}\frac{N(x_n|\mu_k,\Sigma_k)}{\sum^K_{k=1}c_kN(x_n|\mu_k,\Sigma_k)}+\lambda=0$$得到：$$c_k=\frac{\sum^N_{n=1}\gamma(n,k)}{\sum^N_{n=1}\sum^K_{k=1}\gamma(n,k)}$$

采用 EM 算法，实现 GMM 参数重估计。具体算法如下：
1. 初始化
	定义高斯函数个数 $K$，采用 K-means 算法，对每个高斯函数参数 $c_k$、$\mu_k$、$\Sigma_k$ 进行初始化
2. E-step
	根据当前的 $c_k$、$\mu_k$、$\Sigma_k$ 计算后验概率 $\gamma(n,k)$。
3. M-step
	根据 E-step 中计算的 $\gamma(n,k)$ ，更新 $c_k$、$\mu_k$、$\Sigma_k$。
4. 计算对数似然函数$$\ln p(x|c,\mu,\Sigma)=\sum^N_{n=1}\ln\{\sum^K_{k=1}c_kN(x_n|\mu_k,\Sigma_k)\}$$
5. 迭代
	检查对数似然函数是否收敛，若不收敛，则返回步骤 $2$。

---

# GMM 与 HMM 的结合
在 GMM-HMM 中， HMM 模块负责建立状态之间的转移概率分布，而 GMM 模块则负责生成 HMM 的观察值概率。一个 GMM 负责表征一个状态，相邻的 GMM 之间相关性并不强，而每个 GMM 所生成的概率就是 HMM 中所需要的观察值概率。

HMM 的第 $j$ 个状态产生的观察值 $o_t$ 的概率表示如下：$$b_j(o_t)=\sum^K_{k=1}c_{jk}N(o_t|\mu_{jk},\Sigma_{jk})$$其中，$K$ 是 GMM 的阶数，即包含的高斯函数个数

因为 GMM 是统计模型，所以原则上，其参数量要与训练数据规模配套，即训练数据越多，对应的高斯函数也应该越多。大型的语音识别模型所用的 GMM 可达几万个，每个 GMM 都包含 $16$ 个 甚至 $32$ 个高斯函数。

---

# GMM-HMM 的训练
GMM-HMM 的观察值概率用 GMM 来表示，GMM 又包含多个高斯函数，即概率密度函数，因此需要估计的参数包括：
- 起始概率
- 转移概率
- 各个状态中不同概率密度函数的权重
- 各个状态中不同概率密度函数的均值和方差

结合 HMM 的前向-后向算法，定义统计量如下：$$\begin{split}\gamma^c_t(j,k)
&=[\frac{\alpha_t(j)\beta_t(j)}{\sum^N_{j=1}\alpha_t(j)\beta_t(j)}][\frac{c_{jk}N(o_t^c,\mu_{jk},\Sigma_{jk})}{\sum^K_{k=1}c_{jk}N(o_t^c,\mu_{jk},\Sigma_{jk})}]\\
&=\begin{cases}\frac1{P(O|\lambda)}\pi_j\beta_1(j)c_{jk}N(o_1^c,\mu_{jk},U_{jk})&t=1\\
\frac1{P(O|\lambda)}\sum^N_{i=1}\alpha_{t-1}(i)a_{ij}\beta_t(j)c_{jk}N(o_t^c,\mu_{jk},U_{jk})&t>1
\end{cases}
\end{split}$$
结合 HMM 和 GMM 的重估计公式，基于 ML 准则，GMM-HMM 参数的 EM 重估计公式为$$a_{ij}=\frac{\sum^C_{c=1}\sum^{T_c-1}_{t=1}\xi^c_t(i,j)}{\sum^C_{c=1}\sum^{T_c-1}_{t=1}\gamma^c_t(i)}$$$$c_{jk}=\frac{\sum^C_{c=1}\sum^{T_c}_{t=1}\gamma^c_t(j,k)}{\sum^K_{k=1}\sum^C_{c=1}\sum^{T_c}_{t=1}\gamma^c_t(j,k)}$$$$\mu_{jk}=\frac{\sum^C_{c=1}\sum^{T_c}_{t=1}\gamma^c_t(j,k)o^c_t}{\sum^C_{c=1}\sum^{T_c}_{t=1}\gamma^c_t(j,k)}$$$$\Sigma_{jk}=\frac{\sum^C_{c=1}\sum^{T_c}_{t=1}\gamma^c_t(j,k)(o^c_t-\mu_{jk})(o^c_t-\mu_{jk})^{'}}{\sum^C_{c=1}\sum^{T_c}_{t=1}\gamma^c_t(j,k)}$$其中，$C$ 为训练样本数。注意，每个特征 $o^c_t$ 都参与了每个高斯函数的均值和方差的计算，其比重由 $\gamma^c_t(j,k)$ 决定。

---

# 模型自适应
为了提高模型对于未知数据的准确率和提高鲁棒性，因此需要做自适应训练。

## MAP
MAP 算法本质就是重新训练一次，并且平衡原有模型参数和自适应数据的估计，其基本公式如下：$$\hat\lambda=\epsilon\lambda+(1+\epsilon)\lambda^{'},\quad0\leqslant\epsilon\leqslant1$$针对 GMM 的均值参数，基于原有模型均值 $\mu_0$ 和自适应数据 $x=\{x_1,x_2,...,x_T\}$，得到新的自适应均值 $\hat\mu$ 如下：$$\hat\mu=\frac{\tau\mu_0+\sum^T_{t=1}\gamma_tx_t}{\tau+\sum^T_{t=1}\gamma_t}$$其中，$\tau$ 控制原有模型均值和自适应数据之间的平衡（一般取值为 $0\leqslant\tau\leqslant20$）, $\gamma_t$ 是对应高斯函数在 $t$ 时刻的统计量。MAP 一般要求有较多的自适应数据。当自适应训练数据逐渐增多时，MAP 估计会逐步收敛为 ML 估计。

## MLLR
如果自适应数据较少，则无法为 GMM 的每个高斯函数均进行相应的自适应训练。一般方法是共享参数，这样仅用少量的数据即可得到较为满意的训练结果。MLLR 算法就是基于这种思想，将原始模型的参数线性变换后再进行识别的。其优点是，使用少量语音即可对所有模型进行自适应训练，只要得到线性变换矩阵即可。MLLR 对高斯分布均值的线性变换公式如下：$$\hat\mu=A\mu+b$$
假如声特征是 $d$ 维向量，则 $A$ 维 $d\times d$ 维矩阵，$b$ 为 $d$ 维向量。定义 $W=[bA]$，$\eta=[1\mu^T]^T$，则有：$$\hat\mu=W\eta$$其中，转换矩阵 $W$ 采用自适应数据估计得到，并且可以由所有或部分高斯函数共用。公用转换矩阵 $W$ 的高斯函数被划分为一个回归类。回归数目可通过聚类树训练得到，原则是每个回归都有足够的训练数据，否则就合并。

## fMLLR
如果高斯函数的均值和方差共用线性变换矩阵，则可用如下公式表示：$$\hat\mu=A^{'}\mu-b$$$$\hat\Sigma=A^{'}\Sigma {A^{'}}^T$$
这种 MLLR 被称为约束 MLLR （Constrained MLLR, cMLLR），其对应的对数似然为$$L=N(Ax_t+b;\mu,\Sigma)+\log(|A|),\quad A^{'}=A^{-1},b^{'}=Ab$$将 $A$ 直接与特征向量 $x_t$ 相乘，相当于对声学特征做转换，因此 cMLLR 又被称为特征空间 MLLR（feature space MLLR, fMLLR）。fMLLR 与标准的 MLLR 有类似的自适应效果，转换后的特征也可用于 GMM-HMM 以外的其他模型。

## SAT
在模型训练过程中，为每个说话人分别建立 MLLR（或 fMLLR）转换矩阵，则此为说话人自适应训练（Speaker-Adaptive Training, SAT）。SAT 增加了训练复杂度和存储空间，但可有效提升识别效果。
