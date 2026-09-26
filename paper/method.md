# TransSTID：基于可迁移时空节点身份的高效跨城市零样本交通预测

**English Title:** *Transferable Spatio-Temporal Node Identities for Efficient Zero-Shot Cross-City Traffic Forecasting*

# 1 引言

时空预测是众多城市与环境应用的底层技术，涵盖交通管理、空气质量监测等场景。但在实际应用中，带标签观测数据在不同区域分布极不均衡：部分城市积累了多年高密度传感器读数，而另一些场景——如新布设的传感网络、扩建通道，或是受隐私限制的部署环境——仅有少量甚至完全没有训练监督数据。这就催生了将预测模型从数据充足的源网络迁移至数据稀缺的目标网络的需求。然而绝大多数跨域方法仍假设可以获取目标域标签，或至少需要一轮微调阶段，这在真正的冷启动场景中并不现实。

因此，本文研究一个更严苛的问题：在某一道路网络上训练得到的模型，能否在完全未见过的新网络上完成预测，且不使用任何目标侧训练数据？我们将该场景定义为**零样本跨网络预测**。模型部署阶段，仅能获取目标网络拓扑结构与一段历史时间窗口（预测任务本身必需）；不允许使用目标标签，也不支持任何参数更新。核心难点不只是时间序列上的域偏移，更是表征层面的问题：许多性能优异的预测器会将空间身份编码为某种形式，一旦节点集合发生变化，该编码就不再有效。

一个极具代表性的例子是以 STID 为代表的、以多层感知机（MLP）为核心的预测模型家族。STID 通过附加可学习的节点嵌入，并使用共享 MLP 主干网络对动力学特征解码，无需复杂的时空图堆叠结构即可保留空间信息，兼具高预测精度与推理效率。但同样的设计，在零样本迁移场景下暴露出明显缺陷：这类嵌入本质是绑定训练传感器的索引表。只要网络发生变化（拓扑或节点规模改变），该索引表就必须重新训练，模型无法直接开箱即用在未见过的图上。换言之，让 STID 具备空间感知能力的机制，恰恰导致其不具备可迁移性。

基于上述观察，本文确立明确的研究目标：保留基于 MLP 的预测模型的高效性，同时将固定节点嵌入替换为具备迁移能力的节点身份表征。本文核心思路：空间身份不必以每个传感器 ID 对应独立自由向量的形式存储，而是可以由任意图上都可定义的物理量推导得到。从空间结构上，节点的特性由其在拓扑中的位置决定——本文称之为**图条件结构身份（Graph-Conditioned Structural Identity, GSI）**，而非它在训练集中的索引；从时间维度上，当前时间窗口本身就包含丰富的样本相关信号，可以构建无需训练的时间嵌入，能够适配测试输入且不需要目标域优化，符合零样本设定——即**窗口条件时间状态身份（Window-Conditioned Temporal-State Identity, WTI）**。

基于该思路，本文提出一套面向 MLP 型预测模型的可迁移嵌入框架。在空间维度，由拓扑导出的结构特征经过一个共享、置换等变的编码器映射为 GSI，替代不可迁移的嵌入查找表。在时间维度，WTI 以多参照时间统计（Multi-Reference Temporal Statistics, MRTS）从当前窗口构建节点级描述子，并使用源域训练好的映射头进行投影；该过程在测试时实时计算，全程不微调。将这两类身份表征与时滞特征拼接后，送入共享残差 MLP 完成解码，保持 STID 类模型的计算特性。为进一步提升模型对图规模、局部连接关系变化的鲁棒性，我们在源域训练阶段引入子图轮训策略，在零样本测试于陌生网络前，让模型接触不同尺度的诱导子图。

综上，本文主要贡献如下：

1. 指出固定节点嵌入是限制高效 MLP 预测模型实现零样本跨网络部署的核心瓶颈，并将可迁移节点嵌入作为对应的设计目标。
2. 通过互补的空间与时间身份表征实现该目标：由图结构推导得到的 GSI，以及基于当前时间窗口、无需训练的 WTI；二者共同实现节点级自适应，且不依赖绑定节点索引的参数。
3. 保留以 MLP 为主的预测主干，保证模型迁移能力的同时，不牺牲 MLP 模型在大规模预测任务中突出的高效性。
4. 将源域预训练与子图轮训相结合，在陌生网络上完成零样本预测实验，验证结构推导型嵌入和样本条件嵌入可以泛化到源域节点集合之外。

# 3 方法

## 3.1 问题定义

记交通传感器网络为图 $G=(V,E,A)$，其中 $V=\{v_i\}_{i=1}^{N}$ 为节点集合，$A\in\mathbb{R}^{N\times N}$ 为加权邻接矩阵。给定截止时刻 $t$ 的长度为 $T$ 的历史窗口
$$
X_t=[X_{t-T+1},\ldots,X_t]\in\mathbb{R}^{B\times T\times N\times C},
$$
时空预测模型学习映射
$$
\widehat Y_t=f_\theta(X_t;G),\qquad \widehat Y_t\in\mathbb{R}^{B\times H\times N\times C}, \tag{1}
$$
以估计未来 $H$ 个时间步的节点状态。

本文考虑跨城市严格零样本预测。源城市与目标城市分别记为 $G^S=(V^S,E^S,A^S)$ 和 $G^T=(V^T,E^T,A^T)$，二者不要求具有相同的节点数、节点对应关系或拓扑结构。所有面向当前预测任务的可训练参数仅由源域训练集 $\mathcal D^S$ 学习：
$$
\theta^\star
=
\underset{\theta}{\arg\min}\;
\mathbb{E}_{(X_t^S,Y_t^S)\sim\mathcal D^S}
\left[
\mathcal{L}\!\left(f_\theta(X_t^S;G^S),Y_t^S\right)
\right].
\tag{2}
$$

目标城市部署期间，$\theta^*$ 始终冻结。目标历史窗口 $X_t^T$ 仅作为式（1）所定义预测任务的即时输入。因此，目标域预测必须能够写成
$$
\widehat Y_t^T=f_{\theta^\star}(X_t^T;G^T),
\qquad \theta=\theta^\star\ \text{during deployment}.
\tag{3}
$$
即模型在未见图上构造所需表征后，通过一次冻结前向传播完成预测。

## 3.2 可迁移节点身份重参数化

STID 的出发点是，交通预测的困难不仅来自时空依赖建模，也来自样本的不可辨识性：即使两个节点具有相近的近期观测，它们仍可能因所处位置和时间条件不同而呈现不同的未来。为此，STID 为每个训练节点维护一个自由向量，并将其与历史表示和日历表示共同输入共享 MLP。令
$$
E_{\mathrm{id}}^S=[e_1^S,\ldots,e_{N_S}^S]^\top
\in\mathbb{R}^{N_S\times d},
\tag{4}
$$
则 $e_i^S$ 能够直接保存源城市节点 $i$ 的空间差异。这种直接性同时构成了其迁移边界：式（4）是定义在 $V^S$ 上的参数表，而不是作用于任意图的表示函数。当 $V^T\neq V^S$ 且缺少节点对应关系时，模型无法为 $v_i^T$ 确定应读取哪一个表项；在目标城市重新学习 $E_{\mathrm{id}}^T$ 又违背式（3）的零样本约束。简单删除节点身份虽然消除了域绑定，却会重新引入 STID 所揭示的节点不可辨识性。

TransSTID（Transferable Spatio-Temporal Identity Network）由此采用一个不同的视角：跨城市预测不应取消 node-wise identity，而应改变它的参数化方式。我们沿空间与时间两个互补方向重参数化节点身份：图条件结构身份（GSI）回答节点“位于何种结构位置”，窗口条件时间状态身份（WTI）回答节点“当前处于何种动态状态”。节点 $i$ 在时刻 $t$ 的可迁移身份写为
$$
R_{i,t}=
\left[
Z_i^{\mathrm{GSI}}
\Vert B_{i,t}^{\mathrm{WTI}}
\right].
\tag{5}
$$

两类身份均由跨节点共享的函数生成：
$$
Z^{\mathrm{GSI}}=\alpha_\theta\left(A,F(A)\right),
\qquad
B_t^{\mathrm{WTI}}=\beta_\theta\left(X_t,A\right).
\tag{6}
$$

其中，$Z_i^{\mathrm{GSI}}$ 刻画节点在当前图中的结构角色，$B_{i,t}^{\mathrm{WTI}}$ 描述节点在当前历史窗口中的时间状态。与式（4）不同，式（6）的可学习对象是跨节点共享的生成函数，而不是与节点数量共同增长的表项。只要目标拓扑与当前预测窗口可见，式（5）便能为任意目标节点构造空间—时间两部分身份。

这一重参数化保留了 STID 的两个关键性质。首先，模型仍为每个节点生成不同的条件表示，因此没有以参数共享换取节点同质化。其次，身份生成与预测解码彼此分离，复杂的图运算不需要成为预测骨干；后续预测仍由跨节点共享的残差 MLP 完成。TransSTID 的总体计算为
$$
\widehat Y_t=\mathcal P_\theta
\left(
\left[
E_t^{\mathrm{hist}}\Vert Z^{\mathrm{GSI}}\Vert
Q^{\mathrm{RASE}}\Vert B_t^{\mathrm{WTI}}\Vert E_t^{\mathrm{cal}}
\right]
\right),
\tag{7}
$$
其中 $\mathcal P_\theta$ 为 MLP 预测器，$E_t^{\mathrm{hist}}$ 表示近期历史，$E_t^{\mathrm{cal}}$ 提供共享日历坐标，$Q^{\mathrm{RASE}}$ 是道路属性语义嵌入（Road-Attribute Semantic Embedding, RASE），作为辅助静态上下文参与预测；GSI 与 WTI 共同构成式（5）的可迁移时空节点身份。

## 3.3 GSI：图条件结构身份

空间身份的关键并不是记住一个节点在数据矩阵中的行号，而是刻画它在路网中所处的位置。尽管不同城市没有共享节点，枢纽、走廊、连接节点和边缘节点等结构角色仍可由各自拓扑观察得到。GSI 据此将结构身份定义为当前图拓扑的条件函数。

给定加权邻接矩阵 $A$，我们首先构造对称归一化图拉普拉斯矩阵
$$
\widetilde A=\frac{A+A^\top}{2}+I,
\qquad
L_{\mathrm{sym}}=I-D^{-1/2}\widetilde A D^{-1/2},
\tag{8}
$$
其中 $D=\operatorname{diag}(\widetilde A\mathbf 1)$。取最小的 $d_f$ 个非平凡特征分量形成拓扑坐标
$$
F(A)=[u_2,\ldots,u_{d_f+1}]
\in\mathbb{R}^{N\times d_f}.
\tag{9}
$$

与节点 ID 表不同，$F(A)$ 在每个图上由其自身拓扑重新计算，因此对任意节点规模均有定义。式（9）只作为结构坐标的初始化；模型并不把某一个谱坐标直接解释为跨城市节点标识，而是通过共享的局部聚合器将其转换为结构身份。

具体地，令 $H^{(0)}=F(A)$。第 $l$ 层结构编码为
$$
H_i^{(l+1)}=
\sigma\!\left(
W_l
\left[
H_i^{(l)}\;
\Vert\;
\operatorname{MEAN}_{j\in\mathcal N(i)}H_j^{(l)}
\right]
\right),
\tag{10}
$$
最终得到
$$
Z^{\mathrm{GSI}}=\operatorname{LN}(H^{(L_s)})
\in\mathbb{R}^{N\times d_z}.
\tag{11}
$$

式（10）使用同一组参数处理所有节点，并以集合聚合而非邻居次序进行消息汇总。给定按同一置换变换的结构坐标，其编码满足
$$
\alpha_\theta(PAP^\top,PF)
=P\alpha_\theta(A,F),
\tag{12}
$$
说明节点重排只会相应重排输出，而不会改变编码规则。更关键的是，$\alpha_\theta$ 的形状与 $N$ 无关；同一结构编码器可直接应用于节点数量和连通模式不同的目标图。GSI 因而不试图在城市之间建立节点一一对应，而是学习“局部结构如何映射为预测所需的空间条件”。

对于固定路网，$Z^{\mathrm{GSI}}$ 与预测窗口无关，可在部署前计算并缓存。图消息传递由此仅承担身份生成，而不构成逐层传播的预测主干。

## 3.4 WTI：多参照窗口时间状态身份

结构角色无法完整决定节点在某一时刻的预测条件。同一个路段可能处于平稳、增长、衰减或异常波动等不同状态；同一绝对流量在不同城市和不同邻域中也可能具有不同含义。共享的日内和星期嵌入只能说明“当前是什么时间”，不能区分“各节点在这一时间处于什么状态”。我们因此提出 WTI，并以多参照时间统计（Multi-Reference Temporal Statistics, MRTS）作为其状态提取机制，为每个节点生成窗口条件的时间身份。

WTI 不从目标域历史档案学习节点类型，而只对当前可见窗口施加 MRTS 确定性统计算子。对节点 $i$，MRTS 从四个互补视角描述其当前时间状态：
$$
\begin{aligned}
s^{\mathrm{local}}_{i,t}&=\phi_{\mathrm{local}}(X_{t,:,i}),\\
s^{\mathrm{nei}}_{i,t}&=\phi_{\mathrm{nei}}(X_t,A)_i,\\
s^{\mathrm{net}}_{i,t}&=\phi_{\mathrm{net}}(X_t)_i,\\
s^{\mathrm{freq}}_{i,t}&=\phi_{\mathrm{freq}}(X_{t,:,i}).
\end{aligned}
\tag{13}
$$

其中，$s^{\mathrm{local}}$ 概括窗口末值、均值、标准差、趋势、极值范围以及末值相对窗口均值的偏移；$s^{\mathrm{nei}}$ 以当前邻域为参照，描述节点时间轨迹在水平、窗口均值和趋势上的相对偏离；$s^{\mathrm{net}}$ 使用图内分位秩刻画节点当前状态在整个网络中的相对位置；$s^{\mathrm{freq}}$ 则概括低频与高频能量、主导频率以及窗口不同分段之间的变化。邻域和全图在这里充当时间状态的归一化坐标，而不是被编码的空间对象；因此四组描述子分别刻画局部动态、邻域相对动态、网络相对动态和频率动态，均服务于时间身份。

不同统计组的量纲与语义并不相同。我们使用独立的源域共享投影将其映射到潜在空间，并在通道维组成 WTI：
$$
B_{i,t}^{\mathrm{WTI}}=\left[
\ell_{\mathrm{local}}(s^{\mathrm{local}}_{i,t})
\Vert\ell_{\mathrm{nei}}(s^{\mathrm{nei}}_{i,t})
\Vert\ell_{\mathrm{net}}(s^{\mathrm{net}}_{i,t})
\Vert\ell_{\mathrm{freq}}(s^{\mathrm{freq}}_{i,t})
\right].
\tag{14}
$$

需要区分的是，MRTS 在式（13）中的统计算子没有可学习参数，而生成 WTI 的式（14）投影由源域监督训练得到。因此 WTI 的准确性质是目标域免适配：部署时重新计算统计量，但所有投影保持冻结。它既不是目标域微调，也不利用窗口之外的目标流量建立长期行为原型。

WTI 与普通历史编码承担不同功能。历史编码保留模型预测未来所需的细粒度数值序列；WTI 将 MRTS 提取的尺度、趋势、相对位置和频率描述投影为低维条件，使共享 MLP 能够辨识节点当前所处的时间状态。由此，TransSTID 的时间侧贡献不是增加另一套日历查表，而是把节点特异的时间条件表示成可在未见城市即时生成的 node-wise embedding。

## 3.5 辅助语义上下文与身份条件化的 MLP 预测器

TransSTID 保留 STID 的轻量预测路径。首先，将每个节点最近 $T$ 个观测沿时间维映射为历史表示
$$
E_{i,t}^{\mathrm{hist}}
=\operatorname{Conv}_{1\times1}(X_{t,:,i})
\in\mathbb{R}^{d_x}.
\tag{15}
$$

日内时刻和星期索引分别映射为共享的日历嵌入 $e_t^{\mathrm{tod}}$ 与 $e_t^{\mathrm{dow}}$。节点的静态道路功能能够为预测提供拓扑之外的补充条件。为此，节点的去地理标识道路属性 $M_i$ 经冻结文本编码器 $g_{\mathrm{text}}$ 和源域学习的低维投影 $p_\theta$ 得到 RASE：
$$
Q_i^{\mathrm{RASE}}
=p_\theta\!\left(g_{\mathrm{text}}(M_i)\right).
\tag{16}
$$

RASE 只使用道路等级、方向、容量档和检测器类型等可迁移属性，不编码城市名称、绝对地理位置或节点编号。其作用不是定义另一类节点身份，也不是从文本直接预测流量，而是为 MLP 提供与流量无关的道路功能上下文。对于固定路网，RASE 可预先计算并缓存。

节点 $i$ 在时刻 $t$ 的完整输入为
$$
\begin{aligned}
H_{i,t}=\big[&E_{i,t}^{\mathrm{hist}}
\Vert Z_i^{\mathrm{GSI}}
\Vert e_t^{\mathrm{tod}}
\Vert e_t^{\mathrm{dow}}
\Vert Q_i^{\mathrm{RASE}}\\
&\Vert\ell_{\mathrm{local}}(s^{\mathrm{local}}_{i,t})
\Vert\ell_{\mathrm{nei}}(s^{\mathrm{nei}}_{i,t})
\Vert\ell_{\mathrm{net}}(s^{\mathrm{net}}_{i,t})
\Vert\ell_{\mathrm{freq}}(s^{\mathrm{freq}}_{i,t})\big].
\end{aligned}
\tag{17}
$$

所有节点共享同一个残差 MLP。令 $U^{(0)}=H$，则
$$
U^{(l+1)}=U^{(l)}+\mathcal M_l(U^{(l)}),
\qquad l=0,\ldots,L_p-1,
\tag{18}
$$
并由逐节点预测头输出
$$
\widehat Y_t=\operatorname{Head}(U^{(L_p)}).
\tag{19}
$$

本文采用三层残差 MLP。按照当前实现，式（17）各通道维度依次为
$$
(32,32,32,32,8,16,8,8,8),
$$
总融合维度为 176。模型不包含 $N\times d$ 的可学习参数表，因此参数量与城市节点数无关。

这一架构将空间建模限制在轻量的身份生成阶段。对固定目标图，GSI 与 RASE 可预先缓存；每个预测窗口只需计算历史映射、WTI 和共享 MLP。在固定 $T$ 与隐藏维度下，WTI 的统计提取和 MLP 解码均随 $N$ 线性增长，邻域统计随可见边数线性增长。因此，TransSTID 增加的是用于跨图迁移的紧凑身份生成器，同时保留以 MLP 为主体的预测计算形态。

## 3.6 源域学习与零样本部署

直接在单一完整源图上训练仍可能使表示生成器适应特定的图规模和连通模式。为使式（6）中的身份生成函数适应图规模与局部连通性的变化，我们在源域构造变规模诱导子图。对采样子图 $\widetilde G^S\sim\mathcal S(G^S)$，从其诱导邻接矩阵重新计算 $F(\widetilde A^S)$ 与 GSI，并由对应节点的当前窗口重新生成 WTI。训练目标为
$$
\mathcal L_{\mathrm{src}}
=\mathcal L_{\mathrm{full}}
+\lambda\,
\mathbb E_{\widetilde G^S\sim\mathcal S(G^S)}
\mathcal L_{\mathrm{sub}}(\widetilde G^S),
\tag{20}
$$
其中两项均采用 Huber 损失。全图项维持对源域整体交通结构的拟合，子图项则使同一组参数反复面对节点数和局部连接关系的变化。这里不存在目标域内循环或二阶元梯度；所有优化样本仍来自源城市。

目标域部署遵循式（3）。给定 $A^T$，模型首先计算并缓存
$$
Z^{T,\mathrm{GSI}}=\alpha_{\theta^*}(A^T,F(A^T)).
\tag{21}
$$

对每个待预测窗口，再计算
$$
B_t^{T,\mathrm{WTI}}=\beta_{\theta^*}(X_t^T,A^T),
\tag{22}
$$
并将式（21）与式（22）连同由目标道路属性直接计算的 RASE、历史编码和日历编码输入冻结的 MLP：
$$
\widehat Y_t^T =\mathcal P_{\theta^*} \left([E_t^{T,\mathrm{hist}}\Vert Z^{T,\mathrm{GSI}} \Vert Q^{T,\mathrm{RASE}}\Vert B_t^{T,\mathrm{WTI}} \Vert E_t^{T,\mathrm{cal}}]\right).
\tag{23}
$$

整个过程不建立目标节点参数表，不访问目标域长期训练流量，也不更新任何参数。由此，TransSTID 将 STID 中“与节点索引绑定的身份记忆”重参数化为由 GSI、RASE 与 WTI 共同构成的可迁移身份，使节点级可辨识性与跨城市零样本部署在同一个 MLP 框架中成立。