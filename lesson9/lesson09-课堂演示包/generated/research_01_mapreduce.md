# 独立研究一：MapReduce 并行 KMeans

> 主题：如何用 MapReduce 把 KMeans 的"分配—更新"迭代拆成可并行的两阶段，
> 以及它在银行大规模客户分群中的意义。
> 本文为方法说明，不涉及真实银行数据；文中的数值仅作计算示例。

## 一、MapReduce 的基本思想

MapReduce 是 Google 提出的一种分布式计算模型（Dean & Ghemawat, 2004），核心是**把大任务拆小、算完再合并**：

1. **Map（映射）**：每条输入记录独立地产生若干键值对 `(key, value)`，
   不同记录之间互不依赖，因此可以分到成百上千台机器上并行处理。
2. **Shuffle / Sort（洗牌）**：框架把相同 key 的键值对汇聚到同一个计算节点。
3. **Reduce（归约）**：对同一个 key 的所有 value 做汇总（求和、计数、取最大等），
   输出最终结果。

关键性质：**Map 只看单条记录，Reduce 只看同一个 key 的汇总**。
只要算法能写成"局部统计 + 按 key 合并"的形式，就能直接并行化。
KMeans 的每次迭代恰好满足这个条件——因为簇中心只是"簇内样本的均值"。

## 二、Map 阶段：计算各簇的样本数量与特征总和

设第 t 轮迭代已有 K 个簇中心 `μ₁, μ₂, …, μ_K`（这一轮作为**广播变量**发到所有 Map 节点），
第 i 个客户的特征向量为 `xᵢ`（如标准化后的 age、income、balance、product_count）。

**Map 的输入**：一条客户记录 `xᵢ`
**Map 的处理**：

1. 计算 `xᵢ` 到 K 个中心的欧氏距离，取最近的中心 `c = argmin_k ‖xᵢ − μ_k‖²`；
2. 输出键值对：

```text
key   = c                      （该客户被分配到的簇编号）
value = (1, xᵢ₁, xᵢ₂, …, xᵢ_d) （计数 1 + 该客户的 d 维特征向量）
```

用 Python 伪代码表示（Hadoop Streaming 风格，逐行读入）：

```python
# 每个 Map 节点都持有当前轮的 K 个中心（广播）
for line in input_records:                 # 一条客户记录
    x = parse_features(line)               # 标准化后的特征向量
    c = argmin_k(euclidean(x, centers[k]) for k in range(K))
    emit(c, (1, x))                        # key=簇号, value=(1, 特征向量)
```

要点：
- 每个 Map 任务只处理自己分片上的客户，**不需要看到全量数据**；
- 输出的 value 已经是"这一条记录的局部统计"（计数 1 与特征本身）；
- 计算量与客户数成正比、可线性扩展。

## 三、Reduce 阶段：更新各簇中心

Shuffle 之后，同一个簇号 c 的所有 `(1, xᵢ)` 会到达同一个 Reduce 任务。

**Reduce 的输入**：`key = c`，`values = [(1, x₁), (1, x₂), …, (1, x_m)]`
**Reduce 的处理**：

1. 计数：`n_c = Σ 1`（簇 c 的样本数）；
2. 特征求和：`S_c = Σ xᵢ`（按维度分别累加，得到 d 维向量）；
3. 新中心：`μ_c' = S_c / n_c`（逐维取平均）。

```python
# 对每个簇号 c
n_c = 0
S_c = zeros(d)
for (cnt, x) in values:
    n_c += cnt
    S_c += x
new_center[c] = S_c / n_c
```

这与单机 KMeans 的更新公式完全一致：

```
μ_c = (1 / |C_c|) · Σ_{x ∈ C_c} x
```

差别只在于：单机是遍历内存里的数组，MapReduce 是"Map 局部累加 → Shuffle 按簇聚合 → Reduce 求平均"。

**迭代控制**：Reduce 输出的新中心写回分布式存储，作为下一轮 Map 的广播输入；
重复"Map → Reduce"直到中心不再明显移动（或达到最大迭代次数）。
因此 KMeans 在 MapReduce 上是一个**外层循环包住 MapReduce 作业**的结构，
而不是一次 MapReduce 就结束。

## 四、它如何帮助处理大规模客户数据

| 维度 | 单机 KMeans | MapReduce KMeans |
|---|---|---|
| 数据规模 | 受单机内存/磁盘限制 | 数据分片存放在 HDFS 等分布式存储，可到 TB 级 |
| 计算方式 | 单进程遍历全部样本 | 每个分片一个 Map 任务，并行计算 |
| 通信量 | 无 | 每轮只传"簇号 + 特征"的键值对，Reduce 输出仅 K 个中心 |
| 扩展性 | 加内存/换机器 | 增加节点即可近似线性提速 |
| 失败恢复 | 单点故障即中断 | 框架自动重跑失败的分片任务 |

具体收益：

1. **每轮的通信开销很小**。真正的数据始终留在各节点本地，只有汇总结果（K 个中心、K 个计数）
   需要在网络上传输，避免"把上亿条客户记录搬到一台机器"。
2. **Map 完全无状态**，可以放心扩到几千个分片；数据量翻倍时增加节点即可。
3. **与现有大数据栈天然契合**：客户宽表通常就存放在 HDFS/对象存储中，
   MapReduce/Spark 可直接读取，不必先导入单机内存。
4. **注意初始化**：随机选 K 个初始中心在分布式下也有开销，实践中常用 k-means++ 的并行变体
   （如 Spark MLlib 的 k-means||），用少量采样轮次得到较好的初始中心。

## 五、结合银行客户分群举例

场景：某银行有 **8000 万**零售客户，字段包括年龄、年收入、账户日均余额、持有产品数、
近 12 个月交易笔数等（本文不涉及真实数据，仅描述流程）。

**离线批量分群流程**

1. **数据准备**：客户特征宽表按客户号分区存放在 HDFS，例如 2000 个分片、每片约 4 万客户。
2. **初始化**：用 k-means|| 在少量样本上选出 K=6 个初始中心（如"年轻低净值""年轻多产品"
   "中年稳健""中年高收入""高净值""老年保守"等业务可解释的群体雏形），广播到所有 Map 节点。
3. **第 1 轮 Map**：每个分片独立计算本片客户到 6 个中心的距离并打标签，
   输出 `(簇号, (1, 特征向量))`。2000 个 Map 并行，单节点只处理 4 万条记录。
4. **Shuffle + Reduce**：按簇号汇总，6 个 Reduce 各算一个簇的 `n_c` 与 `S_c`，
   得到新的 6 个中心（如"高净值"簇平均年龄 47 岁、平均余额 48 万元等）。
5. **迭代**：把新中心写回，重复第 3–4 步，通常 10~20 轮内收敛（中心移动量小于阈值）。
6. **产出**：每个客户号对应的簇标签写回客户宽表，供下游使用。

**下游业务用法**（示例，非真实业务指标）

- 高净值簇 → 推送专属客户经理名单、财富管理产品货架；
- 年轻多产品簇 → 信用卡分期、消费贷的定向名单；
- 中年稳健簇 → 存款留存与稳健理财转化活动；
- 低净值潜力簇 → 低成本数字化触达，控制营销成本。

**为什么必须并行**：8000 万客户 × 6 个中心 × 20 轮的距离计算量在单机上不可接受；
而在 MapReduce 下，每轮只是"2000 个分片各算一遍 + 6 个 Reduce 求平均"，
增加机器即可缩短时间，且客户数据无需离开存储集群。

## 六、局限与补充

- MapReduce 每轮都要落盘，迭代型算法（KMeans）会有较多 I/O；
  实践中更常用 **Spark MLlib**（内存计算、DAG 优化），思路仍是本文的"局部统计 + 全局平均"。
- 分布式 KMeans 的结果仍依赖初始中心，需要固定随机种子或使用 k-means|| 以保证可复现。
- "簇号"只是并行计算的中间键，**不代表业务含义**；业务命名必须依据各簇的实际画像统计。

## 七、参考资料

1. Dean, J., & Ghemawat, S. (2004). *MapReduce: Simplified Data Processing on Large Clusters.*
   OSDI 2004. https://research.google/pubs/pub62/
2. Zhao, W., Ma, H., & He, Q. (2009). *Parallel K-Means Clustering Based on MapReduce.*
   CloudCom 2009, LNCS 5931, pp. 674–679.
3. MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.*
   Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, pp. 281–297.
4. Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding.*
   SODA 2007.
5. Apache Spark 官方文档：Clustering（K-means / k-means||）—
   https://spark.apache.org/docs/latest/ml-clustering.html
6. Apache Hadoop 官方文档：MapReduce Tutorial —
   https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html
7. scikit-learn 官方文档：KMeans —
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
