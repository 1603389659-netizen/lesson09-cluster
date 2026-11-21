# 第 9 课 AI 使用记录（聚类与客户分群）

> 记录原则：只记录本次实际使用过的提示词与真实运行结果；未发生的操作不补写，
> AI 生成的内容与本人验证的内容分开标注。
> 数据说明：`data/mall_customers_sim.csv`、`data/mall_customers_yuan.csv` 均为**课程提供的模拟数据**，
> 不是真实银行业务数据，报告与结论均按此口径表述。

## 0. 环境与目录

- 工作目录：`lesson09-课堂演示包/`
- Python：`C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe`
- 依赖：pandas 3.1.1、matplotlib 3.11.2、scikit-learn 1.9.1、numpy 2.5.3
- 运行方式：所有脚本在 `lesson09-课堂演示包/` 下执行（脚本内用相对路径 `data/`、`outputs/`）
- 原始 CSV 全程只读，未做任何修改

## 1. 初始起点指令（项目检查 + 数据初检）

### 1.1 实际使用提示词（原文）

**第 1 次：项目检查（只检查，不改文件、不建模）**

```text
现在开始完成《人工智能程序设计与实践》第9次课聚类客户分群作业。

我已经打开老师提供的 Lesson 09 课堂演示包。

请先检查项目，不要修改任何文件，也不要立即开始建模。

一、检查目录：
1. 列出项目全部目录和文件。
2. 检查 data 文件夹中的全部 CSV。
3. 找到 mall_customers_sim.csv 和 mall_customers_yuan.csv。

二、检查数据：
1. 分别读取两个 CSV。
2. 输出行数、列数、字段名称。
3. 输出各列数据类型、缺失值和重复值。
4. 检查客户编号是否重复。
5. 输出数值列的均值、最大值、最小值。
6. 比较两个数据文件的客户记录是否对应。
7. 重点检查 income、balance 的单位和数值范围。

三、检查课程资料：
1. 检查 prompts 文件夹中的全部提示词。
2. 检查 generated 文件夹中的现有 Python 文件。
3. 检查 README.md。

四、输出检查结果：
说明哪些文件已经存在、哪些任务尚未完成。

注意：
- 不修改原始 CSV。
- 不编造数据。
- 不把教学模拟数据描述成真实银行业务数据。
- 暂时不运行聚类。
- 检查完成后等待我确认。
```

**第 2 次：数据初检与可视化**

```text
现在完成第9课客户分群的第一项：数据初检与可视化。

使用 data/mall_customers_sim.csv。

要求：
1. 读取数据并检查300位客户的数据完整性。
2. 打印行列数、缺失值、数值列基本统计。
3. customer_id 只作为客户编号，不能进入模型。
4. 检查 income、balance、age、product_count 四个字段。
5. 画收入-余额散点图。
6. 横轴 income，纵轴 balance，点大小可以映射 age。
7. 设置中文字体、中文标题和坐标轴。
8. 保存图片为 outputs/01_raw_scatter.png。

生成代码：
generated/01_load_and_viz.py

运行代码，打印实际结果。
不要修改原始数据。
```

### 1.2 修改前的情况

- `generated/01_load_and_viz.py` 已存在，但字体为 macOS 专用（Songti SC / STHeiti / PingFang SC），
  在 Windows 上中文会显示为方框；且缺少 customer_id 重复检查、四个建模字段的专门检查。
- 环境未安装 pandas / matplotlib / scikit-learn，默认 `python` 不可用。

### 1.3 修改后结果（真实运行输出）

- 中文字体：`Microsoft YaHei（C:\Windows\Fonts\msyh.ttc）`
- 行数 300、列数 7：`customer_id, age, income, balance, product_count, gender, city`
- 缺失值：全 0；完全重复行：0
- `customer_id` 唯一值 300，重复 0（不进入模型）
- 统计：age 均值 32.29（18~63）、income 均值 17.32（2.15~53.50 万元）、
  balance 均值 13.81（**-0.73**~72.01 万元）、product_count 均值 2.05（0~8）
- 输出图：`outputs/01_raw_scatter.png`（已生成，160 KB 左右）

## 2. KMeans 聚类指令

### 2.1 实际使用提示词（原文）

```text
现在完成第9课客户聚类任务。

使用 data/mall_customers_sim.csv。

采用以下四个实际存在的数值特征：
age、income、balance、product_count。

要求：

1. customer_id 绝不能进入模型。
2. 使用 StandardScaler 对四个特征进行标准化。
3. 使用 KMeans，参数明确写为：
   KMeans(n_clusters=4, random_state=42, n_init=10)
4. 聚类完成后把簇编号写回数据框。
5. 输出四个簇各自的人数和占比。
6. 检查四簇人数之和是否等于300。
7. 输出四簇特征均值。
8. 生成按簇着色的二维散点图，标出簇中心。
9. 保存聚类结果 CSV。

生成：
generated/02_kmeans_cluster.py
generated/clustered_customers.csv
generated/cluster_profile.csv
outputs/02_clustered_scatter.png

运行程序并打印完整结果。

报告中注明：使用的是课程提供的模拟客户数据，且本次特征选择与实践指南所列字段存在差异。不要伪造字段。
```

### 2.2 修改前的情况

- 原 `generated/02_kmeans_cluster.py` 是课程演示版：k=3、只用 income + balance、读 `mall_customers_yuan.csv`，
  产出 `outputs/04_compare_scale.png`，与本作业要求不一致。

### 2.3 修改后结果（真实运行输出）

- 建模特征：`age, income, balance, product_count`；`customer_id` 已排除
- 标准化后四列均值≈0.0000、标准差≈1.0017
- `KMeans(n_clusters=4, random_state=42, n_init=10)`，迭代 5 次，inertia = 323.2085
- 簇人数：50 / 54 / 102 / 94，合计 300（校验通过）
- 各簇均值：
  - 簇 0：age 47.100、income 41.584、balance 48.350、product_count 1.960
  - 簇 1：age 23.926、income 9.317、balance 5.121、product_count 3.926
  - 簇 2：age 39.549、income 19.290、balance 12.226、product_count 1.951
  - 簇 3：age 21.330、income 6.879、balance 2.155、product_count 1.128
- 输出：`generated/clustered_customers.csv`（300×8）、`generated/cluster_profile.csv`、`outputs/02_clustered_scatter.png`

## 3. 标准化修正指令（情况说明）

**如实说明：本次作业中没有单独发出过"标准化修正"提示词。**

原因：第 2 项任务在提出时就已明确写入"使用 StandardScaler 对四个特征进行标准化"，
因此不存在"先不标准化 → 结果出问题 → 再发修正指令"的过程。

与标准化修正最接近的真实操作是第 9 节"量纲陷阱实验"（补充项）：
用 `mall_customers_yuan.csv` 现场演示不标准化时余额以"元"计、贡献 100% 的距离，
再用 StandardScaler 拉回 50%/50%，ARI 由差异实际体现。该实验的提示词原文见 9.1。

> 课程演示包里的 `prompts/02_修正指令A_量纲陷阱.md` 是老师的课堂演示指令，
> 本次作业**没有整段照搬**，故不作为"使用过的提示词"记录。

## 4. 肘部法指令

### 4.1 实际使用提示词（原文）

```text
现在完成第9课第二项：使用肘部法选择最佳K值。

基于前一步相同的数据和四个特征：
age、income、balance、product_count。

要求：

1. 先使用StandardScaler标准化。
2. K从1到9逐一测试。
3. 每次使用KMeans。
4. 参数固定：
   random_state=42
   n_init=10
5. 记录每个K对应的inertia_。
6. 打印每个K的数值表。
7. 绘制K与SSE的折线图。
8. 标注肘部拐点。
9. 解释选择的K值及理由。

生成：
generated/03_choose_k.py
generated/elbow_values.csv
outputs/03_elbow.png

实际运行代码，不允许编造结果。
```

### 4.2 修改前 / 修改后

- **修改前**：首次运行后，脚本自动判定肘部为 K=3，但结论文字里硬写了"取 K=4"，
  两者矛盾，属于措辞与实际结果不一致。
- **修改后**：结论改为完全由实际数值生成，同时标出"几何肘部 K=3"与"建议 K=4"，
  并说明曲线不尖锐、肘部区间为 K=3~4。

### 4.3 真实运行结果

| K | SSE | 下降量 |
|---|---|---|
| 1 | 1200.0000 | — |
| 2 | 642.8650 | 557.1350 |
| 3 | 436.8879 | 205.9771 |
| 4 | 323.2085 | 113.6794 |
| 5 | 262.5755 | 60.6330 |
| 6 | 236.2191 | 26.3564 |
| 7 | 206.6604 | 29.5587 |
| 8 | 188.0984 | 18.5620 |
| 9 | 169.7410 | 18.3574 |

- 几何肘部（点到首尾连线距离最大）：K=3
- K=4→5 降幅仅为 K=3→4 降幅的 53.3%，K=5 之后降幅降到 30 以下
- 最终建议 K=4，并注明几何法给出 K=3 的分歧
- 输出：`generated/elbow_values.csv`、`outputs/03_elbow.png`

## 5. 客户画像指令

### 5.1 实际使用提示词（原文）

```text
现在完成第9课第三项：客户分群画像。

基于已经完成的KMeans结果：

1. 使用groupby('cluster')统计每簇：
   - 客户人数
   - 人数占比
   - 平均年龄
   - 平均收入
   - 平均余额
   - 平均持有产品数

2. 生成客户画像统计表。
3. 绘制分组柱状图或雷达图。
4. 根据每簇实际特征，为四个客群起中文业务名称。
5. 说明每个客群的特征和银行业务建议。
6. 不能仅用簇编号作为结论。
7. 不允许编造客户画像或业务数据。

生成：
generated/04_customer_profile.py
generated/cluster_profile.csv
generated/mall_report.md
outputs/04_customer_profile.png

运行代码并打印四个客群的画像。
```

### 5.2 修改前 / 修改后

- **修改前**：首次生成的柱状图为空图（y 轴刻度 ±0.04），原因是取列时错位——
  `profile.columns[2:]` 带上了"人数占比(%)"列，与特征均值列不匹配，Z 值全为 NaN。
- **修改后**：显式按四个特征列构造 Z 值，重新运行后四簇 × 四特征柱形正常显示。

### 5.3 真实结果

| 簇 | 客群名称 | 人数 | 占比 | 平均年龄 | 平均收入 | 平均余额 | 产品数 |
|---|---|---|---|---|---|---|---|
| 0 | 高净值中年客群 | 50 | 16.67% | 47.10 | 41.58 | 48.35 | 1.96 |
| 1 | 年轻多产品成长客群 | 54 | 18.00% | 23.93 | 9.32 | 5.12 | 3.93 |
| 2 | 中年稳健理财客群 | 102 | 34.00% | 39.55 | 19.29 | 12.23 | 1.95 |
| 3 | 年轻低净值潜力客群 | 94 | 31.33% | 21.33 | 6.88 | 2.15 | 1.13 |

- 命名由各簇均值与全局均值（年龄 32.29、收入 17.32、余额 13.81、产品数 2.05）比较得出，
  不是按簇编号硬编码。
- 输出：`generated/cluster_profile.csv`、`generated/mall_report.md`、`outputs/04_customer_profile.png`

## 6. 更换特征集指令

### 6.1 实际使用提示词（原文）

```text
现在完成第9课自主实践第一项：更换特征集。

使用 data/mall_customers_sim.csv。

进行两组实验：

第一组：
income、balance。

第二组：
income、balance、age。

要求：

1. 两组分别使用StandardScaler。
2. 两组均使用KMeans，n_clusters=4、random_state=42、n_init=10。
3. 分别输出四簇人数。
4. 比较两组客户画像。
5. 比较加入age后，客户分群发生了什么变化。
6. 分析特征选择对客户分群业务含义的影响。
7. 不允许修改原始数据。

生成：
generated/05_feature_comparison.py
generated/feature_comparison.csv
generated/feature_comparison.md
outputs/05_feature_comparison.png

运行并输出真实结果。
```

### 6.2 修改前 / 修改后

- **修改前**：报告里对"变化程度"使用了人工阈值措辞（ARI<0.6 等），
  与实际交叉表不符——真实情况是 123 人的簇被按年龄拆成 54/72 两块、
  两个高净值小簇合并为 50 人。
- **修改后**：改为按交叉表自动识别"被拆分 / 被合并"的簇，并给出具体人数；
  年龄极差接近时如实写"几乎不变"，不夸大。

### 6.3 真实结果

| 实验 | 各簇人数 | inertia |
|---|---|---|
| income + balance | 126 / 123 / 26 / 25 | 45.1693 |
| income + balance + age | 121 / 57 / 72 / 50 | 104.9310 |

- ARI = 0.7270
- 实验一簇 1（123 人）→ 实验二拆为 54 人（平均 29.63 岁）与 69 人（平均 43.15 岁），最大一块仅占 56.1%
- 实验一簇 2（26 人）+ 簇 3（25 人）→ 实验二合并为 50 人（平均 47.10 岁、余额 48.35 万元）
- 各簇平均年龄极差：26.09 岁 → 26.15 岁（几乎不变）
- 输出：`generated/feature_comparison.csv`、`generated/feature_comparison.md`、`outputs/05_feature_comparison.png`

## 7. 稳定性检验指令

### 7.1 实际使用提示词（原文）

```text
现在完成第9课自主实践第二项：聚类稳定性检验。

使用同一份课程模拟数据和同一组基准特征：
income、balance。

要求：

1. StandardScaler标准化。
2. 分别使用random_state=7、42、2026。
3. 每次使用KMeans(n_clusters=4, n_init=10)。
4. 输出三次聚类各簇人数。
5. 使用ARI（调整兰德指数）比较不同随机种子的聚类结果。
6. 分析三次聚类是否稳定。
7. 绘制对比图。

生成：
generated/06_stability_test.py
generated/stability_results.csv
generated/stability_analysis.md
outputs/06_stability.png

注意：簇编号可能发生交换，不能直接根据簇编号判断结果是否变化。
```

### 7.2 真实结果

| random_state | 簇 0 | 簇 1 | 簇 2 | 簇 3 | inertia_ | n_iter_ |
|---|---|---|---|---|---|---|
| 7 | 126 | 123 | 26 | 25 | 45.1693 | 5 |
| 42 | 126 | 123 | 26 | 25 | 45.1693 | 4 |
| 2026 | 126 | 123 | 26 | 25 | 45.1693 | 6 |

- 两两 ARI 全部 = 1.0000（用重编号后的标签复算仍为 1.0000）
- 判定：完全稳定；三次 inertia 完全相同，仅迭代次数不同（初始中心不同）
- 输出：`generated/stability_results.csv`、`generated/stability_analysis.md`、`outputs/06_stability.png`

### 7.3 修改情况

- 本步一次运行即得到结果，没有因错误返工；
  仅删除了脚本中两行未使用的占位变量（不影响结果）。

## 8. 二维散点图指令

### 8.1 实际使用提示词（原文）

```text
现在完成第9课自主实践第三项：二维散点图。

使用已经完成的四簇聚类结果。

要求：

1. 横轴income，纵轴balance。
2. 每个客户用一个散点表示。
3. 根据cluster编号使用不同颜色。
4. 加入中文图例。
5. 标注四个簇的名称和人数。
6. 不要重新编造聚类结果。
7. 使用真实运行得到的cluster列。

生成：
generated/07_cluster_scatter.py
outputs/07_cluster_scatter.png

运行并检查图片。
```

### 8.2 修改前 / 修改后

- **修改前**：四个簇中心标注框集中在原点附近，簇 1 的标注框压住了部分橙色散点。
- **修改后**：为四个簇分别设置标注偏移，簇 1 标注移到左侧空白区、簇 3 放到下方，
  四个标注框互不重叠。
- 数据来源：读取 `generated/clustered_customers.csv` 的真实 `cluster` 列，**未重新聚类**。

### 8.3 真实结果

- 300 个点，横轴 income（万元）、纵轴 balance（万元）
- 图例（簇编号 · 客群名称 · 人数）：簇 0 高净值中年客群 50；簇 1 年轻多产品成长客群 54；
  簇 2 中年稳健理财客群 102；簇 3 年轻低净值潜力客群 94
- 输出：`outputs/07_cluster_scatter.png`

## 9. 补充：量纲陷阱实验（真实完成，未列入上述 8 项）

### 9.1 实际使用提示词（原文）

```text
现在完成第9课量纲陷阱实验。

使用：
data/mall_customers_sim.csv
data/mall_customers_yuan.csv

要求：

1. 检查两份数据是否对应同一批客户。
2. 比较income和balance的单位。
3. 使用原始数据演示不标准化时，金额量纲对欧氏距离的影响。
4. 使用StandardScaler重新聚类。
5. 比较标准化前后的聚类结果。
6. 不修改原始CSV。
7. 输出两种情况下的簇人数和画像。
8. 绘制原始坐标与标准化坐标的对比图。

生成：
generated/08_scale_experiment.py
generated/scale_comparison.csv
outputs/08_scale_comparison.png

注意：标准化不是修改客户，而是改变距离计算时的度量衡。
```

### 9.2 真实结果

- 两份文件为同一批 300 位客户，只有 balance 单位不同（万元 vs 元，放大 10000 倍），
  `balance_yuan = balance_sim × 10000` 验证为 True。
- 不标准化：平方欧氏距离贡献 income 约 0.00%、balance 100.00%
- 标准化后：income 50.00%、balance 50.00%
- 簇人数：原始量纲 `[20, 106, 34, 140]` → 标准化 `[25, 126, 123, 26]`，ARI = 0.7627
- 输出：`generated/scale_comparison.csv`、`outputs/08_scale_comparison.png`

## 10. 本人验证过程与截图文件名

### 10.1 实际执行与核验方式

1. 每个脚本均在 `lesson09-课堂演示包/` 目录下用上述 Python 3.12 解释器实际运行，
   终端输出被逐段检查（人数合计、ARI、inertia 等数值）。
2. 每张 PNG 生成后都用图片查看实际打开确认：
   - 中文是否正常渲染（无方框乱码）
   - 四簇颜色、图例人数是否与终端输出一致
   - 簇中心星形、标注框是否重叠遮挡
3. 校验点：
   - 聚类人数合计 = 300
   - `customer_id` 未出现在任何特征矩阵中
   - `clustered_customers.csv` 行数 300、列数 8（原 7 列 + cluster）
   - 稳定性 ARI 用原始标签与重编号标签各算一次，结果一致
4. 未知项如实说明：本记录不声称对业务结论做过真实银行数据验证，
   所有画像与建议仅基于模拟数据的统计结果。

### 10.2 图片文件清单（均为程序实际生成）

| 步骤 | 图片文件 | 内容 |
|---|---|---|
| 1 | `outputs/01_raw_scatter.png` | 300 客户收入-余额散点（点大小=年龄） |
| 2 | `outputs/02_clustered_scatter.png` | k=4 四簇着色散点 + 簇中心 |
| 3 | `outputs/03_elbow.png` | K-SSE 折线，标注几何肘部 K=3 与建议 K=4 |
| 4 | `outputs/04_customer_profile.png` | 四簇 × 四特征分组柱状图（Z-score） |
| 5 | `outputs/05_feature_comparison.png` | 两组特征聚类对比 + 规模对比 |
| 6 | `outputs/06_stability.png` | 三个随机种子散点 + ARI 热力图 |
| 7 | `outputs/07_cluster_scatter.png` | 四簇二维散点（名称+人数标注） |
| 8 | `outputs/08_scale_comparison.png` | 原始量纲 vs 标准化 + 距离贡献对比 |

### 10.3 代码与数据产物清单

| 文件 | 说明 |
|---|---|
| `generated/01_load_and_viz.py` | 数据初检与散点图 |
| `generated/02_kmeans_cluster.py` | k=4 四特征聚类（作业版，覆盖了课程演示版） |
| `generated/clustered_customers.csv` | 300×8，含真实 cluster 列 |
| `generated/cluster_profile.csv` | 四簇画像（含客群名称） |
| `generated/03_choose_k.py` / `elbow_values.csv` | 肘部法 |
| `generated/04_customer_profile.py` / `mall_report.md` | 客户画像与报告 |
| `generated/05_feature_comparison.py` / `feature_comparison.csv` / `feature_comparison.md` | 更换特征集 |
| `generated/06_stability_test.py` / `stability_results.csv` / `stability_analysis.md` | 稳定性检验 |
| `generated/07_cluster_scatter.py` | 二维散点图 |
| `generated/08_scale_experiment.py` / `scale_comparison.csv` | 量纲陷阱实验 |

### 10.4 待本人补充的验证

以下内容需要本人在提交前亲自确认，本记录不代为填写：

- [ ] 逐张打开 8 张 PNG，确认与终端数字一致（图名见 10.2）
- [ ] 按老师要求补充课程平台要求的截图（若有指定命名/目录）
- [ ] 确认最终提交目录与老师要求一致

## 11. 说明与边界

1. 本记录中的提示词均为本次会话实际发送的原文，未改写、未补造。
2. `prompts/` 目录下的 7 份课程演示指令仅作参考资料阅读过，未整段作为作业提示词使用。
3. "标准化修正"没有独立提示词，已在第 3 节如实说明，并以量纲陷阱实验作为其实证。
4. 所有数值均来自程序实际运行输出，未手工修改或编造。
