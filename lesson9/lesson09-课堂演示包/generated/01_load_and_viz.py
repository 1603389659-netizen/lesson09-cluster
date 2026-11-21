# -*- coding: utf-8 -*-
"""
第 1 步：数据初检与初始可视化
客户分群项目第一件事：读取 data/mall_customers_sim.csv，完成 300 位客户的数据完整性
检查，并绘制收入-余额散点图（点大小映射年龄）。

约定：
- customer_id 仅作为客户编号，不进入模型。
- 建模字段：income、balance、age、product_count。
- 不修改原始 CSV。
"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 中文字体设置（优先 Windows 系统字体，兜底 macOS 字体） ----------
def setup_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 Bold
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
        r"C:\Windows\Fonts\SIMLI.TTF",    # 隶书
    ]
    for f in candidates:
        if os.path.exists(f):
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            matplotlib.rcParams["font.sans-serif"] = (
                [prop.get_name()] + matplotlib.rcParams["font.sans-serif"]
            )
            print(f"使用中文字体：{prop.get_name()}（{f}）")
            break
    else:
        # 兜底：常见字体名列表
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "SimSun",
            "Songti SC", "STHeiti", "PingFang SC", "Arial Unicode MS",
        ]
        print("未找到 Windows 中文字体文件，使用字体名候选列表")
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_chinese_font()

# ---------- 路径（在 lesson09-课堂演示包/ 目录下运行） ----------
CSV_PATH = "data/mall_customers_sim.csv"
OUT_PNG = "outputs/01_raw_scatter.png"
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ==================== 一、读取数据 ====================
print("=" * 64)
print("一、数据读取")
print("=" * 64)
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"数据文件：{CSV_PATH}")
print(f"行数：{df.shape[0]}   列数：{df.shape[1]}")
print(f"字段名称：{list(df.columns)}")

# ==================== 二、数据完整性检查 ====================
print("\n" + "=" * 64)
print("二、数据完整性检查")
print("=" * 64)

print("\n各列数据类型：")
print(df.dtypes.to_string())

print("\n各列缺失值数量：")
print(df.isna().sum().to_string())
total_na = int(df.isna().sum().sum())
print(f"\n总行数：{len(df)}   缺失值总数：{total_na}")
print(f"完全重复行数：{int(df.duplicated().sum())}")

# customer_id：仅作客户编号，不进入模型
print("\n--- customer_id 检查（仅作客户编号，不进入模型）---")
print(f"customer_id 唯一值数量：{df['customer_id'].nunique()}")
print(f"customer_id 重复数量：{int(df['customer_id'].duplicated().sum())}")

# ==================== 三、建模字段检查 ====================
print("\n" + "=" * 64)
print("三、建模字段检查（income / balance / age / product_count）")
print("=" * 64)
model_cols = ["income", "balance", "age", "product_count"]

print("\n数值列基本统计：")
print(df[model_cols].describe().round(2).to_string())

print("\n各字段数值范围与单位：")
print(f"  age           均值 {df['age'].mean():.2f}  最小 {df['age'].min()}  最大 {df['age'].max()}  （岁）")
print(f"  income        均值 {df['income'].mean():.2f}  最小 {df['income'].min():.2f}  最大 {df['income'].max():.2f}  （万元）")
print(f"  balance       均值 {df['balance'].mean():.2f}  最小 {df['balance'].min():.2f}  最大 {df['balance'].max():.2f}  （万元）")
print(f"  product_count 均值 {df['product_count'].mean():.2f}  最小 {df['product_count'].min()}  最大 {df['product_count'].max()}  （个）")

# ==================== 四、收入-余额散点图 ====================
print("\n" + "=" * 64)
print("四、生成收入-余额散点图")
print("=" * 64)
fig, ax = plt.subplots(figsize=(10, 7))
# 点大小映射 age：最小年龄对应 20，每岁放大 8
sizes = (df["age"] - df["age"].min()) * 8 + 20
scatter = ax.scatter(
    df["income"], df["balance"],
    s=sizes,
    alpha=0.6, c="#4C72B0", edgecolors="white", linewidths=0.5,
)
ax.set_title("300 位客户：收入-余额分布（点越大年龄越高）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"已保存：{OUT_PNG}")

print("\n数据初检完成。customer_id 已排除，建模字段：", model_cols)
