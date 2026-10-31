# AEF-Index: Adaptive Error-Feedback Index
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Streaming_Update_Ready-success.svg)]()

> **专为长尾、极端脏数据设计的轻量级、支持流式增删的自适应概率排名索引。**

## 💡 项目背景 (Motivation)
传统的排名估计（如 ECDF、t-digest）在海量数据面临三大死穴：
1. **内存爆炸**：全量排序或存储直方图开销极大。
2. **长尾失真**：在极值区间或偏斜分布下，等分位插值误差极大。
3. **静态死板**：无法应对高频的流式 `Insert/Delete`，或者面临严重的写放大。

**AEF-Index** 是一个基于“全局稀疏二分 + 局部误差反馈进化”的二级索引架构。前期的理论推导已证明：**本结构下的理论误差严格遵循 \(O(1/\sqrt{n})\) 曲线**。在工程落地中，我们在 100 万级脏数据下，仅用极小内存，将排名误差压低至 `10^-6` 级别，并原生支持微秒级的流式增量更新。

## 🚀 核心创新 (Why It Works)
- **Absolute Rank Mapping**：抛弃概率抽样放大，直接映射真实全局排名，彻底消除二项分布采样方差。
- **O(1) Spike Catching**：针对大范围重复尖峰（如极端脏数据 88.88），采用字典哈希直接拦截，查询误差清零。
- **Error-Feedback Evolution**：局部桶内根据实际误差反馈，动态插入真实数据锚点。具备早停（Early-Stopping）机制，训练开销极小。
- **Local Incremental Retraining (局部增量重训)**：结合 `REBUILD_THRESHOLD` 缓冲池机制，流式插入时仅对命中的局部桶进行延迟重建，避免了传统索引的全量重训与写放大。

## 📊 性能与实验数据 (Benchmark)

在 **1,000,000** 条包含 30% 取整坍缩与 10% 极端尖峰（88.88）的脏数据上，测试集为 3000 条完全未见过的陌生数据（Out-of-Sample）。

通过调整全局采样点（`n_global`），AEF-Index 可动态平衡“内存占用”与“极致精度”：

### 配置 A：火力全开 (n_global = 10000)
- **内存占用**：813.81 KB
- **增量插入测试**：插入 10,000 条后，Mean Error 仅从 `2.35e-06` 微升至 `4.70e-06`。

| 指标 (静态盲测) | Pure ECDF (纯全局) | AEF-Index (配置A) | 提升幅度 |
| :--- | :--- | :--- | :--- |
| Mean Error | 1.61e-03 | **6.54e-06** | **99.59%** |
| p50 Error | 1.43e-03 | **5.27e-07** | **99.96%** |
| p99 Error | 4.51e-03 | **2.05e-05** | **99.54%** |
| Max Error | 5.17e-03 | **8.11e-03** | (极值边界) |

### 配置 B：极致轻量 (n_global = 2000)
- **内存占用**：214.88 KB
- **增量插入测试**：插入 10,000 条后，Mean Error 稳定在 `2.62e-05`。

| 指标 (静态盲测) | Pure ECDF (纯全局) | AEF-Index (配置B) | 提升幅度 |
| :--- | :--- | :--- | :--- |
| Mean Error | 1.61e-03 | **2.27e-05** | **98.59%** |
| p50 Error | 1.43e-03 | **1.22e-06** | **99.91%** |
| p99 Error | 4.51e-03 | **5.08e-04** | **88.74%** |
| Max Error | 5.17e-03 | **8.11e-03** | (极值边界) |

*(注：增量测试报告的耗时包含测试脚本中为验证真实排名而执行的全量重排开销；算法本身的局部 `insert` 耗时在微秒级。)*

## 🖼️ 可视化结果 (Visualizations)

<div align="center">
  <img src="assets/aef_incremental_update_n_global = 2000.png" alt="AEF-Index Error CDF and Incremental Update" width="95%">
  <p><i>左图：CDF 曲线（AEF-Index 死死贴底，碾压纯 ECDF）。右图：增量更新动态误差曲线（10,000次插入后误差仅微弱上升，展现极强流式鲁棒性）。</i></p>
</div>

## 📂 快速开始 (Quick Start)
```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/AEF-Index.git
cd AEF-Index

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行主基准测试（包含静态对比与增量更新）
python benchmark.py

# 4. (可选) 运行病理测试，观察误差反馈与物理边界
python tests/test_evolution.py
python tests/test_collapse.py

## 🙏 参考与致谢 (References & Acknowledgements)

AEF-Index 的设计与实现，深受以下经典算法与数据结构思想的启发。在此向各位研究者致敬。

### 核心参考
1.  **Tim Kraska et al. "The Case for Learned Index Structures." SIGMOD 2018.**
    *   **参考点**：本文首次提出用模型替代传统索引结构的思想，是AEF-Index“全局稀疏路由”层的灵感来源[reference:3]。
2.  **Ted Dunning. "The t-digest: Efficient estimates of distributions." Software Impacts 2021.**
    *   **参考点**：t-digest 在流式分位数估计上的精妙设计，启发了 AEF-Index 的“局部误差反馈进化”机制。
3.  **Jialin Ding et al. "ALEX: An Updatable Adaptive Learned Index." SIGMOD 2020.**
    *   **参考点**：ALEX 是首个支持动态更新的学习型索引，其“局部自适应”思想为 AEF-Index 的增量更新与延迟重建策略提供了重要参考[reference:5]。

### 其他参考
4.  **Patrick O'Neil et al. "The Log-Structured Merge-Tree (LSM-Tree)." Acta Informatica 1996.**
    *   **参考点**：LSM-Tree 的“缓冲池”与“批处理”思想，启发了 AEF-Index 的 `REBUILD_THRESHOLD` 延迟重建机制。
5.  **Rudolf Bayer & Edward M. McCreight. "Organization and Maintenance of Large Ordered Indices." Acta Informatica 1972.**
    *   **参考点**：B+ 树作为传统索引的标杆，其在节点分裂与平衡上的设计，是 AEF-Index 实现“节点分裂（打法2）”思路的基石[reference:7]。
6.  **Tim Peters. "Timsort." 2002.**
    *   **参考点**：Python 内置排序算法 Timsort 对“现实世界数据”的适应性设计，启发了 AEF-Index 对“脏数据”与“长尾分布”进行专项优化的思路[reference:8]。

### 免责声明
AEF-Index 是一个独立设计的原型系统，并非上述任何算法的直接实现或封装。其核心创新在于将“全局模型路由”、“局部误差反馈”与“哈希尖峰拦截”相结合，形成了一套全新的轻量级自适应索引架构。
