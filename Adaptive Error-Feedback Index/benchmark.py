import numpy as np
import time
import matplotlib.pyplot as plt
from aef_index import AEFIndex

# ==========================================
# 🔧 全局参数配置区（只改这里！）
# ==========================================
CONFIG = {
    # ---- 数据规模 ----
    "STATIC_N":        1_000_000,   # Part 1 静态全量数据量 (可以改 5_000_000)
    "SMALL_N":           200_000,   # Part 2 初始流式数据量
    "TEST_Q_SIZE":         3_000,   # 测试查询数量
    "INCREMENT_N":        10_000,   # Part 2 增量插入的数据量
    
    # ---- 脏数据特征 ----
    "ROUND_RATIO":          0.30,   # 取整坍缩的比例
    "SPIKE_RATIO":          0.10,   # 尖峰 88.88 的比例
    
    # ---- AEF-Index 模型参数 ----
    "N_GLOBAL":             10000,   # 全局根节点采样量
    "MAX_ANCHORS":            10,   # 局部桶内最大锚点数
    "ITERATIONS":             0,   # 误差反馈迭代次数
    "DUP_THRESHOLD":       10000,   # 尖峰哈希拦截阈值
    "REBUILD_THRESHOLD":     100,   # 延迟重建阈值 (插满多少条更新一次锚点)
}


class PureECDF:
    def __init__(self, full_data, n=20000):
        idx = np.unique(np.random.randint(0, len(full_data), size=n))
        self.F = np.sort(full_data[idx])
        self.n = len(self.F)
        self.total_N = len(full_data)

    def predict(self, x):
        import bisect
        j = bisect.bisect_left(self.F, x)
        return j / self.n * self.total_N if j < self.n else float(self.total_N)


def gen_dirty_data(N, seed=42):
    """生成带取整坍缩和尖峰的脏数据"""
    np.random.seed(seed)
    d = np.random.lognormal(mean=3, sigma=1.5, size=N)
    
    m1 = np.random.rand(N) < CONFIG["ROUND_RATIO"]
    d[m1] = np.round(d[m1])
    
    m2 = np.random.rand(N) < CONFIG["SPIKE_RATIO"]
    d[m2] = 88.88
    
    return np.sort(d)


if __name__ == "__main__":
    # ==========================================
    # Part 1：静态精度对比
    # ==========================================
    print("=" * 65)
    print(f"Part 1: 静态精度对比 (Data Size = {CONFIG['STATIC_N']})")
    print("=" * 65)

    data = gen_dirty_data(CONFIG["STATIC_N"])

    np.random.seed(100)
    Q = np.random.lognormal(mean=3, sigma=1.5, size=CONFIG["TEST_Q_SIZE"])
    m1 = np.random.rand(CONFIG["TEST_Q_SIZE"]) < CONFIG["ROUND_RATIO"]
    Q[m1] = np.round(Q[m1])
    Q[::100] = 88.88
    true_ranks = np.searchsorted(data, Q)

    model_pure = PureECDF(data, n=20000)

    t0 = time.time()
    # 注意：静态测试里为了拿到最好效果，rebuild_threshold 不生效
    model_evolved = AEFIndex(
        n_global=CONFIG["N_GLOBAL"], 
        max_anchors=CONFIG["MAX_ANCHORS"], 
        iterations=CONFIG["ITERATIONS"],
        dup_threshold=CONFIG["DUP_THRESHOLD"]
    )
    model_evolved.fit(data)
    print(f"[训练耗时] {time.time() - t0:.3f}s")

    t0 = time.time()
    preds_pure = np.array([model_pure.predict(x) for x in Q])
    preds_evolved = np.array([model_evolved.predict(x) for x in Q])
    print(f"[查询耗时] {(time.time() - t0) * 1000:.1f} ms")

    err_pure = np.abs(preds_pure - true_ranks) / CONFIG["STATIC_N"]
    err_evolved = np.abs(preds_evolved - true_ranks) / CONFIG["STATIC_N"]

    print("-" * 65)
    print(f"{'指标':>18} | {'纯全局':>14} | {'AEF-Index':>14}")
    print("-" * 65)
    for name, fn in [("Mean Error", np.mean),
                     ("p50 Error", lambda x: np.percentile(x, 50)),
                     ("p99 Error", lambda x: np.percentile(x, 99)),
                     ("Max Error", np.max)]:
        a, b = fn(err_pure), fn(err_evolved)
        print(f"{name:>18} | {a:>14.2e} | {b:>14.2e}")
    print("=" * 65)


    # ==========================================
    # Part 2：增量更新测试
    # ==========================================
    print("\n" + "=" * 65)
    print(f"Part 2: 增量更新测试 (初始 {CONFIG['SMALL_N']} + 插入 {CONFIG['INCREMENT_N']} 条)")
    print("=" * 65)

    small = gen_dirty_data(CONFIG["SMALL_N"], seed=42)
    
    # 初始化支持增量的模型
    model_inc = AEFIndex(
        n_global=CONFIG["N_GLOBAL"], 
        max_anchors=CONFIG["MAX_ANCHORS"], 
        iterations=CONFIG["ITERATIONS"],
        dup_threshold=CONFIG["DUP_THRESHOLD"],
        rebuild_threshold=CONFIG["REBUILD_THRESHOLD"]
    )
    model_inc.fit(small)

    # 生成增量数据
    np.random.seed(777)
    new_data = np.random.lognormal(mean=3, sigma=1.5, size=CONFIG["INCREMENT_N"])
    m1 = np.random.rand(CONFIG["INCREMENT_N"]) < CONFIG["ROUND_RATIO"]
    new_data[m1] = np.round(new_data[m1])
    new_data[::50] = 88.88

    check_points = []
    dynamic_errs = []
    current_data = small.copy()

    # 固定考核题目
    np.random.seed(999)
    Q_eval = small[np.random.randint(0, len(small), size=300)]

    t0 = time.time()
    for idx, x in enumerate(new_data):
        model_inc.insert(x)
        current_data = np.sort(np.append(current_data, x))

        # 每 1000 条统计一次误差
        if (idx + 1) % max(1, CONFIG["INCREMENT_N"] // 10) == 0 or (idx + 1) == CONFIG["INCREMENT_N"]:
            true_r = np.searchsorted(current_data, Q_eval)
            pred_r = np.array([model_inc.predict(v) for v in Q_eval])
            mean_err = np.mean(np.abs(pred_r - true_r) / len(current_data))
            
            check_points.append(idx + 1)
            dynamic_errs.append(mean_err)
            print(f"  -> 已插入 {idx+1:>6} 条, 当前 Mean Error: {mean_err:.2e}")
    
    insert_time = time.time() - t0
    print(f"[插入总耗时] {insert_time * 1000:.1f} ms (平均 {insert_time / CONFIG['INCREMENT_N'] * 1e6:.0f} μs/条)")

    # 最终全量数据与整体评估
    full_after = np.sort(np.concatenate([small, new_data]))
    np.random.seed(999)
    Q_after = full_after[np.random.randint(0, len(full_after), size=CONFIG["TEST_Q_SIZE"])]
    true_after = np.searchsorted(full_after, Q_after)
    pred_after = np.array([model_inc.predict(x) for x in Q_after])
    err_after = np.abs(pred_after - true_after) / len(full_after)

    print(f"[插入后精度] Mean Error: {np.mean(err_after):.2e}")
    print(f"[插入后精度] p99 Error : {np.percentile(err_after, 99):.2e}")
    print(f"[插入后精度] Max Error : {np.max(err_after):.2e}")
    print("=" * 65)


    # ==========================================
    # Part 3：可视化
    # ==========================================
    print("\n正在生成增量更新可视化图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：CDF 对比
    cdf_after = np.arange(1, len(err_after) + 1) / len(err_after)
    model_pure_after = PureECDF(full_after, n=20000)
    pred_pure_after = np.array([model_pure_after.predict(x) for x in Q_after])
    err_pure_after = np.abs(pred_pure_after - true_after) / len(full_after)

    ax1.plot(np.sort(err_pure_after), np.arange(1, len(err_pure_after) + 1) / len(err_pure_after), 
             label='Pure ECDF', color='tab:blue', linewidth=2)
    ax1.plot(np.sort(err_after), cdf_after, 
             label='AEF-Index (Incremental)', color='tab:orange', linewidth=2)
    ax1.set_xlim(0, max(np.percentile(err_pure_after, 99.9), np.percentile(err_after, 99.9)))
    ax1.set_ylim(0.9, 1.0)
    ax1.set_xlabel('Relative Rank Error')
    ax1.set_ylabel('CDF (90%-100%)')
    ax1.set_title('Error CDF After Incremental Inserts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：动态误差随插入条数的变化
    ax2.plot(check_points, dynamic_errs, marker='o', linestyle='-', color='tab:green', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Items Inserted')
    ax2.set_ylabel('Mean Relative Rank Error')
    ax2.set_title('Dynamic Error During Incremental Updates')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('aef_incremental_update_final.png', dpi=150)
    print("=> 对比图已保存为: aef_incremental_update_final.png")
    plt.show()