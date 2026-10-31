import numpy as np
from aef_index import AEFIndex

# ==========================================
# 数据构造：连续分布，但密度从 0 到 100 线性增加
# P(x) ∝ x  →  CDF 反函数 sqrt 变换
# ==========================================
np.random.seed(42)
u = np.random.rand(10000)
subset = np.sqrt(u) * 100            # 数据在 [0,100]，右半区密集
subset = np.sort(subset)
print(f"数据分布：左半 [0,50] 有 {int((subset<50).sum())} 条，"
      f"右半 [50,100] 有 {int((subset>=50).sum())} 条\n")


class EqualSpacingModel(AEFIndex):
    """对照组：等距初始化，不加进化"""
    def _evolve_bucket(self, subset):
        unique_subset = np.unique(subset)
        lo, hi = unique_subset[0], unique_subset[-1]
        # 强制等距，iterations=0 时就是纯等距锚点
        if self.iterations == 0:
            return np.linspace(lo, hi, self.max_anchors)
        return super()._evolve_bucket(subset)


# ==========================================
# 三组实验：等距 + 进化次数递增
# ==========================================
header = f"{'实验':<6} | {'初始化':<10} | {'迭代':>4} | {'最大段误差':>10} | {'锚点分布(前6)':<40}"
print("=" * 95)
print(header)
print("-" * 95)

for label, iters in [("A", 0), ("B", 10), ("C", 100)]:
    m = EqualSpacingModel(max_anchors=6, iterations=iters)
    m._evolve_bucket(subset)    # 直接测这个桶的进化结果
    bucket = m._evolve_bucket(subset) if iters > 0 else np.linspace(subset[0], subset[-1], 6)

    # 算每段的真实数据占比
    ranks = np.searchsorted(subset, bucket, side='left') / len(subset)
    deltas = np.diff(ranks)
    max_err = np.max(np.abs(deltas - 1.0/(len(bucket)-1)))

    bkt_str = ", ".join([f"{v:.1f}" for v in bucket[:6]])
    print(f"{label:<6} | {'等距/进化':<10} | {iters:>4} | {max_err:>10.4f} | {bkt_str:<40}")

print("=" * 95)