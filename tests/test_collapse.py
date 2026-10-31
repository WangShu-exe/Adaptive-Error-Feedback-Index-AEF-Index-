import numpy as np
from aef_index import AEFIndex

np.random.seed(42)
# 90% 数据在 [0, 1]，10% 数据在 [100, 200]
subset = np.sort(np.concatenate([
    np.random.rand(9000) * 1, 
    100 + np.random.rand(1000) * 100
]))
N = len(subset)

# 这次我们用 10 个锚点，预期分段是 9 段
model_q = AEFIndex(max_anchors=10, iterations=0)
bucket_q = model_q._evolve_bucket(subset)
ranks_q = np.searchsorted(subset, bucket_q, side='left') / N
deltas_q = np.diff(ranks_q)
# 正确计算最大段误差
err_q = np.max(np.abs(deltas_q - 1.0/9)) * N
print(f"[神仙开局] 锚点: {', '.join([f'{v:.2f}' for v in bucket_q])}")
print(f"最大段误差: {err_q:.1f} 名\n")

model_evo = AEFIndex(max_anchors=10, iterations=50)
bucket_evo = model_evo._evolve_bucket(subset)
ranks_evo = np.searchsorted(subset, bucket_evo, side='left') / N
deltas_evo = np.diff(ranks_evo)
err_evo = np.max(np.abs(deltas_evo - 1.0/9)) * N
print(f"[进化后]   锚点: {', '.join([f'{v:.2f}' for v in bucket_evo])}")
print(f"最大段误差: {err_evo:.1f} 名")