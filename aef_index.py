import numpy as np
import bisect

class AEFIndex:
    def __init__(self, n_global=1000, max_anchors=30, iterations=10,
                 dup_threshold=200000, rebuild_threshold=100):
        self.n_global = n_global
        self.max_anchors = max_anchors
        self.iterations = iterations
        self.dup_threshold = dup_threshold
        self.rebuild_threshold = rebuild_threshold # 攒够多少条脏数据才重算锚点

        self.root_samples = None
        self.global_ranks = None
        self.children = {}      
        self.subsets = {}       
        self.dirty_counts = {}  # 记录每个桶累积的插入次数
        self.dup_dict = {}      
        self.N = 0

    def _evolve_bucket(self, subset):
        unique_subset = np.unique(subset)
        if len(unique_subset) <= self.max_anchors:
            return unique_subset

        bucket = np.quantile(subset, np.linspace(0, 1, self.max_anchors))
        bucket = np.unique(bucket)

        if len(bucket) < self.max_anchors:
            extra = np.quantile(unique_subset, np.linspace(0, 1, self.max_anchors))
            bucket = np.unique(np.concatenate([bucket, extra]))
            if len(bucket) > self.max_anchors:
                bucket = bucket[:self.max_anchors]

        stuck = 0
        for epoch in range(self.iterations):
            ranks_left = np.searchsorted(subset, bucket, side='left')
            ranks_right = np.searchsorted(subset, bucket, side='right')
            ranks = (ranks_left + ranks_right) / (2.0 * len(subset))
            actual_deltas = np.diff(ranks)
            if len(actual_deltas) == 0:
                break
            span = ranks[-1] - ranks[0]
            if span <= 0:
                break
            actual_deltas = actual_deltas / span

            expected = 1.0 / (len(bucket) - 1)
            errors = actual_deltas - expected
            worst_idx = int(np.argmax(errors))

            if errors[worst_idx] < expected * 0.05:
                break

            x_new = (bucket[worst_idx] + bucket[worst_idx + 1]) / 2.0
            pos = np.searchsorted(unique_subset, x_new)
            left = unique_subset[pos - 1] if pos > 0 else None
            right = unique_subset[pos] if pos < len(unique_subset) else None

            if left is None: x_new = right
            elif right is None: x_new = left
            else: x_new = left if (x_new - left) <= (right - x_new) else right

            if x_new in (bucket[worst_idx], bucket[worst_idx + 1]):
                stuck += 1
                if stuck >= 5: break
                continue
            stuck = 0

            bucket = np.insert(bucket, worst_idx + 1, x_new)
            ranks_new = np.searchsorted(subset, bucket) / len(subset)
            deltas = np.diff(ranks_new)
            if len(deltas) > 2:
                candidate_idxs = list(range(1, len(deltas) - 1))
                if (worst_idx + 1) in candidate_idxs:
                    candidate_idxs.remove(worst_idx + 1)
                if candidate_idxs:
                    redundant = candidate_idxs[int(np.argmin(deltas[candidate_idxs]))]
                    bucket = np.delete(bucket, redundant)

        return np.unique(bucket)

    def fit(self, full_data):
        self.N = len(full_data)
        idx = np.unique(np.random.randint(0, self.N, size=self.n_global))
        self.root_samples = np.sort(full_data[idx])

        unique_vals, unique_counts = np.unique(full_data, return_counts=True)
        big_dup = np.where(unique_counts >= self.dup_threshold)[0]
        for k in big_dup:
            v = float(unique_vals[k])
            start = int(np.searchsorted(full_data, v, side='left'))
            self.dup_dict[v] = start

        self.global_ranks = np.searchsorted(full_data, self.root_samples, side='left').astype(np.float64)
        counts = np.diff(self.global_ranks).astype(int)

        total_anchors = 0
        for i in range(len(counts)):
            cnt = counts[i]
            if cnt <= 0: continue

            l = int(self.global_ranks[i])
            r = int(self.global_ranks[i + 1])
            subset = full_data[l:r]

            n_local = min(self.max_anchors, max(2, int(np.sqrt(cnt))))
            bkt = self._evolve_bucket(subset) if cnt > n_local else np.unique(subset)

            if len(bkt) > 0:
                bkt_local_ranks = np.searchsorted(subset, bkt, side='left').astype(np.float64)
                self.children[i] = (bkt, bkt_local_ranks)
                self.subsets[i] = subset
                self.dirty_counts[i] = 0
                total_anchors += len(bkt)

        print(f"[AEF-Index] 训练完成。区间 {len(self.children)}，锚点 {total_anchors}，"
              f"内存 {(total_anchors * 16) / 1024:.2f} KB")

    def _locate_bucket(self, x):
        n = len(self.root_samples)
        i = bisect.bisect_left(self.root_samples, x)
        if i > 0 and i < n and self.root_samples[i] > x:
            i -= 1
        return max(0, min(i, n - 2))

    def predict(self, x):
        if x in self.dup_dict:
            return float(self.dup_dict[x])

        i = self._locate_bucket(x)
        if i in self.children:
            bucket, bucket_local_ranks = self.children[i]
            j = bisect.bisect_left(bucket, x)
            base_rank = self.global_ranks[i]
            
            if j == 0:
                return float(base_rank + bucket_local_ranks[0])
            elif j == len(bucket):
                return float(base_rank + bucket_local_ranks[-1])
            else:
                v0, v1 = bucket[j - 1], bucket[j]
                r0, r1 = base_rank + bucket_local_ranks[j - 1], base_rank + bucket_local_ranks[j]
                if v1 > v0:
                    frac = (x - v0) / (v1 - v0)
                    return float(r0 + frac * (r1 - r0))
                return float(r0)

        n = len(self.root_samples)
        j_glob = bisect.bisect_left(self.root_samples, x)
        if j_glob == n:
            return float(self.N)
        if j_glob < n and self.root_samples[j_glob] == x:
            return float(self.global_ranks[j_glob])
        return float((j_glob - 0.5) / n * self.N)

    def insert(self, x):
        x = float(x)
        i = self._locate_bucket(x)

        self.global_ranks[i + 1:] += 1
        self.N += 1
        
        # 【修复 1】：无论 x 是不是尖峰，只要比尖峰小，尖峰排名就要后移
        for k in self.dup_dict:
            if k > x:
                self.dup_dict[k] += 1

        if x in self.dup_dict:
            return

        if i in self.subsets:
            subset = self.subsets[i]
            pos = int(np.searchsorted(subset, x, side='left'))
            self.subsets[i] = np.insert(subset, pos, x)
            
            # 【修复 2】：延迟重建
            self.dirty_counts[i] += 1
            if self.dirty_counts[i] >= self.rebuild_threshold:
                bkt = self._evolve_bucket(self.subsets[i])
                if len(bkt) > 0:
                    local_ranks = np.searchsorted(self.subsets[i], bkt, side='left').astype(np.float64)
                    self.children[i] = (bkt, local_ranks)
                self.dirty_counts[i] = 0