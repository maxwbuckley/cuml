# cuML RF — GPU perf TODO (for the 5090 desktop)

Written 2026-09-16 on a Mac with no CUDA toolchain. **Nothing in this file has been
measured.** Every claim is either (a) read off the cuML source, or (b) a number
published by another project on a different workload. Treat (b) as a direction, never
as an expected result.

Baseline commit: `61eff9aa2` (main, clean).
Line numbers below were accurate at that commit and will drift — grep the quoted code.

## Where the ideas came from

[MechaFauna-ai/Falcata](https://github.com/MechaFauna-ai/Falcata), a CUDA-native
LightGBM fork. Relevant reading, in order of usefulness:

- `docs/performance.md` — what landed, with leave-one-out ablation numbers
- `docs/perf-dead-ends.md` — what didn't, and why. Read this **before** starting any
  item, it will save a week
- `src/treelearner/cuda/cuda_data_partition.cu` — their partition kernels

Caveat that applies throughout: Falcata is leaf-wise gradient boosting, so its
histogram cells are (gradient, hessian) pairs and its hot loop is shaped differently
from ours. Its headline optimization (level-batched growth) is something cuML has had
since the batched-levelalgo rewrite. Only the items below actually transfer.

---

## Step 0 — do this first, it re-orders everything else

### 0a. Microbenchmark: fp64 vs fp32 shared-memory `atomicAdd`

The 5090 is a consumer part at 1/64 FP64 throughput. `RegressionBin::label_sum` is a
`double` and takes one `atomicAdd(double*)` **per row, per sampled column, per level**
(`cpp/src/decisiontree/batched-levelalgo/bins.cuh:108-114`). If FP64 atomics are as
expensive as the rate card suggests, that single instruction dominates regression
forest training on this card and item 3 below jumps to the top of the list.

Write a standalone `.cu` that reproduces the histogram geometry — 128 threads/block,
128 bins, the real bin struct sizes — and compare:

| arm | cell |
|---|---|
| A | `{double label_sum; unsigned long long count;}` (today) |
| B | `{float  label_sum; unsigned int count;}` |
| C | `{unsigned long long count;}` (classification today) |
| D | `{unsigned int count;}` (item 2) |

Report achieved atomic throughput and the ratio A/B and C/D. **This measurement
decides whether item 3 is worth an API change.** Takes an afternoon; do not skip it.

### 0b. Measurement discipline

Falcata's dead-ends doc records a "2.6× win" that evaporated when the box was quiet —
run-to-run spread was 18–55% under contention, 1–5% clean. Their rules, adopted here:

- Quiet desktop. No browser, no other CUDA process, no compile running.
- Medians of 3+, and report the spread, not just the median.
- **Interleaved A/B** (alternate arms within one session), not "all of A then all of B" —
  thermal drift on a desktop 5090 is larger than several of the effects below.
- Anything under ~1s per cell is unquotable. Use enough trees to get past 5s.
- A perf delta smaller than the measured spread is not a result.

### 0c. Baseline + a correctness net

Record baselines before touching anything: classification (binary and ~20-class),
regression, weighted variants, at a couple of `n_rows`/`n_cols`/`max_depth` points.

For each item below the file says whether the output must be **bit-identical**. Where
it must, diff the serialized trees, not the accuracy score — an accuracy match hides
structural drift. Note that cuML RF **regression is already run-to-run
non-reproducible** (fp64 atomic ordering), so "bit-identical" there means identical
tree structure and split thresholds, not identical leaf values to the last bit.

---

## Item 1 — `__ldcs` on the feature-value load in the histogram kernel

**Effort:** hours. **Bit-identical:** yes, by construction (cache hint only).
**Risk:** none beyond "might not help".

`buildHistogramsKernel`
(`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh:363-366`)
touches three arrays with very different reuse:

| array | access | reuse |
|---|---|---|
| `dataset.row_ids[i]` | coalesced | re-read by each of the ≤10 column blocks in the launch — **wants to stay resident** |
| `dataset.labels[row]` | scattered | re-read by each column block — **wants to stay resident** |
| `dataset.value(row, col)` | scattered | once per column per level; working set is a whole column, won't fit L2 anyway — **pure streaming** |

The streaming array is currently evicting the two that have reuse. Mark it evict-first.

Falcata measured +8–10% from exactly this split (their bin matrix streamed, their
gradient buffers pinned). Our loads are 4–8 bytes where theirs are 1, so expect less.

Implementation note: `Dataset::value` is `HDI`
(`cpp/src/decisiontree/batched-levelalgo/dataset.h:31`) so `__ldcs` can't go in it
directly. Add a `__device__`-only `value_streaming()` and call it **from the histogram
kernel only** — leave `countLocalLeftKernel` and the partition scan on the normal path,
they have different reuse.

Optional follow-on if this pays: an L2 persisting window over `labels` and `row_ids`
(`cudaStreamSetAttribute` / `accessPolicyWindow`). Falcata got +5.3%, but only after
sizing the carve-out to the buffer actually pinned — a fixed carve stranded L2. Size it
from `n_sampled_rows`, not a constant.

## Item 2 — 32-bit counts for unweighted classification

**Effort:** ~a day. **Bit-identical:** yes (a bin count cannot exceed `n_rows`).
**Risk:** build time, not correctness.

`ClassificationBin::count` is `unsigned long long`
(`cpp/src/decisiontree/batched-levelalgo/bins.cuh:15-18`). Guarded on
`n_rows < 2^32`, uint32 is exactly the same number. Buys:

- native 32-bit shared `ATOM.ADD` instead of 64-bit
- half the shared histogram → more blocks resident
- **the shared-memory path engages for more classes.** The gate is 16 KiB
  (`builder.cuh:166`, `:589`); at the default 128 bins the size is `1024·C + 524`, so
  **C ≥ 16 currently falls back to the global-memory histogram**. With uint32 that
  moves to C ≥ 32. A 20-class problem flips from global to shared outright — likely
  the biggest single effect in this item.

Scope it honestly before building it:

- **multiclass** — real win, largest around C = 8…31
- **binary** — small (atomic width and occupancy only)
- **regression / weighted classification** — **no win.** `{double, uint32}` still pads
  to 16 bytes, so `RegressionBin` and `WeightedClassificationBin` save nothing. Don't
  bother templating them.

Cost: another template dimension over the classification kernels
(`cpp/src/decisiontree/batched-levelalgo/kernels/classification-*.cu`), i.e. compile
time and binary size. Check that cost is acceptable before committing.

Free sliver, do it at the same time: `RegressionBin::count` can also be uint32 under
the same guard. No memory saving (alignment padding), but it narrows one of the two
atomics per row on the regression path.

## Item 3 — fp32 accumulator option (**the big one on this card, if 0a says so**)

**Effort:** days. **Bit-identical:** NO — changes accuracy.
**Risk:** needs a parameter, i.e. an API change.

This is the one that does not fit the "drop-in" bar, and the one most likely to matter
on a 5090. Falcata's `cuda_precision=fp32` (their `docs/performance.md` §6) measured
−12% to −36% train time at equal-or-better quality, plus a **separate** effect: halving
the histogram pool moved it from not-fitting to mostly-fitting L2, worth +26% on deep
trees on its own. The 5090's L2 is the same 96 MB their measurement was taken on.

Two mechanisms, both pointing the same way on a 1/64-FP64 part:

1. fp64 → fp32 atomics in the hot loop
2. histogram pool halves → L2 residency on deep trees

Gate on 0a. If the fp64/fp32 atomic ratio comes back near 1, this item is only the L2
half and drops well down the list. If it comes back near the 1/64 rate card, this is
the headline item for regression on consumer hardware.

If it proceeds: mirror Falcata and make it an explicit parameter with fp64 as the
default. **Do not** flip precision silently — the accuracy delta is real and
dataset-dependent (they found it dataset-dependent enough to keep it a user dial rather
than a planner decision). Quality-gate it on several regression datasets, reporting
RMSE deltas, not just wall time.

## Item 4 — stop reading the feature value twice during partitioning

**Effort:** ~a day + review. **Bit-identical:** yes. **Risk:** low, but it's a real
patch with a new scratch buffer.

`countLocalLeftKernel` evaluates `dataset.value(row, split.colid) <= split.quesval`
(`kernels/builder_kernels_impl.cuh:75`) and the scan's `partition_state` lambda then
evaluates the identical predicate over every row again (`:188`). Two scattered gathers
where one would do.

Cache the predicate as one bit per row between the passes. Falcata does this with
`__ballot_sync` (`src/treelearner/cuda/cuda_data_partition.cu:1436`), and their §10
took per-level partition cost from 17.4 ms to 2.2 ms on a 92M-row workload where the
partition was 77% of GPU time. **Profile cuML's partition share first** — if it isn't a
meaningful slice of our profile, this is not worth the patch.

## Item 5 — drop the redundant workload-info rebuild

**Effort:** ~15 lines. **Bit-identical:** yes. **Risk:** none.

`doSplit` calls `updateWorkloadInfo(work_items)` (`builder.cuh:508`) and re-uploads
`d_work_items` (`:507`), but `computeBestSplits` already did both (`:532-534`). When no
feature-resampling retry round ran — the common case — that is identical data recomputed
and re-sent. The host loop is O(n_sampled_rows / 128) iterations on the critical path
between two stream syncs; at 1M rows, ~7.8k iterations plus a ~190 KB H2D, per level,
per tree.

Guard on "the last `computeBestSplits` call used the full work-item list" and skip.

Expect this to be small — it's host-side, and it may hide behind async work entirely.
Worth doing because it's free, not because it's big. If it doesn't measure, keep it
anyway as a simplification.

---

## Larger, not drop-in — only after the above

### Pre-binned feature matrix

The largest structural gap. cuML never materializes bins: it stores raw fp32/fp64 and
re-derives the bin with a `lower_bound` binary search over the quantile array for every
(row, sampled column) at every node batch
(`kernels/builder_kernels_impl.cuh:365-374`). Every other GPU tree library
(LightGBM, XGBoost, CatBoost, Falcata) bins once into uint8, 4-bit-packed when ≤16 bins.

Per value: 1 byte and zero compares, versus 8 bytes and ~7 dependent compares. A
depth-20 tree re-pays that search 20× per row per tree.

Costs an extra `n_rows × n_cols` byte buffer and a rewrite of `Dataset::value` plus the
histogram inner loop. Budget it as a project. Note it would also make item 1 more
valuable (1-byte streaming loads are exactly the shape Falcata measured).

### Histogram subtraction (parent − sibling)

cuML zeroes and rebuilds histograms for every node in the batch, both children included
(`builder.cuh:637`). Build only the smaller sibling and derive the larger. Up to ~2×
less histogram work. Needs sibling pairing in `NodeQueue::Push` and parent histograms
kept live (memory against `max_batch_size × max_n_bins × n_blks_for_cols ×
num_outputs`). Cheaper to reason about **after** item 2, because integer counts subtract
exactly while fp64 sums do not.

---

## Do NOT do these — measured losses elsewhere

Falcata tried all of these and recorded them in `docs/perf-dead-ends.md`. Re-read their
entry before overriding any of them.

- **Shape formulas for kernel knobs.** `n_blks_for_cols = 10` (`builder.cuh:202`) and
  the 16 KiB smem cutoff (`:166`) are baked constants and it's tempting to derive them
  from dataset shape. Falcata did, and got **+35% on one dataset and −45% on another
  from the identical model** — the crossover depends on the runtime leaf-size
  distribution and achieved occupancy, not on shape. If these get replaced, replace
  them by *measurement* with a cached per-shape result, never by a fitted formula.
- **CUDA graphs for the level loop.** Below their noise floor once level batching
  existed. cuML already syncs only twice per batch (`builder.cuh:522`, `:544`).
- **NVRTC / runtime JIT of the histogram kernel.** As a blanket default it measured
  −30% on their flagship shape.
- **Tensor-core histogram construction.** ~2× *slower* than shared atomics after three
  escalating implementations. The per-column shared atomics never actually conflict, so
  there is no serialization to remove — the premise was wrong.
- **Multi-GPU beyond the level-batched all-reduce.** Refuted in every variant they
  tried, including NVLink transport, bigger data, and feature-parallel. Our
  `distributed` path already all-reduces once per histogram batch (`builder.cuh:598`),
  which is the arrangement they found 1.54× better than per-split. Leave it alone.
- **Per-tree compact column view.** Doesn't map to cuML: we sample features **per node**
  (`column_samples[nid * dataset.n_sampled_cols + colIndex]`), not per tree, so the
  union over a batch is most columns — and with column-major input a column is already
  contiguous.

---

## Suggested order

1. **0a microbenchmark** — gates item 3, cheap, decides the whole plan
2. **0b/0c** baselines and the bit-identical diff harness
3. **Item 1** (`__ldcs`) and **item 5** (redundant rebuild) — one PR, both free
4. **Item 2** (uint32 classification counts) — if the multiclass numbers justify the
   compile-time cost
5. **Item 3** (fp32 option) — only if 0a came back hot; it needs a design discussion
   about the parameter before any code
6. Profile. Then decide between item 4, pre-binning, and subtraction **from the profile**,
   not from this file.

## Open questions for the desktop session

- What fraction of GPU time is `buildHistogramsKernel` vs the partition kernels vs
  `findBestSplitsKernel`, at a few shapes? Everything above is guesswork until this
  exists. Get it from `nsys` before writing any patch beyond item 5.
- Does the 16 KiB smem gate actually fire on realistic multiclass workloads, and is the
  global-memory histogram path as slow as the gate assumes? The constant is documented
  as "measured locally" (`builder.cuh:162-166`) — verify it still holds on Blackwell.
- Is `n_blks_for_cols = 10` anywhere near right on this card? Measure, don't derive
  (see the dead-ends list).
