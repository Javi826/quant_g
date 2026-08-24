#profile_matrix_layout.py
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_days", type=int, default=1800,
                     help="Rows in the (days, cols) layout — total calendar-day span.")
parser.add_argument("--n_cols", type=int, default=100_000,
                     help="Columns — number of rule/combo slots. Production scale is ~966000; "
                          "reduce here if RAM is limited (each layout needs n_days*n_cols*4 bytes).")
parser.add_argument("--local_days", type=int, default=180,
                     help="Length of the daily_values segment written per column, matching a typical "
                          "combo's trade span.")
args = parser.parse_args()

N_DAYS      = args.n_days
N_COLS      = args.n_cols
LOCAL_DAYS  = min(args.local_days, N_DAYS)
BYTES_TOTAL = N_DAYS * N_COLS * 4

print(f"n_days={N_DAYS}  n_cols={N_COLS}  local_days={LOCAL_DAYS}  "
      f"per_matrix_size={BYTES_TOTAL / 1e9:.2f} GB")

# =============================================================================
# Precompute the same random write plan for both layouts, so both benchmarks
# perform the exact same number of writes at the exact same row offsets.
# =============================================================================
rng = np.random.default_rng(42)
row_offsets = rng.integers(0, N_DAYS - LOCAL_DAYS + 1, size=N_COLS)
payload     = rng.random(LOCAL_DAYS, dtype=np.float64).astype(np.float32)

# =============================================================================
# LAYOUT A — (n_days, n_cols), current production layout: writes a column
# =============================================================================
matrix_col_major = np.zeros((N_DAYS, N_COLS), dtype=np.float32)

t0 = time.perf_counter()
for col_idx in range(N_COLS):
    r0 = row_offsets[col_idx]
    matrix_col_major[r0:r0 + LOCAL_DAYS, col_idx] = payload
elapsed_col_major = time.perf_counter() - t0

del matrix_col_major

# =============================================================================
# LAYOUT B — (n_cols, n_days), transposed: writes a contiguous row segment
# =============================================================================
matrix_row_major = np.zeros((N_COLS, N_DAYS), dtype=np.float32)

t0 = time.perf_counter()
for col_idx in range(N_COLS):
    r0 = row_offsets[col_idx]
    matrix_row_major[col_idx, r0:r0 + LOCAL_DAYS] = payload
elapsed_row_major = time.perf_counter() - t0

del matrix_row_major

# =============================================================================
# REPORT
# =============================================================================
speedup = elapsed_col_major / elapsed_row_major if elapsed_row_major > 0 else float("inf")

print(f"\n{'=' * 70}")
print(f"LAYOUT A  (days, cols) — column writes (current)   : {elapsed_col_major:8.3f} s")
print(f"LAYOUT B  (cols, days) — row writes (transposed)    : {elapsed_row_major:8.3f} s")
print(f"SPEEDUP (A / B)                                      : {speedup:8.2f}x")
print(f"{'=' * 70}")