# === FILE: optimize_MC.py ===
# --------------------------------
import math
import optuna
import numpy as np
from numba import njit, prange
from ZX_utils import seed_for_symbol
from scipy.stats import skew, kurtosis

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
DTYPE = np.float32
EPS = 1e-12

# -----------------------------
# RNG LINEAR CONGRUENTIAL
# -----------------------------
@njit
def _lcg_next(state):
    a = np.uint64(6364136223846793005)
    c = np.uint64(1442695040888963407)
    MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
    state = (a * state + c) & MASK
    return state

@njit
def _lcg_uniform(state):
    state = _lcg_next(state)
    u = state / 18446744073709551616.0
    return state, u

@njit
def _box_muller_from_uniforms(u1, u2):
    if u1 <= 1e-18:
        u1 = 1e-18
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)

@njit
def generate_normal(state):
    state, u1 = _lcg_uniform(state)
    state, u2 = _lcg_uniform(state)
    z = _box_muller_from_uniforms(u1, u2)
    return state, z

# -----------------------------
# GENERACIÓN DE PATHS (ANTITHETIC VARIATES)
# -----------------------------
@njit(parallel=True)
def generate_paths_numba_array(
    last_price, mu_bar, sigma_bar,
    n_paths, n_obs, n_substeps, min_price,
    jump_prob_per_substep, jump_mu, jump_sigma,
    seeds
):
    """
    Genera paths sintéticos usando antithetic variates y volatilidad estocástica.
    Por cada path p, genera su contrapartida p_ant con z → -z.
    """
    dt_bar = 1.0
    dt_step = dt_bar / max(1, n_substeps)
    sqrt_dt_step = math.sqrt(dt_step)
    VOL_FACTOR = 0.4
    MIN_VOL_MULT = 0.1

    n_total = n_paths * 2
    paths_array3d = np.empty((n_total, n_obs, 4), dtype=DTYPE)

    for p in prange(n_paths):
        state = np.uint64(seeds[p])
        s_pos = last_price
        s_neg = last_price

        for i in range(n_obs):
            path_min_pos = 1e308
            path_max_pos = 0.0
            path_min_neg = 1e308
            path_max_neg = 0.0

            open_price_pos = s_pos
            open_price_neg = s_neg

            for k in range(n_substeps):
                # Genera z y antitético -z
                state, z = generate_normal(state)
                z_neg = -z

                # Volatilidad base aleatoria
                state, u_vm = _lcg_uniform(state)
                vol_multiplier = max(MIN_VOL_MULT, 1.0 + VOL_FACTOR * (u_vm * 2.0 - 1.0))
                if i > 0:
                    last_ret = math.log(s_pos / paths_array3d[p, i-1, 3])
                    # Factor de ajuste: mayores retornos absolutos ⇒ mayor sigma
                    vol_adjust = 1.0 + 0.8 * math.tanh(last_ret * 13.0)
                else:
                    vol_adjust = 1.0
                sigma_t = sigma_bar * vol_multiplier * vol_adjust

                # --- Ajuste estocástico para skew/kurtosis ---
                if z < 0:
                    sigma_t *= 1.1  # 10% más volatilidad en caídas
                else:
                    sigma_t *= 0.95  # 5% menos en subidas

                # Step normal
                inc_pos = (mu_bar - 0.5 * sigma_t**2) * dt_step + sigma_t * sqrt_dt_step * z
                inc_neg = (mu_bar - 0.5 * sigma_t**2) * dt_step + sigma_t * sqrt_dt_step * z_neg
                s_pos = max(s_pos * math.exp(inc_pos), min_price)
                s_neg = max(s_neg * math.exp(inc_neg), min_price)

                # Jump
                state, u_jump = _lcg_uniform(state)
                if u_jump < jump_prob_per_substep:
                    state, z_j = generate_normal(state)
                    jump_factor = math.exp(jump_mu + jump_sigma * z_j)
                    s_pos = max(s_pos * jump_factor, min_price)
                    s_neg = max(s_neg * jump_factor, min_price)

                # Min/max tracking
                path_min_pos = min(path_min_pos, s_pos)
                path_max_pos = max(path_max_pos, s_pos)
                path_min_neg = min(path_min_neg, s_neg)
                path_max_neg = max(path_max_neg, s_neg)

            # Guardar vela
            paths_array3d[p, i, 0] = open_price_pos
            paths_array3d[p, i, 1] = max(path_min_pos, min_price)
            paths_array3d[p, i, 2] = path_max_pos
            paths_array3d[p, i, 3] = s_pos

            paths_array3d[p + n_paths, i, 0] = open_price_neg
            paths_array3d[p + n_paths, i, 1] = max(path_min_neg, min_price)
            paths_array3d[p + n_paths, i, 2] = path_max_neg
            paths_array3d[p + n_paths, i, 3] = s_neg

    return paths_array3d


# -----------------------------
# GENERACIÓN DE PATHS PARA UN SÍMBOLO
# -----------------------------
def generate_paths_for_symbol(
        df_hist,
        n_paths,
        n_obs,
        n_substeps,
        vol_scale,
        min_price,
        jump_prob_per_substep,
        jump_mu,
        jump_sigma,
        timeframe,
        base_seed=42
    ):
    sym_name = getattr(df_hist, "name", None) or "symbol"
    df = df_hist

    if 'close' not in df.columns or len(df) < 2:
        return np.full((n_paths, n_obs, 4), np.nan, dtype=DTYPE)

    closes = df["close"].astype(float)
    logret = np.log(closes / closes.shift(1)).iloc[1:]
    mu_bar = float(np.nanmean(logret)) if not logret.empty else 0.0
    sigma_bar = max(float(np.nanstd(logret, ddof=0)) * vol_scale, 1e-8)
    last_price = float(df["close"].iloc[-1])

    seeds = np.array([seed_for_symbol(sym_name, base_seed=base_seed, path_idx=i) for i in range(n_paths)], dtype=np.uint64)
    paths_array3d = generate_paths_numba_array(last_price, mu_bar, sigma_bar,
                                               n_paths, n_obs, n_substeps, min_price,
                                               jump_prob_per_substep, jump_mu, jump_sigma, seeds)
    return paths_array3d

# -----------------------------
# KERNELS NUMBA PARA MÉTRICAS
# -----------------------------
@njit
def _acf_series_numba(x, nlags=50):
    N = x.shape[0]
    if N == 0:
        return np.zeros(nlags+1, dtype=DTYPE)
    xm = 0.0
    for i in range(N):
        xm += x[i]
    xm /= N
    max_lag = nlags if nlags < N-1 else N-1
    acf = np.zeros(max_lag+1, dtype=DTYPE)
    s0 = 0.0
    for i in range(N):
        t = x[i] - xm
        s0 += t * t
    if s0 == 0.0:
        acf[0] = 1.0
        for l in range(1, max_lag+1):
            acf[l] = 0.0
        if max_lag < nlags:
            out = np.zeros(nlags+1, dtype=DTYPE)
            out[:max_lag+1] = acf
            return out
        return acf
    for lag in range(max_lag+1):
        cov = 0.0
        for i in range(N - lag):
            cov += (x[i] - xm) * (x[i + lag] - xm)
        acf[lag] = cov / s0
    if max_lag < nlags:
        out = np.zeros(nlags+1, dtype=DTYPE)
        out[:max_lag+1] = acf
        return out
    return acf

@njit
def _ks_statistic_numba(a, b):
    na = a.shape[0]
    nb = b.shape[0]
    if na == 0 or nb == 0:
        return 1.0
    sa = np.sort(a.copy())
    sb = np.sort(b.copy())
    i = 0
    j = 0
    ca = 0
    cb = 0
    D = 0.0
    while i < na or j < nb:
        va = sa[i] if i < na else 1e308
        vb = sb[j] if j < nb else 1e308
        if va <= vb:
            ca += 1
            i += 1
            d = (ca/na - cb/nb)
        else:
            cb += 1
            j += 1
            d = (cb/nb - ca/na)
        if d > D:
            D = d
    return D

@njit
def _wasserstein_1d_numba(a, b):
    na = a.shape[0]
    nb = b.shape[0]
    if na == 0 or nb == 0:
        return 1e308
    sa = np.sort(a.copy())
    sb = np.sort(b.copy())
    i = 0
    j = 0
    prev_x = min(sa[0], sb[0])
    ca = 0
    cb = 0
    total = 0.0
    while i < na or j < nb:
        xa = sa[i] if i < na else 1e308
        xb = sb[j] if j < nb else 1e308
        x = xa if xa <= xb else xb
        dx = x - prev_x
        if dx > 0.0:
            Fa = ca / na
            Fb = cb / nb
            total += abs(Fa - Fb) * dx
            prev_x = x
        while i < na and sa[i] == x:
            ca += 1
            i += 1
        while j < nb and sb[j] == x:
            cb += 1
            j += 1
    return total

@njit(parallel=True)
def _compute_paths_metrics_numba(hist_rets, syn_array, nlags):
    npaths = syn_array.shape[0]
    dists = np.empty(npaths, dtype=DTYPE)
    ksD   = np.empty(npaths, dtype=DTYPE)
    acf_sum = np.zeros(nlags+1, dtype=DTYPE)
    for p in prange(npaths):
        r = syn_array[p]
        dists[p] = _wasserstein_1d_numba(hist_rets, r)
        ksD[p]   = _ks_statistic_numba(hist_rets, r)
        acf_p = _acf_series_numba(np.abs(r), nlags=nlags)
        for k in range(acf_p.shape[0]):
            acf_sum[k] += acf_p[k]
    return dists, ksD, acf_sum

# -----------------------------
# EVALUACIÓN SINTÉTICO vs HISTÓRICO (con retorno de métricas)
# -----------------------------
def evaluate_synthetic_vs_real(df_hist, arr_syn_list, nlags_acf=50, return_metrics=False):
    if arr_syn_list is None:
        return None

    syns = [arr_syn_list] if isinstance(arr_syn_list, np.ndarray) else [a for a in arr_syn_list if a is not None and getattr(a, "size", 0) > 0]
    if len(syns) == 0:
        return None

    closes_hist = df_hist['close'].to_numpy(dtype=DTYPE)
    if closes_hist.size < 2:
        return None
    hist_rets_full = np.log(closes_hist[1:] / closes_hist[:-1])

    syn_rets_list = []
    for arr in syns:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            closes = arr[:, :, 3].astype(DTYPE, copy=False)
            if closes.shape[1] < 2:
                continue
            rets = np.log(closes[:, 1:] / closes[:, :-1])
            for p in range(rets.shape[0]):
                syn_rets_list.append(rets[p].astype(DTYPE, copy=False))
        elif arr.ndim == 2 and arr.shape[1] > 3:
            closes = arr[:, 3].astype(DTYPE, copy=False)
            if closes.size < 2:
                continue
            rets = np.log(closes[1:] / closes[:-1])
            syn_rets_list.append(rets.astype(DTYPE, copy=False))
        else:
            continue

    if len(syn_rets_list) == 0:
        return None

    min_len = min(len(hist_rets_full), min(len(r) for r in syn_rets_list))
    if min_len < 2:
        return None

    hist_rets = hist_rets_full[-min_len:]
    syn_array = np.vstack([r[-min_len:].astype(DTYPE, copy=False) for r in syn_rets_list])

    hist_mean = float(np.mean(hist_rets))
    hist_std  = float(np.std(hist_rets, ddof=1)) if min_len > 1 else 0.0
    hist_skew = float(skew(hist_rets, bias=False))
    hist_kurt = float(kurtosis(hist_rets, bias=False))

    syn_means = np.mean(syn_array, axis=1)
    syn_stds  = np.std(syn_array, axis=1, ddof=1) if syn_array.shape[1] > 1 else np.zeros(syn_array.shape[0], dtype=DTYPE)
    syn_skews = skew(syn_array, axis=1, bias=False)
    syn_kurts = kurtosis(syn_array, axis=1, bias=False)

    sim_mean = 100.0 * max(0.0, 1.0 - abs(hist_mean - float(np.mean(syn_means))) / (abs(hist_std) + EPS))
    sim_std  = 100.0 / (1.0 + abs(hist_std - float(np.mean(syn_stds))) / (abs(hist_std) + 1e-8))
    sim_skew = 100.0 / (1.0 + abs(hist_skew - float(np.mean(syn_skews))) / (abs(hist_skew) + 1e-8))
    sim_kurt = 100.0 / (1.0 + abs(hist_kurt - float(np.mean(syn_kurts))) / (abs(hist_kurt) + 1e-8))

    dists, ksD, acf_sum = _compute_paths_metrics_numba(np.ascontiguousarray(hist_rets), np.ascontiguousarray(syn_array), nlags_acf)
    npaths = syn_array.shape[0]

    syn_acfs_mean = acf_sum / npaths
    hist_acf_abs = _acf_series_numba(np.abs(hist_rets), nlags=nlags_acf)
    diff_norm = 0.0
    hist_norm = 0.0
    for k in range(hist_acf_abs.shape[0]):
        diff_norm += (hist_acf_abs[k] - syn_acfs_mean[k]) ** 2
        hist_norm += hist_acf_abs[k] ** 2
    sim_acf = max(0.0, 100.0 * (1.0 - (diff_norm ** 0.5) / (hist_norm ** 0.5 + 1e-12))) if hist_norm > 0 else 0.0

    sim_ks = np.mean(100.0 * (1.0 - ksD))
    p95 = float(np.percentile(dists, 95)) + 1e-12
    wasser_scores = np.ones(npaths, dtype=DTYPE) * 100.0 if p95 <= 0 else 100.0 * (1.0 - np.clip(dists / p95, 0.0, 1.0))
    sim_wass = float(np.mean(wasser_scores))

    weights = np.array([0.20, 0.20, 0.10, 0.10, 0.10, 0.10, 0.20], dtype=DTYPE)
    metrics = np.array([sim_mean, sim_std, sim_skew, sim_kurt, sim_acf, sim_ks, sim_wass], dtype=DTYPE)
    score_total = float(np.sum(metrics * weights))

    if return_metrics:
        return np.append(metrics, score_total)
    return score_total

# -----------------------------
# OPTUNA PARA OPTIMIZACIÓN
# -----------------------------
def objective(trial, df_hist, n_paths, n_obs, n_substeps, min_price, timeframe, base_seed):
    vol_scale = trial.suggest_float("vol_scale", 0.3, 1.0, step=0.05)
    jump_prob_per_substep = trial.suggest_float("jump_prob_per_substep", 0.0, 0.030, step=0.0025)
    jump_mu = trial.suggest_float("jump_mu", -0.02, 0.02, step=0.0025)
    jump_sigma = trial.suggest_float("jump_sigma", 0.002, 0.024, step=0.002)

    df_syn_list = generate_paths_for_symbol(
        df_hist, n_paths=n_paths, n_obs=n_obs, n_substeps=n_substeps,
        vol_scale=vol_scale, min_price=min_price,
        jump_prob_per_substep=jump_prob_per_substep, jump_mu=jump_mu, jump_sigma=jump_sigma,
        timeframe=timeframe, base_seed=base_seed
    )
    score = evaluate_synthetic_vs_real(df_hist, df_syn_list)
    return score if score is not None else -np.inf

def optimize_params_optuna(df_hist, n_trials, n_paths, n_obs, n_substeps, min_price, timeframe, seed: int = 42):
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, df_hist, n_paths, n_obs, n_substeps, min_price, timeframe, seed), n_trials=n_trials)
    return study.best_params, study.best_value

def optimize_for_symbol(symbol, ohlcv_data, n_trials, n_paths, n_obs, n_substeps, min_price, timeframe, base_seed: int = 42):
    df_hist = ohlcv_data[symbol]
    best_params, best_score = optimize_params_optuna(
        df_hist, n_trials, n_paths, n_obs, n_substeps, min_price, timeframe, seed=base_seed
    )
    return symbol, best_params, best_score

# -----------------------------
# SUMMARY SCORE CON MÉTRICAS
# -----------------------------
def summary_score_all_paths(ohlcv_data, n_paths, n_obs, n_substeps, base_seed=42, DTYPE=np.float64):
    """
    Genera paths para todos los símbolos y devuelve un único score promedio ponderado
    junto con la media de cada métrica individual.
    """
    scores = []
    metrics_list = []

    for symbol, df_hist in ohlcv_data.items():
        df_syn_list = generate_paths_for_symbol(
            df_hist, n_paths=n_paths, n_obs=n_obs, n_substeps=n_substeps,
            vol_scale=1.0,
            min_price=df_hist['close'].min(),
            jump_prob_per_substep=0.01,
            jump_mu=0.0,
            jump_sigma=0.01,
            timeframe='1H',
            base_seed=base_seed
        )

        metrics = evaluate_synthetic_vs_real(df_hist, df_syn_list, return_metrics=True)
        if metrics is not None:
            metrics_list.append(metrics)
            scores.append(metrics[-1])  # último = score_total

    if len(scores) == 0:
        return None

    metrics_array = np.vstack(metrics_list)
    mean_metrics = np.mean(metrics_array, axis=0)

    return {
        "score_total": float(np.mean(scores)),
        "sim_mean": float(mean_metrics[0]),
        "sim_std": float(mean_metrics[1]),
        "sim_skew": float(mean_metrics[2]),
        "sim_kurt": float(mean_metrics[3]),
        "sim_acf": float(mean_metrics[4]),
        "sim_ks": float(mean_metrics[5]),
        "sim_wass": float(mean_metrics[6])
    }
