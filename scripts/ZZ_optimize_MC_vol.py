import math
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from Z_utils import get_symbols, filter_symbols
from ZZ_connect_03 import connect_bitget
from scipy.stats import ks_2samp
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import optuna

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# -----------------------------
# CONFIGURACIÓN INICIAL
# -----------------------------
TIMEFRAME        = '1h'
DATA_FOLDER      = "crypto_2025"
MIN_VOL_USDT     = 2500000
MIN_PRICE        = 0.0001
BASE_SEED        = 42

# -----------------------------
# Parámetros para Monte Carlo
# -----------------------------
MC_N_PATHS        = 10       
MC_N_OBS_PER_PATH = 1000    
MC_N_SUBSTEPS     = 10       

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def _align_next_time(ts, timeframe_td):
    ts = pd.Timestamp(ts)
    floored = ts.floor(freq=timeframe_td)
    return floored + timeframe_td if floored == ts else floored + timeframe_td

def _seed_for_symbol_path(symbol, base_seed, path_index):
    base = int(base_seed) % (2**31-1) if base_seed is not None else abs(hash(symbol)) % (2**31-1)
    return int((base + int(path_index)) % (2**31-1))

# -----------------------------
# UTILIDADES VOLUMEN
# -----------------------------
def _fit_lognormal_params(vol_series, eps=1e-9):
    v = np.asarray(vol_series).astype(float) + eps
    logv = np.log(v)
    mu = float(np.mean(logv))
    sigma = float(np.std(logv, ddof=0))
    return mu, sigma

def generate_volume_for_bars_parametric(df_hist, synthetic_df, rng, vol_multiplier_series=None,
                                        hour_seasonality=True, alpha_absret=8.0,
                                        jump_spike_mu=1.0, jump_spike_sigma=0.6, jump_flags=None, eps=1e-8):
    """
    Parametric: log-normal base, escalado por vol_multiplier, por |ret|, con estacionalidad horaria.
    jump_flags: lista/array booleana con len == len(synthetic_df) indicando si hubo jump en la barra.
    """
    if 'volume' in df_hist.columns and len(df_hist['volume'].dropna())>10:
        mu_ln, sigma_ln = _fit_lognormal_params(df_hist['volume'].dropna(), eps=eps)
    else:
        mu_ln, sigma_ln = np.log(1.0 + eps), 1.0

    if hour_seasonality and hasattr(df_hist.index, 'hour'):
        hourly = df_hist['volume'].groupby(df_hist.index.hour).mean().reindex(range(24)).fillna(method='ffill').fillna(1.0)
        hourly = hourly / np.nanmean(hourly)
    else:
        hourly = None

    vols = []
    closes = synthetic_df['close'].values
    logrets = np.zeros(len(closes))
    logrets[1:] = np.log(np.maximum(closes[1:], eps) / np.maximum(closes[:-1], eps))

    for i in range(len(synthetic_df)):
        base_sample = rng.lognormal(mean=mu_ln, sigma=sigma_ln)
        vm = float(vol_multiplier_series[i]) if (vol_multiplier_series is not None) else 1.0
        abs_fac = 1.0 + alpha_absret * min(1.0, abs(logrets[i]))
        vol = base_sample * vm * abs_fac
        if hourly is not None:
            hour = synthetic_df.index[i].hour
            vol *= float(hourly.loc[hour])
        # spike por jump
        if jump_flags is not None and jump_flags[i]:
            spike = rng.lognormal(jump_spike_mu, jump_spike_sigma)
            vol *= spike
        vols.append(max(vol, eps))
    return np.array(vols)

def generate_volume_for_bars_empirical(df_hist, synthetic_df, rng, n_buckets=5, add_noise=True, jump_flags=None, eps=1e-8):
    """
    Bootstrap condicional: muestreamos volúmenes históricos por bucket de |logret|.
    """
    hist = df_hist.copy()
    hist['logret'] = np.log(hist['close'] / hist['close'].shift(1)).fillna(0.0)
    try:
        hist['absret_bucket'] = pd.qcut(np.abs(hist['logret']), q=n_buckets, duplicates='drop')
    except Exception:
        hist['absret_bucket'] = pd.cut(np.abs(hist['logret']), bins=n_buckets, duplicates='drop')

    bucket_vols = {}
    for b, g in hist.groupby('absret_bucket'):
        bucket_vols[str(b)] = g['volume'].values

    vols = []
    closes = synthetic_df['close'].values
    logrets_syn = np.zeros(len(synthetic_df))
    logrets_syn[1:] = np.log(np.maximum(closes[1:], eps) / np.maximum(closes[:-1], eps))

    global_pool = hist['volume'].values if 'volume' in hist.columns and len(hist['volume'].dropna())>0 else np.array([1.0])

    for i in range(len(synthetic_df)):
        # pick bucket by matching approximate quantile
        absr = abs(logrets_syn[i])
        # simple heuristic: choose random bucket present
        buckets = list(bucket_vols.keys())
        if len(buckets) == 0:
            sample = rng.choice(global_pool)
        else:
            chosen = rng.choice(buckets)
            arr = bucket_vols[chosen]
            if len(arr)==0:
                sample = rng.choice(global_pool)
            else:
                sample = rng.choice(arr)
        if add_noise:
            sample = sample * max(0.001, (1.0 + 0.1 * rng.normal()))
        if jump_flags is not None and jump_flags[i]:
            spike = rng.lognormal(1.0, 0.6)
            sample = sample * spike
        vols.append(max(sample, eps))
    return np.array(vols)

# -----------------------------
# GENERACIÓN DE PATHS SINTÉTICOS (con volumen integrado)
# -----------------------------
def generate_synthetic_path_fixed_bar(df_hist, n_obs=MC_N_OBS_PER_PATH, n_substeps=MC_N_SUBSTEPS, vol_scale=1.0, seed=None,
                                      min_price=MIN_PRICE, jump_prob_per_substep=0.0022,
                                      jump_mu=-0.002, jump_sigma=0.015, timeframe=TIMEFRAME,
                                      volume_mode='parametric', volume_params=None):
    """
    volume_mode: 'parametric' o 'empirical'
    volume_params: dict con parámetros pasados a la función de volumen
    """
    rng = np.random.default_rng(seed)
    df = df_hist.copy().sort_index()
    closes = df["close"].astype(float)
    logret = np.log(closes / closes.shift(1)).iloc[1:]
    mu_bar = float(np.nanmean(logret)) if not logret.empty else 0.0
    sigma_bar = max(float(np.nanstd(logret, ddof=0))*vol_scale, 1e-8)
    
    dt_bar = 1.0
    dt_step = dt_bar / max(1, n_substeps)
    open_price = float(df["close"].iloc[-1])
    last_ts = pd.Timestamp(df.index[-1])
    current_date = _align_next_time(last_ts, pd.Timedelta(timeframe))

    synthetic_rows = []
    vol_multiplier_series = []
    jump_flags_series = []

    for i in range(n_obs):
        s = open_price
        path = [s]
        z = rng.standard_normal(size=n_substeps)
        u = rng.random(size=n_substeps)
        sigma_t = sigma_bar
        jumped_flag = False
        # accumulative vol multiplier for the bar (we'll average or take max)
        vol_multipliers_sub = []

        for k in range(n_substeps):
            vol_multiplier = max(0.1, 1.0 + 0.5*rng.normal(0,0.8))
            vol_multipliers_sub.append(vol_multiplier)
            sigma_t = sigma_bar * vol_multiplier
            increment = (mu_bar - 0.5*sigma_t**2)*dt_step + sigma_t*math.sqrt(dt_step)*z[k]
            s = max(s*math.exp(increment), min_price)
            
            if u[k] < jump_prob_per_substep:
                jump_factor = math.exp(rng.normal(jump_mu,jump_sigma))
                s = max(s*jump_factor, min_price)
                jumped_flag = True
            path.append(s)

        open_p = open_price
        close_p = float(path[-1])
        high_p = float(max(np.max(path), open_p, close_p))
        low_p  = float(max(min(np.min(path), open_p, close_p), min_price))
        synthetic_rows.append([open_p, low_p, high_p, close_p, current_date])
        open_price = close_p
        current_date += pd.Timedelta(timeframe)

        # summarize vol multiplier for the bar
        vol_multiplier_series.append(float(np.mean(vol_multipliers_sub)))
        jump_flags_series.append(bool(jumped_flag))

    df_bars = pd.DataFrame(synthetic_rows, columns=["open","low","high","close","time"]).set_index("time")

    # parámetros por defecto para volumen
    if volume_params is None:
        volume_params = {}

    # generar volúmenes
    if volume_mode == 'parametric':
        vols = generate_volume_for_bars_parametric(df_hist, df_bars, rng,
                                                   vol_multiplier_series=vol_multiplier_series,
                                                   jump_flags=jump_flags_series,
                                                   **volume_params)
    else:
        vols = generate_volume_for_bars_empirical(df_hist, df_bars, rng,
                                                  jump_flags=jump_flags_series,
                                                  **volume_params)

    df_bars['volume'] = vols
    return df_bars

def generate_paths_for_symbol_fixed_bar(df_hist, n_paths=MC_N_PATHS, **kwargs):
    paths = []
    for i in range(n_paths):
        seed = _seed_for_symbol_path(df_hist.name if hasattr(df_hist,'name') else 'symbol', BASE_SEED, i)
        try:
            p = generate_synthetic_path_fixed_bar(df_hist, seed=seed, **kwargs)
            paths.append(p)
        except Exception as e:
            print(f"Error generating path {i}: {e}")
            paths.append(None)
    return paths

# -----------------------------
# MÉTRICAS DE SIMILITUD (precio + volumen)
# -----------------------------
def _acf_series(x, nlags=50):
    x = np.asarray(x) - np.mean(x)
    result = np.correlate(x, x, mode='full')
    acf = result[result.size//2:] / (result[result.size//2] + 1e-18)
    return acf[:nlags+1]

def mean_similarity(h, s, std_hist, eps=1e-12):
    diff = abs(h - s)
    return 100.0 * max(0.0, 1.0 - diff / (abs(std_hist) + eps))
    
def moment_similarity(h, s, eps=1e-8):
    return 100.0 / (1.0 + abs(h - s) / (abs(h) + eps))

def wasserstein_similarity(hist_rets, syn_rets_list):
    hist_array = np.array(hist_rets)
    wasser_scores = []
    # normalización: 95th percentile de dists entre sintéticos
    ref_dists = [wasserstein_distance(hist_array, np.array(s)) for s in syn_rets_list if len(s)>0]
    if len(ref_dists) == 0:
        return 0.0
    ref95 = np.percentile(ref_dists, 95) + 1e-12
    for r in syn_rets_list:
        r_array = np.array(r)
        dist = wasserstein_distance(hist_array, r_array)
        score = 100 * (1 - np.clip(dist / ref95, 0, 1))
        wasser_scores.append(score)
    return np.mean(wasser_scores)

def ks_similarity(hist_rets, syn_rets_list):
    return np.mean([100*(1 - ks_2samp(hist_rets, r)[0]) for r in syn_rets_list])

def acf_similarity(hist_rets, syn_rets_list, nlags=50):
    hist_acf_abs = _acf_series(np.abs(hist_rets), nlags=nlags)
    syn_acf_abs_means = np.mean([_acf_series(np.abs(r), nlags=nlags) for r in syn_rets_list], axis=0)
    return max(0.0, 100 * (1 - np.linalg.norm(hist_acf_abs - syn_acf_abs_means) / (np.linalg.norm(hist_acf_abs) + 1e-12)))

# Métricas de volumen
def ks_similarity_volume(hist_vols, syn_vols_list):
    hist = np.asarray(hist_vols)
    scores = []
    for v in syn_vols_list:
        try:
            stat = ks_2samp(hist, np.asarray(v))[0]
            scores.append(100*(1 - stat))
        except Exception:
            scores.append(0.0)
    return np.mean(scores)

def wasserstein_similarity_volume(hist_vols, syn_vols_list):
    hist = np.asarray(hist_vols)
    dists = [wasserstein_distance(hist, np.asarray(v)) for v in syn_vols_list]
    if len(dists)==0:
        return 0.0
    ref95 = np.percentile(dists, 95) + 1e-12
    scores = [100*(1 - np.clip(d/ref95, 0, 1)) for d in dists]
    return np.mean(scores)

def acf_similarity_volume(hist_vols, syn_vols_list, nlags=50):
    hist_acf = _acf_series(hist_vols, nlags=nlags)
    syn_mean = np.mean([_acf_series(v, nlags=nlags) for v in syn_vols_list], axis=0)
    return max(0.0, 100 * (1 - np.linalg.norm(hist_acf - syn_mean) / (np.linalg.norm(hist_acf) + 1e-12)))

def corr_absret_volume_similarity(df_hist_cut, syns_cut):
    # compute pearson corr between |logret| and volume for hist and synthetic avg, then compare
    hist_rets = np.log(df_hist_cut['close'] / df_hist_cut['close'].shift(1)).dropna()
    hist_vol = df_hist_cut['volume'].iloc[1:len(hist_rets)+1] if 'volume' in df_hist_cut.columns else None
    if hist_vol is None or len(hist_vol)==0:
        return 50.0  # neutral if no data
    hist_corr = abs(np.corrcoef(np.abs(hist_rets), hist_vol)[0,1])
    syn_corrs = []
    for s in syns_cut:
        syn_rets = np.log(s['close'] / s['close'].shift(1)).dropna()
        syn_vol = s['volume'].iloc[1:len(syn_rets)+1] if 'volume' in s.columns else None
        if syn_vol is None or len(syn_vol)==0:
            continue
        try:
            syn_corrs.append(abs(np.corrcoef(np.abs(syn_rets), syn_vol)[0,1]))
        except Exception:
            pass
    if len(syn_corrs)==0:
        return 50.0
    syn_corr = np.mean(syn_corrs)
    # similarity: closer -> 100
    return 100.0 * max(0.0, 1.0 - abs(hist_corr - syn_corr) / (abs(hist_corr) + 1e-12))

# -----------------------------
# EVALUACIÓN SINTÉTICO vs HISTÓRICO (para Optuna)
# -----------------------------
def evaluate_synthetic_vs_real_improved(df_hist, df_syn_list, vol_weight=0.25):
    """
    Devuelve score combinado entre precio y volumen.
    vol_weight: peso del score de volumen en el score final (ej. 0.25)
    """
    syns = [df for df in df_syn_list if df is not None and "close" in df.columns and "volume" in df.columns]
    if len(syns) == 0:
        return None

    n_obs = min(len(df_hist), min(len(s) for s in syns))
    if n_obs < 10:
        return None

    df_hist_cut = df_hist.iloc[-n_obs:]
    syns_cut = [s.iloc[-n_obs:].reset_index(drop=True) for s in syns]

    hist_rets = np.log(df_hist_cut['close'] / df_hist_cut['close'].shift(1)).dropna()
    syn_rets_list = [np.log(s['close'] / s['close'].shift(1)).dropna() for s in syns_cut]

    # Price similarities (tal como antes)
    sim_mean = mean_similarity(hist_rets.mean(), np.mean([r.mean() for r in syn_rets_list]), hist_rets.std(ddof=1))
    sim_std  = moment_similarity(hist_rets.std(ddof=1), np.mean([r.std(ddof=1) for r in syn_rets_list]))
    sim_skew = moment_similarity(skew(hist_rets, bias=False), np.mean([skew(r, bias=False) for r in syn_rets_list]))
    sim_kurt = moment_similarity(kurtosis(hist_rets, bias=False), np.mean([kurtosis(r, bias=False) for r in syn_rets_list]))
    sim_acf  = acf_similarity(hist_rets, syn_rets_list)
    sim_ks   = ks_similarity(hist_rets, syn_rets_list)
    sim_wass = wasserstein_similarity(hist_rets, syn_rets_list)

    price_score = np.mean([sim_mean, sim_std, sim_skew, sim_kurt, sim_acf, sim_ks, sim_wass])

    # Volume similarities
    hist_vols = df_hist_cut['volume'].dropna().values if 'volume' in df_hist_cut.columns else np.array([])
    syn_vols_list = [s['volume'].dropna().values for s in syns_cut]

    if len(hist_vols) == 0 or any(len(v)==0 for v in syn_vols_list):
        # si no hay volumen histórico bien formado, devolvemos solo precio
        return price_score

    vol_ks = ks_similarity_volume(hist_vols, syn_vols_list)
    vol_wass = wasserstein_similarity_volume(hist_vols, syn_vols_list)
    vol_acf = acf_similarity_volume(hist_vols, syn_vols_list)
    vol_corr = corr_absret_volume_similarity(df_hist_cut, syns_cut)

    vol_score = np.mean([vol_ks, vol_wass, vol_acf, vol_corr])

    final_score = (1.0 - vol_weight) * price_score + vol_weight * vol_score
    return final_score

# -----------------------------
# OPTUNA PARA OPTIMIZACIÓN
# -----------------------------
# -----------------------------
# OPTUNA PARA OPTIMIZACIÓN
# -----------------------------
def objective(trial, df_hist, n_paths=MC_N_PATHS):
    vol_scale = trial.suggest_float("vol_scale", 0.3, 0.9, step=0.05)
    jump_prob_per_substep = trial.suggest_float("jump_prob_per_substep", 0.0, 0.015, step=0.0025)
    jump_mu = trial.suggest_float("jump_mu", -0.01, 0.01, step=0.0025)
    jump_sigma = trial.suggest_float("jump_sigma", 0.002, 0.024, step=0.002)

    # parámetros de volumen
    alpha_absret = trial.suggest_float("alpha_absret", 2.0, 20.0, step=0.5)
    jump_spike_mu = trial.suggest_float("jump_spike_mu", 0.2, 2.0, step=0.1)
    jump_spike_sigma = trial.suggest_float("jump_spike_sigma", 0.2, 1.5, step=0.05)

    params_trial = {
        "vol_scale": vol_scale,
        "jump_prob_per_substep": jump_prob_per_substep,
        "jump_mu": jump_mu,
        "jump_sigma": jump_sigma,
        "volume_mode": "parametric",
        "volume_params": {          # <-- SOLO aquí dentro
            "alpha_absret": alpha_absret,
            "jump_spike_mu": jump_spike_mu,
            "jump_spike_sigma": jump_spike_sigma
        }
    }

    df_syn_list = generate_paths_for_symbol_fixed_bar(df_hist,
                                                      n_paths=n_paths,
                                                      n_obs=MC_N_OBS_PER_PATH,
                                                      n_substeps=MC_N_SUBSTEPS,
                                                      **params_trial)
    score = evaluate_synthetic_vs_real_improved(df_hist, df_syn_list)
    return score if score is not None else -np.inf



def optimize_params_optuna(df_hist, n_trials=50, n_paths=MC_N_PATHS):
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df_hist, n_paths=n_paths), n_trials=n_trials)
    return study.best_params, study.best_value


# -----------------------------
# PIPELINE PRINCIPAL
# -----------------------------
if __name__ == '__main__':
    start_time = time.time()
    exchange = connect_bitget()
    symbols = get_symbols(exchange)

    try:
        ohlcv_data, filtered_symbols, removed_symbols = filter_symbols(
            symbols, MIN_VOL_USDT, TIMEFRAME, data_folder=DATA_FOLDER, mode='backtesting', min_price=MIN_PRICE
        )
    except Exception:
        ohlcv_data = {}

    symbols_list = list(ohlcv_data.keys())

    def optimize_for_symbol(symbol):
        df_hist = ohlcv_data[symbol]
        best_params, best_score = optimize_params_optuna(df_hist, n_trials=50, n_paths=MC_N_PATHS)
        return symbol, best_params, best_score

    results = Parallel(n_jobs=-1)(delayed(optimize_for_symbol)(s) for s in symbols_list)
    best_params_dict = {symbol: params for symbol, params, score in results}

    for symbol, params, score in results:
        print(f"{symbol} -> Mejor score = {score:.2f}")

    # Generar paths finales y tabla de similitud
    final_paths = {}
    final_similarity = []

    for symbol, params, score in results:
        print(f"Generando paths finales para {symbol} ...")
        df_hist = ohlcv_data[symbol]
        # si best_params_dict no contiene volume_mode, usar parametric por defecto
        # extraemos los parámetros guardados
        params_local = best_params_dict.get(symbol, {}) or {}
        
        # asegurarnos de que volume_mode y volume_params existan
        if 'volume_mode' not in params_local:
            params_local['volume_mode'] = 'parametric'
        
        if 'volume_params' not in params_local or params_local['volume_params'] is None:
            params_local['volume_params'] = {}
        
        # eliminamos keys peligrosas que puedan estar fuera de volume_params
        for k in ['alpha_absret', 'jump_spike_mu', 'jump_spike_sigma']:
            if k in params_local:
                params_local.pop(k)
        
        # ahora sí generamos los paths
        df_syn_list = generate_paths_for_symbol_fixed_bar(df_hist, n_paths=MC_N_PATHS,
                                                          n_obs=MC_N_OBS_PER_PATH, n_substeps=MC_N_SUBSTEPS,
                                                          **params_local)


        final_paths[symbol] = df_syn_list

        syns = [df for df in df_syn_list if df is not None and "close" in df.columns and "volume" in df.columns]
        if len(syns) == 0:
            continue

        n_obs = min(len(df_hist), min(len(s) for s in syns))
        df_hist_cut = df_hist.iloc[-n_obs:]
        syns_cut = [s.iloc[-n_obs:].reset_index(drop=True) for s in syns]

        hist_rets = np.log(df_hist_cut['close'] / df_hist_cut['close'].shift(1)).dropna()
        syn_rets_list = [np.log(s['close'] / s['close'].shift(1)).dropna() for s in syns_cut]

        hist_moments = {
            "mean": float(hist_rets.mean()),
            "std": float(hist_rets.std(ddof=1)),
            "skew": float(stats.skew(hist_rets, bias=False)),
            "kurt": float(stats.kurtosis(hist_rets, bias=False))
        }
        syn_moments_avg = {
            "mean": np.mean([r.mean() for r in syn_rets_list]),
            "std": np.mean([r.std(ddof=1) for r in syn_rets_list]),
            "skew": np.mean([stats.skew(r, bias=False) for r in syn_rets_list]),
            "kurt": np.mean([stats.kurtosis(r, bias=False) for r in syn_rets_list])
        }

        moment_sims = {
            "mean": mean_similarity(hist_moments["mean"], syn_moments_avg["mean"], hist_moments["std"]),
            "std":  moment_similarity(hist_moments["std"], syn_moments_avg["std"]),
            "skew": moment_similarity(hist_moments["skew"], syn_moments_avg["skew"]),
            "kurt": moment_similarity(hist_moments["kurt"], syn_moments_avg["kurt"]),
        }

        # métricas de volumen
        hist_vols = df_hist_cut['volume'].dropna().values if 'volume' in df_hist_cut.columns else np.array([])
        syn_vols_list = [s['volume'].dropna().values for s in syns_cut]

        vol_metrics = {
            "vol_ks": ks_similarity_volume(hist_vols, syn_vols_list) if len(hist_vols)>0 else np.nan,
            "vol_wass": wasserstein_similarity_volume(hist_vols, syn_vols_list) if len(hist_vols)>0 else np.nan,
            "vol_acf": acf_similarity_volume(hist_vols, syn_vols_list) if len(hist_vols)>0 else np.nan,
            "vol_corr": corr_absret_volume_similarity(df_hist_cut, syns_cut) if len(hist_vols)>0 else np.nan
        }

        final_similarity.append({
            "symbol": symbol,
            "sim_mean": moment_sims["mean"],
            "sim_std": moment_sims["std"],
            "sim_skew": moment_sims["skew"],
            "sim_kurt": moment_sims["kurt"],
            "acf_similarity": acf_similarity(hist_rets, syn_rets_list),
            "ks_similarity": ks_similarity(hist_rets, syn_rets_list),
            "wasserstein_similarity": wasserstein_similarity(hist_rets, syn_rets_list),
            "vol_ks": vol_metrics["vol_ks"],
            "vol_wass": vol_metrics["vol_wass"],
            "vol_acf": vol_metrics["vol_acf"],
            "vol_corr_absret": vol_metrics["vol_corr"]
        })

    df_similarity = pd.DataFrame(final_similarity)
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    print(df_similarity)

    end_time = time.time()
    hours, remainder = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Tiempo total: {int(hours)} h {int(minutes)} min {int(seconds)} s")

# -----------------------------
# FUNCIÓN PARA EXPORTAR PARÁMETROS ÓPTIMOS
# -----------------------------
def get_best_params_for_all(symbols, ohlcv_data, n_trials=50, n_paths=MC_N_PATHS):
    def optimize_for_symbol_local(symbol):
        df_hist = ohlcv_data[symbol]
        best_params, best_score = optimize_params_optuna(df_hist, n_trials=n_trials, n_paths=n_paths)
        return symbol, best_params, best_score

    results = Parallel(n_jobs=-1)(delayed(optimize_for_symbol_local)(s) for s in symbols)
    best_params_dict = {symbol: params for symbol, params, score in results}
    return best_params_dict
