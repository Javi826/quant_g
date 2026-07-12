#shared/shared_batchs/rule_mining/rule_runner.py
import os
import logging

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.pipeline.wfo import run_wfo_is, WFO_WINDOW_CONFIG
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.plotting import plot_filter_comparison, plot_portfolio_comparison
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE
from shared_batchs.runs.run_correlation import decorrelate_by_profit
from shared_batchs.runs.run_best_wfo_portfolio import find_best_portfolio_combination_wfo

from shared_batchs.rule_mining.rule_generator import generate_all_rules, MAX_DEPTH
from shared_batchs.rule_mining.rule_deploy import run_deploy_rule, _save_rule_deploy_batch

logger = logging.getLogger("BOT_batch.rule_mining.runner")

_OP_SLUG = {">": "gt", "<": "lt"}


def _slugify_label(label: str) -> str:
    slug = label
    for op, tag in _OP_SLUG.items():
        slug = slug.replace(op, tag)
    slug = slug.replace(" AND ", "_AND_").replace(" ", "_")
    slug = slug.replace("[", "").replace("]", "").replace("-", "m")
    return slug

def _run_single_rule(
    i: int,
    total: int,
    rule: dict,
    ohlcv_data: dict,
    param_names: list,
    lists_for_grid: list,
    order_amount: int,
    timeframe: str,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    stability_th: float,
    dtype,
    inner_n_jobs: int,
    show_progress: bool,
    n_symbols: int,
    log_level: int,
    save_trades: bool,
    brief_trades_folder: str,
) -> dict:

    logging.basicConfig(level=log_level, format="%(message)s", force=True)
    logging.getLogger("joblib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    rule_id = f"{i:05d}_{timeframe}_{rule['side']}_{_slugify_label(rule['label'])}"

    best_params, approved_wfo, wfo_net_gain, wfo_max_dd, _, wfo_test_trades, df_results, param_stability = run_wfo_is(
        ohlcv_data          = ohlcv_data,
        param_names         = param_names,
        lists_for_grid      = lists_for_grid,
        signal_fn           = rule["signal_fn"],
        signal_params_keys  = [],
        order_amount        = order_amount,
        timeframe           = timeframe,
        net_gain_th         = net_gain_th,
        dd_th               = dd_th,
        r2_th               = r2_th,
        stability_th        = stability_th,
        dtype               = dtype,
        n_jobs              = inner_n_jobs,
        show_progress       = show_progress,
        n_symbols           = n_symbols,
    )

    n_windows = len(df_results) - 1 if df_results is not None else 0
    n_trades  = 0 if wfo_test_trades is None else len(wfo_test_trades)

    if save_trades and wfo_test_trades is not None and not wfo_test_trades.empty:
        os.makedirs(brief_trades_folder, exist_ok=True)
        wfo_test_trades.to_csv(
            os.path.join(brief_trades_folder, f"trades_wfo_test_{rule_id}.csv"),
            index=False,
        )

    metrics = None
    if wfo_test_trades is not None and not wfo_test_trades.empty:
        metrics = compute_metrics(wfo_test_trades, capital=INITIAL_BALANCE, name="")

    logger.debug(f"[{i + 1}/{total}] {rule['side']:<5} {rule['label']} -> "
                 f"{'PASS' if approved_wfo else 'FAIL'} NetGain={wfo_net_gain:.1f}% DD={wfo_max_dd:.1f}%")

    return {
        "rule_id":         rule_id,
        "timeframe":        timeframe,
        "side":             rule["side"],
        "specs":            rule["specs"],
        "signal_fn":        rule["signal_fn"],
        "label":            rule["label"],
        "approved":         approved_wfo,
        "net_gain":         wfo_net_gain,
        "max_dd":           wfo_max_dd,
        "n_trades":         n_trades,
        "n_windows":        n_windows,
        "win_rate":         metrics["Win_Rate"]      if metrics else 0.0,
        "profit_factor":    metrics["Profit_Factor"] if metrics else 0.0,
        "calmar":           metrics["Calmar"]        if metrics else 0.0,
        "r_squared":        metrics["R_Squared"]     if metrics else 0.0,
        "param_stability":  param_stability,
        "best_params":      best_params,
        "wfo_test_trades":  wfo_test_trades,
    }

def run_rule_mining(
    ohlcv_data: dict,
    timeframe: str,
    param_grid: dict,
    order_amount: int,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    stability_th: float,
    dtype,
    rules_n_jobs: int = 1,
    inner_n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    max_depth: int = MAX_DEPTH,
    log_level: int = logging.INFO,
    save_trades: bool = False,
    brief_trades_folder: str = None,
) -> list:

    param_names    = list(param_grid.keys())
    lists_for_grid = [param_grid[k] for k in param_names]

    arr_sample = next(iter(ohlcv_data.values()))
    all_rules  = generate_all_rules({
        "open":  arr_sample["open"],
        "high":  arr_sample["high"],
        "low":   arr_sample["low"],
        "close": arr_sample["close"],
        "volume_quote": arr_sample["volume_quote"],
    }, max_depth=max_depth)

    total_rules = len(all_rules)
    logger.info(f"RULE MINING ── {timeframe} ── total candidate rules: {total_rules}")

    with tqdm_joblib(tqdm(desc=f"RULE MINING {timeframe}", total=total_rules, dynamic_ncols=True)):
        raw_results = Parallel(n_jobs=rules_n_jobs)(
            delayed(_run_single_rule)(
                i, total_rules, rule, ohlcv_data, param_names, lists_for_grid, order_amount,
                timeframe, net_gain_th, dd_th, r2_th, stability_th, dtype, inner_n_jobs, show_progress, n_symbols,
                log_level, save_trades, brief_trades_folder,
            )
            for i, rule in enumerate(all_rules)
        )

    if raw_results:
        _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe, {})
        logger.info(
            f"STAGE 1 ── WFO completed  ── {raw_results[0]['n_windows']} windows | "
            f"train={_wfo_cfg.get('train_months')}m  test={_wfo_cfg.get('test_months')}m"
        )

    return raw_results


def finalize_rule_mining(
    all_raw_results: list,
    ohlcv_data_by_timeframe: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    data_folder: str,
    inner_n_jobs: int = -1,
    n_symbols: int = None,
    show_plots: bool = False,
    correlation_threshold: float = 0.75,
    run_correlation: bool = True,
    run_best_portfolio: bool = True,
    run_deploy: bool = False,
    symbols_live_folder: str = None,
    deploy_output_path: str = None,
) -> list:

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    validated_wfo_test = [
        (r["rule_id"], r["wfo_test_trades"])
        for r in all_raw_results
        if r["approved"] and r["wfo_test_trades"] is not None and not r["wfo_test_trades"].empty
    ]

    _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "PRE-CORRELATION")
    _print_min_by_group(all_raw_results, [rid for rid, _ in validated_wfo_test])
    if run_correlation and validated_wfo_test:
        logger.info(f"\n{'─' * 115}\n  CORRELATION ANALYSIS RULE MINING — Profit (threshold={correlation_threshold})\n{'─' * 115}")
        validated_wfo_test = decorrelate_by_profit(
            strategy_trades_wfo_test = validated_wfo_test,
            initial_balance          = INITIAL_BALANCE,
            threshold                = correlation_threshold,
        )

    _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-CORRELATION")

    if show_plots:
        for rule_id, trades in validated_wfo_test:
            plot_filter_comparison(
                strategy_id        = f"{rule_id}_wfo_test",
                trades_df_baseline = trades,
                trades_df_r01      = None,
                data_folder        = data_folder,
                initial_balance    = INITIAL_BALANCE,
                regime_enabled     = False,
            )

        if validated_wfo_test:
            plot_portfolio_comparison(
                strategy_trades_baseline = validated_wfo_test,
                strategy_trades_regime01 = None,
                data_folder              = data_folder,
                initial_balance          = INITIAL_BALANCE,
                title                    = "Portfolio WFO Test — Validated only",
            )

    best_combo_ids = []
    if run_best_portfolio and validated_wfo_test:
        top_portfolios = find_best_portfolio_combination_wfo(
            validated_wfo_trades = validated_wfo_test,
            initial_balance      = INITIAL_BALANCE,
            show_plots           = show_plots,
        )
        if top_portfolios:
            best_combo_ids = list(top_portfolios[0]["combo"])

    if run_deploy and best_combo_ids:
        deploy_map  = {}
        label_width = max(len(rid) for rid in best_combo_ids) + 2

        for rule_id in best_combo_ids:
            rule_info = raw_by_id[rule_id]
            rule_tf   = rule_info["timeframe"]

            run_deploy_rule(
                rule_id             = rule_id,
                specs               = rule_info["specs"],
                side                = rule_info["side"],
                timeframe           = rule_tf,
                ohlcv_is            = ohlcv_data_by_timeframe[rule_tf],
                signal_fn           = rule_info["signal_fn"],
                param_grid          = param_grid,
                order_amount        = order_amount,
                n_symbols           = n_symbols,
                approved            = rule_info["approved"],
                dtype               = dtype,
                n_jobs              = inner_n_jobs,
                symbols_live_folder = symbols_live_folder,
                deploy_map          = deploy_map,
                label_width         = label_width,
            )

        _save_rule_deploy_batch(
            output_path = deploy_output_path,
            deploy_map  = deploy_map,
        )

    return validated_wfo_test


def _short_id(rule_id: str) -> str:
    parts = rule_id.split("_")
    return "_".join(parts[:3])

def _print_ranking(all_raw_results: list, highlight_ids: list, stage_label: str) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(highlight_ids)]
    rows.sort(key=lambda r: r["net_gain"], reverse=True)

    id_width = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2

    print(f"\n{'=' * 160}")
    print(f"  RULE MINING RESULTS — {stage_label} ── {len(rows)} / {len(all_raw_results)} tested")
    print(f"{'=' * 160}")
    print(f"{'ID':<{id_width}}{'SIDE':<6}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'WIN%':<8}{'PF':<8}{'CALMAR':<8}{'R2':<8}{'STAB':<8}{'TRADES':<8}RULE")
    print(f"{'-' * 160}")

    for r in rows:
        print(
            f"{_short_id(r['rule_id']):<{id_width}}{r['side']:<6}{r['net_gain']:<12.1f}{r['max_dd']:<10.1f}"
            f"{r['win_rate']:<8.1f}{r['profit_factor']:<8.2f}{r['calmar']:<8.2f}{r['r_squared']:<8.3f}{r['param_stability']:<8.3f}"
            f"{r['n_trades']:<8}{r['label']}"
        )

    print(f"{'=' * 160}\n")
    
def _print_min_by_group(all_raw_results: list, highlight_ids: list) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(highlight_ids)]
    if not rows:
        return

    metrics = ["net_gain", "max_dd", "win_rate", "profit_factor", "calmar", "r_squared", "param_stability", "n_trades"]
    groups  = {}
    for r in rows:
        key = (r["timeframe"], r["side"])
        groups.setdefault(key, []).append(r)

    print(f"\n{'=' * 115}")
    print(f"  MIN METRICS BY TIMEFRAME + SIDE ── {len(groups)} group(s)")
    print(f"{'=' * 115}")
    print(f"{'TIMEFRAME':<12}{'SIDE':<8}{'N':<6}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'WIN%':<8}{'PF':<8}{'CALMAR':<8}{'R2':<8}{'STAB':<8}{'TRADES':<8}")
    print(f"{'-' * 115}")

    for (tf, side), group_rows in sorted(groups.items()):
        mins = {m: min(r[m] for r in group_rows) for m in metrics}
        print(
            f"{tf:<12}{side:<8}{len(group_rows):<6}{mins['net_gain']:<12.1f}{mins['max_dd']:<10.1f}"
            f"{mins['win_rate']:<8.1f}{mins['profit_factor']:<8.2f}{mins['calmar']:<8.2f}"
            f"{mins['r_squared']:<8.3f}{mins['param_stability']:<8.3f}{mins['n_trades']:<8}"
        )

    print(f"{'=' * 115}\n")