#shared/shared_batchs/rule_mining/rule_runner.py
import os
import logging

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.pipeline.wfo import run_wfo_is, WFO_WINDOW_CONFIG
from shared_batchs.pipeline.montecarlo import pipe_montecarlo
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.plotting import plot_filter_comparison, plot_portfolio_comparison
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE
from shared_batchs.runs.run_correlation import decorrelate_by_profit
from shared_batchs.runs.run_portfolio import find_best_portfolio_combination_wfo

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
    wfr_th: float,
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

    (
        best_params, approved_wfo, wfo_net_gain, wfo_max_dd, _, wfo_test_trades, df_results, wfo_wfr,
        _window_best_params, _window_test_arrays, _window_test_start_ts,
    ) = run_wfo_is(
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
        wfr_th              = wfr_th,
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
        "wfr":              wfo_wfr,
        "sharpe":           metrics["Sharpe"] if metrics else np.nan,
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
    wfr_th: float,
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
                timeframe, net_gain_th, dd_th, r2_th, wfr_th, dtype, inner_n_jobs, show_progress, n_symbols,
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
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    data_folder: str,
    inner_n_jobs: int = -1,
    n_symbols: int = None,
    show_plots: bool = False,
    correlation_threshold: float = 0.75,
    run_correlation: bool = True,
    pipeline_montecarlo: bool = True,
    montecarlo_ruin_th: float = 5.0,
    pipeline_multiverse: bool = True,
    multiverse_p_value_th: float = 0.05,
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

    _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-WFO")
    _print_min_by_group(all_raw_results, [rid for rid, _ in validated_wfo_test])


    # -------------------------------------------------------------------
    # STAGE 3 ── Correlation (portfolio construction: drop redundant rules)
    # -------------------------------------------------------------------
    if run_correlation and validated_wfo_test:
        candidates_before_corr = [rid for rid, _ in validated_wfo_test]
        validated_wfo_test = decorrelate_by_profit(
            strategy_trades_wfo_test = validated_wfo_test,
            initial_balance          = INITIAL_BALANCE,
            threshold                = correlation_threshold,
        )
        survivors_corr = [rid for rid, _ in validated_wfo_test]
        _print_ranking(all_raw_results, candidates_before_corr, "POST-CORRELATION", survivor_ids=survivors_corr)
    else:
        _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-CORRELATION")

    # -------------------------------------------------------------------
    # STAGE 4 ── Montecarlo (bootstrap of executed trades — validates RISK)
    # -------------------------------------------------------------------
    if pipeline_montecarlo and validated_wfo_test:
        candidates_before_mc = [rid for rid, _ in validated_wfo_test]
        survivors = []
        for rule_id, trades in validated_wfo_test:
            approved_mc, prob_ruin = pipe_montecarlo(
                wfo_test_trades = trades,
                initial_balance = INITIAL_BALANCE,
                prob_ruin_th    = montecarlo_ruin_th,
            )
            raw_by_id[rule_id]["montecarlo_prob_ruin"] = prob_ruin
            if approved_mc:
                survivors.append((rule_id, trades))
        validated_wfo_test = survivors
        survivors_mc = [rid for rid, _ in validated_wfo_test]
        _print_ranking(all_raw_results, candidates_before_mc, "POST-MONTECARLO", survivor_ids=survivors_mc)
    else:
        _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-MONTECARLO")

    # -------------------------------------------------------------------
    # STAGE 5 ── Multiverse / MCPT (log-return permutation — validates EDGE via p-value)
    # -------------------------------------------------------------------
    if pipeline_multiverse and validated_wfo_test:
        from shared_batchs.pipeline.multiverse import pipe_multiverse

        candidates_before_mv = [rid for rid, _ in validated_wfo_test]
        survivors = []
        for rule_id, trades in validated_wfo_test:
            rule_info = raw_by_id[rule_id]
            approved_mv, p_value_mv = pipe_multiverse(
                ohlcv_data          = ohlcv_data_by_timeframe[rule_info["timeframe"]],
                timeframe           = rule_info["timeframe"],
                param_grid          = param_grid,
                signal_fn           = rule_info["signal_fn"],
                signal_params_keys  = [],
                order_amount        = order_amount,
                net_gain_th         = net_gain_th,
                dd_th               = dd_th,
                r2_th               = r2_th,
                wfr_th              = wfr_th,
                dtype               = dtype,
                n_symbols           = n_symbols,
                real_profit         = float(trades["profit"].sum()),
                p_value_th          = multiverse_p_value_th,
            )
            raw_by_id[rule_id]["multiverse_p_value"] = p_value_mv
            if approved_mv:
                survivors.append((rule_id, trades))
        validated_wfo_test = survivors
        survivors_mv = [rid for rid, _ in validated_wfo_test]
        _print_ranking(all_raw_results, candidates_before_mv, "POST-MULTIVERSE", survivor_ids=survivors_mv)
    else:
        _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-MULTIVERSE")

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

def _print_ranking(all_raw_results: list, candidate_ids: list, stage_label: str, survivor_ids: list = None) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(candidate_ids)]
    rows.sort(key=lambda r: r["net_gain"], reverse=True)

    show_status  = survivor_ids is not None
    survivor_set = set(survivor_ids) if show_status else None

    id_width    = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width = max((len(r["label"]) for r in rows), default=8) + 2

    count_str = f"{len(survivor_ids)} / {len(rows)} passed" if show_status else f"{len(rows)} / {len(all_raw_results)} tested"

    logger.info(f"\n{'─' * 160}")
    logger.info(f"  RULE MINING RESULTS — {stage_label} ── {count_str}")
    logger.info(f"{'─' * 160}")

    status_header = f"  {'STATUS':<8}" if show_status else ""
    logger.info(f"{'ID':<{id_width}}{'SIDE':<6}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'PF':<8}{'CALMAR':<8}{'R2':<8}{'WFR':<8}{'MC_RUIN':<9}{'MV_PVAL':<9}{'TRADES':<8}{'RULE':<{label_width}}{status_header}")
    logger.info(f"{'─' * 160}")

    for r in rows:
        status_cell = f"  {('✅' if r['rule_id'] in survivor_set else '❌'):<8}" if show_status else ""
        logger.info(
            f"{_short_id(r['rule_id']):<{id_width}}{r['side']:<6}{r['net_gain']:<12.1f}{r['max_dd']:<10.1f}"
            f"{r['profit_factor']:<8.2f}{r['calmar']:<8.2f}{r['r_squared']:<8.3f}"
            f"{r['wfr']:<8.2f}{r.get('montecarlo_prob_ruin', 0.0):<9.1f}"
            f"{r.get('multiverse_p_value', 1.0):<9.3f}"
            f"{r['n_trades']:<8}{r['label']:<{label_width}}{status_cell}"
        )

    logger.info(f"{'─' * 160}\n")

def _print_min_by_group(all_raw_results: list, highlight_ids: list) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(highlight_ids)]
    if not rows:
        return

    threshold_metrics = ["net_gain", "max_dd", "r_squared"]
    groups = {}
    for r in rows:
        key = (r["timeframe"], r["side"])
        groups.setdefault(key, []).append(r)

    group_stats = {}
    for key, group_rows in groups.items():
        group_stats[key] = {
            m: (min(r[m] for r in group_rows), max(r[m] for r in group_rows))
            for m in threshold_metrics
        }

    logger.debug(f"\n{'─' * 115}")
    logger.debug(f"  MIN/MAX METRICS BY TIMEFRAME + SIDE ── {len(groups)} group(s)")
    logger.debug(f"{'─' * 115}")
    logger.debug(
        f"{'TIMEFRAME':<12}{'SIDE':<8}{'N':<6}"
        f"{'NET_GAIN% min/max':<22}{'MAX_DD% min/max':<20}{'R2 min/max':<16}"
    )
    logger.debug(f"{'─' * 115}")

    for (tf, side), group_rows in sorted(groups.items()):
        s = group_stats[(tf, side)]
        logger.debug(
            f"{tf:<12}{side:<8}{len(group_rows):<6}"
            f"{f'{s['net_gain'][0]:.1f} / {s['net_gain'][1]:.1f}':<22}"
            f"{f'{s['max_dd'][0]:.1f} / {s['max_dd'][1]:.1f}':<20}"
            f"{f'{s['r_squared'][0]:.3f} / {s['r_squared'][1]:.3f}':<16}"
        )

    logger.debug(f"{'─' * 115}")


    anchors = {key: max(group_rows, key=lambda r: r["net_gain"]) for key, group_rows in groups.items()}

    safe_net_gain = min(a["net_gain"]  for a in anchors.values())  # higher better → min of anchors
    safe_max_dd   = max(abs(a["max_dd"]) for a in anchors.values())  # lower |dd| better → max of anchors
    safe_r2       = min(a["r_squared"] for a in anchors.values())  # higher better → min of anchors

    logger.debug("\n  Anchor row per group (highest NET_GAIN, used to derive joint-safe thresholds):")
    for (tf, side), a in sorted(anchors.items()):
        logger.debug(f"    {tf:<10}{side:<8}{a['rule_id']}")

    logger.debug(
        f"\n  JOINT-SAFE THRESHOLDS (guaranteed \u22651 survivor per group, all conditions at once) ── "
        f"NET_GAIN>={safe_net_gain:.1f}  MAX_DD<={safe_max_dd:.1f}  R2>={safe_r2:.3f}"
    )
    logger.debug(f"{'─' * 115}\n")