#shared/shared_batchs/rule_mining/rule_runner.py
import logging
from shared_batchs.pipeline.wfo import pipe_wfo
from shared_batchs.pipeline.dsr import pipe_dsr
from shared_batchs.pipeline.montecarlo import pipe_montecarlo
from shared_batchs.pipeline.correlation import pipe_correlation
from shared_batchs.utils.plotting import plot_filter_comparison, plot_portfolio_comparison
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE
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


def _build_rule_id(i: int, timeframe: str, rule: dict) -> str:
    return f"{i:05d}_{timeframe}_{rule['side']}_{_slugify_label(rule['label'])}"

def _build_rule_dicts(ohlcv_data: dict, timeframe: str, max_depth: int) -> list:
    """Generate all candidate rules for one timeframe, each tagged with a
    unique rule_id — the schema every pipe (DSR, WFO, ...) expects."""
    arr_sample = next(iter(ohlcv_data.values()))
    all_rules  = generate_all_rules({
        "open":  arr_sample["open"],
        "high":  arr_sample["high"],
        "low":   arr_sample["low"],
        "close": arr_sample["close"],
        "volume_quote": arr_sample["volume_quote"],
    }, max_depth=max_depth)

    return [
        {
            "rule_id":   _build_rule_id(i, timeframe, rule),
            "timeframe": timeframe,
            "side":      rule["side"],
            "specs":     rule["specs"],
            "signal_fn": rule["signal_fn"],
            "label":     rule["label"],
        }
        for i, rule in enumerate(all_rules)
    ]
# =============================================================================
# ORCHESTRATOR 
# =============================================================================
def run_rule_mining(
    ohlcv_data_by_timeframe: dict,
    ohlcv_arr_by_timeframe: dict,
    timeframes: list,
    param_grid: dict,
    order_amount: int,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    dsr_th: float,
    run_dsr: bool = True,
    rules_n_jobs: int = 1,
    inner_n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    max_depth: int = MAX_DEPTH,
    log_level: int = logging.INFO,
    save_trades: bool = False,
    brief_trades_folder: str = None,
) -> list:
    # -------------------------------------------------------------------
    # PHASE A ── DSR, one timeframe at a time, ALL timeframes before moving on.
    # -------------------------------------------------------------------
    all_dsr_results = []
    for timeframe in timeframes:
        rules = _build_rule_dicts(ohlcv_data_by_timeframe[timeframe], timeframe, max_depth)
        logger.info(f"RULE MINING ── {timeframe} ── total candidate rules: {len(rules)}")

        dsr_results = pipe_dsr(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr_by_timeframe[timeframe],
            param_grid   = param_grid,
            order_amount = order_amount,
            dtype        = dtype,
            dsr_th       = dsr_th,
            enabled      = run_dsr,
            timeframe    = timeframe,
        )
        all_dsr_results.extend([{**r, **_empty_wfo_fields()} for r in dsr_results])

    passed_dsr_ids = {r["rule_id"] for r in all_dsr_results if r["passed_dsr"]}
    _print_ranking(all_dsr_results, list(passed_dsr_ids), "POST-DSR", survivor_ids=list(passed_dsr_ids), debug=True)

    # -------------------------------------------------------------------
    # PHASE B ── WFO, one timeframe at a time, only for rules that passed DSR.
    # -------------------------------------------------------------------
    wfo_by_id = {}
    for timeframe in timeframes:
        rules_this_tf = [r for r in all_dsr_results if r["timeframe"] == timeframe and r["passed_dsr"]]

        wfo_results = pipe_wfo(
            rules               = rules_this_tf,
            ohlcv_arr           = ohlcv_arr_by_timeframe[timeframe],
            param_grid          = param_grid,
            order_amount        = order_amount,
            timeframe           = timeframe,
            net_gain_th         = net_gain_th,
            dd_th               = dd_th,
            r2_th               = r2_th,
            wfr_th              = wfr_th,
            dtype               = dtype,
            enabled             = True,
            rules_n_jobs        = rules_n_jobs,
            inner_n_jobs        = inner_n_jobs,
            show_progress       = show_progress,
            n_symbols           = n_symbols,
            log_level           = log_level,
            save_trades         = save_trades,
            brief_trades_folder = brief_trades_folder,
        )
        wfo_by_id.update({r["rule_id"]: r for r in wfo_results})

    all_raw_results = [
        wfo_by_id[r["rule_id"]] if r["rule_id"] in wfo_by_id else r
        for r in all_dsr_results
    ]

    wfo_candidate_ids = list(wfo_by_id.keys())
    _print_ranking(all_raw_results, wfo_candidate_ids, "POST-WFO", survivor_ids=[r["rule_id"] for r in all_raw_results if r["approved"]])

    return all_raw_results


def _empty_wfo_fields() -> dict:
    """Placeholder WFO fields for rules not yet sent to WFO (e.g. filtered
    out by DSR, or before Phase B has run)."""
    return {
        "approved":        False,
        "net_gain":        0.0,
        "max_dd":          0.0,
        "n_trades":        0,
        "n_windows":       0,
        "win_rate":        0.0,
        "profit_factor":   0.0,
        "calmar":          0.0,
        "r_squared":       0.0,
        "wfr":             0.0,
        "best_params":     None,
        "wfo_test_trades": None,
    }


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
    pipeline_correlation: bool = True,
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

    _print_min_by_group(all_raw_results, [rid for rid, _ in validated_wfo_test])

    # -------------------------------------------------------------------
    # STAGE ── Correlation (portfolio construction: drop redundant rules)
    # -------------------------------------------------------------------
    if validated_wfo_test:
        candidates_before_corr = [rid for rid, _ in validated_wfo_test]
        rules_for_corr = [raw_by_id[rid] for rid in candidates_before_corr]
        survivors_rules = pipe_correlation(
            rules           = rules_for_corr,
            initial_balance = INITIAL_BALANCE,
            threshold       = correlation_threshold,
            enabled         = pipeline_correlation,
        )
        survivors_corr      = [r["rule_id"] for r in survivors_rules]
        validated_wfo_test  = [(rid, raw_by_id[rid]["wfo_test_trades"]) for rid in survivors_corr]
        _print_ranking(all_raw_results, candidates_before_corr, "POST-CORRELATION", survivor_ids=survivors_corr)
        _print_min_by_group(all_raw_results, survivors_corr)
    else:
        _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-CORRELATION")
    # -------------------------------------------------------------------
    # STAGE ── Montecarlo (bootstrap of executed trades — validates RISK)
    # -------------------------------------------------------------------
    if validated_wfo_test:
        candidates_before_mc = [rid for rid, _ in validated_wfo_test]
        rules_for_mc = [raw_by_id[rid] for rid in candidates_before_mc]
        mc_results = pipe_montecarlo(
            rules           = rules_for_mc,
            initial_balance = INITIAL_BALANCE,
            prob_ruin_th    = montecarlo_ruin_th,
            enabled         = pipeline_montecarlo,
        )
        for r in mc_results:
            raw_by_id[r["rule_id"]]["montecarlo_prob_ruin"] = r["montecarlo_prob_ruin"]

        validated_wfo_test = [
            (r["rule_id"], r["wfo_test_trades"]) for r in mc_results if r["passed_montecarlo"]
        ]
        survivors_mc = [rid for rid, _ in validated_wfo_test]
        _print_ranking(all_raw_results, candidates_before_mc, "POST-MONTECARLO", survivor_ids=survivors_mc)
        _print_min_by_group(all_raw_results, survivors_mc)
    else:
        _print_ranking(all_raw_results, [rid for rid, _ in validated_wfo_test], "POST-MONTECARLO")

    # -------------------------------------------------------------------
    # STAGE ── Multiverse / MCPT (log-return permutation — validates EDGE via p-value)
    # -------------------------------------------------------------------
    if validated_wfo_test:
        from shared_batchs.pipeline.multiverse import pipe_multiverse

        candidates_before_mv = [rid for rid, _ in validated_wfo_test]
        rules_for_mv = [raw_by_id[rid] for rid in candidates_before_mv]
        mv_results = pipe_multiverse(
            rules                   = rules_for_mv,
            ohlcv_data_by_timeframe = ohlcv_data_by_timeframe,
            param_grid              = param_grid,
            order_amount            = order_amount,
            net_gain_th             = net_gain_th,
            dd_th                   = dd_th,
            r2_th                   = r2_th,
            wfr_th                  = wfr_th,
            dtype                   = dtype,
            n_symbols               = n_symbols,
            p_value_th              = multiverse_p_value_th,
            enabled                 = pipeline_multiverse,
        )
        for r in mv_results:
            raw_by_id[r["rule_id"]]["multiverse_p_value"] = r["multiverse_p_value"]

        validated_wfo_test = [
            (r["rule_id"], r["wfo_test_trades"]) for r in mv_results if r["passed_multiverse"]
        ]
        survivors_mv = [rid for rid, _ in validated_wfo_test]
        _print_ranking(all_raw_results, candidates_before_mv, "POST-MULTIVERSE", survivor_ids=survivors_mv)
        _print_min_by_group(all_raw_results, survivors_mv)
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
            show_plots            = show_plots,
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


def _print_ranking(all_raw_results: list, candidate_ids: list, stage_label: str, survivor_ids: list = None, debug: bool = False) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(candidate_ids)]
    rows.sort(key=lambda r: r["rule_id"])

    show_status  = survivor_ids is not None
    survivor_set = set(survivor_ids) if show_status else None
    log_fn       = logger.debug if debug else logger.info

    id_width    = max((len(_short_id(r["rule_id"])) for r in rows), default=8) + 2
    label_width = max((len(r["label"]) for r in rows), default=8) + 2

    count_str = f"{len(survivor_ids)} / {len(candidate_ids)} passed" if show_status else f"{len(rows)} / {len(candidate_ids)} tested"

    log_fn(f"\n{'─' * 170}")
    log_fn(f"  RULE MINING RESULTS — {stage_label} ── {count_str}")
    log_fn(f"{'─' * 170}")

    status_header = f"  {'STATUS':<8}" if show_status else ""
    log_fn(
        f"{'ID':<{id_width}}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'PF':<8}{'CALMAR':<8}{'R2':<8}"
        f"{'DSR':<8}{'WFR':<8}{'MC_RUIN':<9}{'MV_PVAL':<9}{'TRADES':<8}{'RULE':<{label_width}}{status_header}"
    )
    log_fn(f"{'─' * 170}")
    for r in rows:
        status_cell = f"  {('✅' if r['rule_id'] in survivor_set else '❌'):<8}" if show_status else ""
        log_fn(
            f"{_short_id(r['rule_id']):<{id_width}}{r['net_gain']:<12.1f}{r['max_dd']:<10.1f}"
            f"{r['profit_factor']:<8.2f}{r['calmar']:<8.2f}{r['r_squared']:<8.3f}"
            f"{r.get('dsr', 0.0):<8.3f}{r['wfr']:<8.2f}{r.get('montecarlo_prob_ruin', 0.0):<9.1f}"
            f"{r.get('multiverse_p_value', 0.0):<9.3f}"
            f"{r['n_trades']:<8}{r['label']:<{label_width}}{status_cell}"
        )

    log_fn(f"{'─' * 170}\n")

def _print_min_by_group(all_raw_results: list, highlight_ids: list) -> None:
    rows = [r for r in all_raw_results if r["rule_id"] in set(highlight_ids)]
    if not rows:
        return
    threshold_metrics = ["net_gain", "max_dd", "r_squared", "dsr", "wfr"]
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
    logger.info(f"\n{'─' * 140}")
    logger.info(f"  MIN/MAX METRICS BY TIMEFRAME + SIDE ── {len(groups)} group(s)")
    logger.info(f"{'─' * 140}")
    logger.info(
        f"{'TIMEFRAME':<12}{'SIDE':<8}{'N':<6}"
        f"{'NET_GAIN% min/max':<22}{'MAX_DD% min/max':<20}{'R2 min/max':<16}"
        f"{'DSR min/max':<16}{'WFR min/max':<16}"
    )
    logger.info(f"{'─' * 140}")
    for (tf, side), group_rows in sorted(groups.items()):
        s = group_stats[(tf, side)]
        logger.info(
            f"{tf:<12}{side:<8}{len(group_rows):<6}"
            f"{f'{s['net_gain'][0]:.1f} / {s['net_gain'][1]:.1f}':<22}"
            f"{f'{s['max_dd'][0]:.1f} / {s['max_dd'][1]:.1f}':<20}"
            f"{f'{s['r_squared'][0]:.3f} / {s['r_squared'][1]:.3f}':<16}"
            f"{f'{s['dsr'][0]:.3f} / {s['dsr'][1]:.3f}':<16}"
            f"{f'{s['wfr'][0]:.2f} / {s['wfr'][1]:.2f}':<16}"
        )
    logger.info(f"{'─' * 140}")
    anchors = {key: max(group_rows, key=lambda r: r["net_gain"]) for key, group_rows in groups.items()}
    safe_net_gain = min(a["net_gain"]  for a in anchors.values())
    safe_max_dd   = max(abs(a["max_dd"]) for a in anchors.values())
    safe_r2       = min(a["r_squared"] for a in anchors.values())
    safe_dsr      = min(a["dsr"] for a in anchors.values())
    safe_wfr      = min(a["wfr"] for a in anchors.values())
    logger.info("\n  Anchor row per group (highest NET_GAIN, used to derive joint-safe thresholds):")
    for (tf, side), a in sorted(anchors.items()):
        logger.info(f"    {tf:<10}{side:<8}{a['rule_id']}")
    logger.info(
        f"\n  JOINT-SAFE THRESHOLDS (guaranteed ≥1 survivor per group, all conditions at once) ── "
        f"NET_GAIN>={safe_net_gain:.1f}  MAX_DD<={safe_max_dd:.1f}  R2>={safe_r2:.3f}  "
        f"DSR>={safe_dsr:.3f}  WFR>={safe_wfr:.2f}"
    )
    logger.info(f"{'─' * 140}\n")