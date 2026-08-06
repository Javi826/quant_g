#shared/shared_batchs/rule_mining/rule_runner.py
import os
import logging
from shared_batchs.pipeline.wfo import pipe_wfo
from shared_config import VOLUME_COL
from shared_batchs.pipeline.backtest_runner import pipe_backtesting
from shared_batchs.pipeline.dsr import pipe_dsr
from shared_batchs.pipeline.montecarlo import pipe_montecarlo
from shared_batchs.pipeline.correlation import pipe_correlation
from shared_batchs.utils.plotting import plot_rule_mining_filter_comparison, plot_rule_mining_portfolio_comparison
from shared_batchs.setup.config_backtest import INITIAL_BALANCE
from shared_batchs.runs.run_portfolio import find_best_portfolio_combination_wfo
from shared_batchs.rule_mining.rule_generator import generate_all_rules, MAX_DEPTH
from shared_batchs.rule_mining.rule_deploy import run_deploy_rule, save_rule_deploy_batch
from shared_batchs.utils.reporting import print_rule_mining_ranking, print_rule_mining_min_by_group
from shared_batchs.pipeline.dsr import pipe_dsr, empty_dsr_fields
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

    arr_sample = next(iter(ohlcv_data.values()))
    all_rules  = generate_all_rules({
        "open":  arr_sample["open"],
        "high":  arr_sample["high"],
        "low":   arr_sample["low"],
        "close": arr_sample["close"],
        VOLUME_COL: arr_sample[VOLUME_COL],
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

def _empty_wfo_fields() -> dict:

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
# =============================================================================
# ORCHESTRATOR 
# =============================================================================

def run_rule_mining_pipeline(
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
    data_folder: str,
    run_dsr: bool = True,
    pipeline_wfo: bool = True,
    rules_n_jobs: int = 1,
    inner_n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    max_depth: int = MAX_DEPTH,
    log_level: int = logging.INFO,
    save_trades: bool = False,
    brief_trades_folder: str = None,
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
    run_config: dict = None,
) -> list:
    # -------------------------------------------------------------------
    # BACKTESTING — one timeframe at a time, ALL timeframes before moving on.
    # -------------------------------------------------------------------
    all_dsr_results = []
    for timeframe in timeframes:
        rules = _build_rule_dicts(ohlcv_data_by_timeframe[timeframe], timeframe, max_depth)
        logger.info(f"RULE MINING ── {timeframe} ── total candidate rules: {len(rules)}")

        raw_results, n_combos = pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr_by_timeframe[timeframe],
            param_grid   = param_grid,
            order_amount = order_amount,
            dtype        = dtype,
            timeframe    = timeframe,
        )

        if run_dsr:
            dsr_results = pipe_dsr(
                raw_results = raw_results,
                dsr_th      = dsr_th,
                n_combos    = n_combos,
                timeframe   = timeframe,
            )
        else:
            logger.info(f"DSR ── {timeframe} ── disabled — passing all rules through untouched")
            dsr_results = [{**r, **empty_dsr_fields()} for r in raw_results]

        del raw_results  # libera el universo bruto de este timeframe antes de pasar al siguiente

        all_dsr_results.extend([{**r, **_empty_wfo_fields()} for r in dsr_results])

    passed_dsr_ids = {r["rule_id"] for r in all_dsr_results if r["passed_dsr"]}
    print_rule_mining_ranking(all_dsr_results, list(passed_dsr_ids), "POST-DSR", survivor_ids=list(passed_dsr_ids), debug=True)
    # -------------------------------------------------------------------
    # WFO, one timeframe at a time, only for rules that passed DSR.
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
            enabled             = pipeline_wfo,
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
    print_rule_mining_ranking(all_raw_results, wfo_candidate_ids, "POST-WFO", survivor_ids=[r["rule_id"] for r in all_raw_results if r["approved"]])

    # -------------------------------------------------------------------
    # Portfolio construction on the flattened, cross-timeframe pool.
    # -------------------------------------------------------------------
    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    validated_after_wfo = [
        (r["rule_id"], r["wfo_test_trades"])
        for r in all_raw_results
        if r["approved"] and r["wfo_test_trades"] is not None and not r["wfo_test_trades"].empty
    ]

    print_rule_mining_min_by_group(all_raw_results, [rid for rid, _ in validated_after_wfo])

    # -------------------------------------------------------------------
    # Correlation (portfolio construction: drop redundant rules)
    # -------------------------------------------------------------------
    if validated_after_wfo:
        candidates_before_corr = [rid for rid, _ in validated_after_wfo]
        rules_for_corr = [raw_by_id[rid] for rid in candidates_before_corr]
        survivors_rules = pipe_correlation(
            rules           = rules_for_corr,
            initial_balance = INITIAL_BALANCE,
            threshold       = correlation_threshold,
            enabled         = pipeline_correlation,
        )
        survivors_corr              = [r["rule_id"] for r in survivors_rules]
        validated_after_correlation = [(rid, raw_by_id[rid]["wfo_test_trades"]) for rid in survivors_corr]
        print_rule_mining_ranking(all_raw_results, candidates_before_corr, "POST-CORRELATION", survivor_ids=survivors_corr)
        print_rule_mining_min_by_group(all_raw_results, survivors_corr)
    else:
        validated_after_correlation = validated_after_wfo
        print_rule_mining_ranking(all_raw_results, [rid for rid, _ in validated_after_correlation], "POST-CORRELATION")
    # -------------------------------------------------------------------
    # Montecarlo (bootstrap of executed trades — validates RISK)
    # -------------------------------------------------------------------
    if validated_after_correlation:
        candidates_before_mc = [rid for rid, _ in validated_after_correlation]
        rules_for_mc = [raw_by_id[rid] for rid in candidates_before_mc]
        mc_results = pipe_montecarlo(
            rules           = rules_for_mc,
            initial_balance = INITIAL_BALANCE,
            prob_ruin_th    = montecarlo_ruin_th,
            enabled         = pipeline_montecarlo,
        )
        for r in mc_results:
            raw_by_id[r["rule_id"]]["montecarlo_prob_ruin"] = r["montecarlo_prob_ruin"]

        validated_after_montecarlo = [
            (r["rule_id"], r["wfo_test_trades"]) for r in mc_results if r["passed_montecarlo"]
        ]
        survivors_mc = [rid for rid, _ in validated_after_montecarlo]
        print_rule_mining_ranking(all_raw_results, candidates_before_mc, "POST-MONTECARLO", survivor_ids=survivors_mc)
        print_rule_mining_min_by_group(all_raw_results, survivors_mc)
    else:
        validated_after_montecarlo = validated_after_correlation
        print_rule_mining_ranking(all_raw_results, [rid for rid, _ in validated_after_montecarlo], "POST-MONTECARLO")

    # -------------------------------------------------------------------
    # Multiverse / MCPT (log-return permutation — validates EDGE via p-value)
    # -------------------------------------------------------------------
    if validated_after_montecarlo:
        from shared_batchs.pipeline.multiverse import pipe_multiverse

        candidates_before_mv = [rid for rid, _ in validated_after_montecarlo]
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

        validated_after_multiverse = [
            (r["rule_id"], r["wfo_test_trades"]) for r in mv_results if r["passed_multiverse"]
        ]
        survivors_mv = [rid for rid, _ in validated_after_multiverse]
        print_rule_mining_ranking(all_raw_results, candidates_before_mv, "POST-MULTIVERSE", survivor_ids=survivors_mv)
        print_rule_mining_min_by_group(all_raw_results, survivors_mv)
    else:
        validated_after_multiverse = validated_after_montecarlo
        print_rule_mining_ranking(all_raw_results, [rid for rid, _ in validated_after_multiverse], "POST-MULTIVERSE")

    if show_plots:
        for rule_id, trades in validated_after_multiverse:
            plot_rule_mining_filter_comparison(
                strategy_id        = f"{rule_id}_wfo_test",
                trades_df_baseline = trades,
                trades_df_r01      = None,
                data_folder        = data_folder,
                initial_balance    = INITIAL_BALANCE,
                regime_enabled     = False,
            )

        if validated_after_multiverse:
            plot_rule_mining_portfolio_comparison(
                strategy_trades_baseline = validated_after_multiverse,
                strategy_trades_regime01 = None,
                data_folder              = data_folder,
                initial_balance          = INITIAL_BALANCE,
                title                    = "Portfolio WFO Test — Validated only",
            )

    top_portfolios = []
    if run_best_portfolio and validated_after_multiverse:
        top_portfolios = find_best_portfolio_combination_wfo(
            validated_wfo_trades = validated_after_multiverse,
            initial_balance      = INITIAL_BALANCE,
            show_plots            = show_plots,
        )

    if run_deploy and top_portfolios:
        for top_idx, portfolio in enumerate(top_portfolios, start=1):
            combo_ids = list(portfolio["combo"])
            if not combo_ids:
                continue

            deploy_map  = {}
            label_width = max(len(rid) for rid in combo_ids) + 2

            for rule_id in combo_ids:
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

            save_rule_deploy_batch(
                output_path = _build_top_output_path(deploy_output_path, top_idx),
                deploy_map  = deploy_map,
                run_config  = run_config,
            )

    return validated_after_multiverse, all_dsr_results

def _build_top_output_path(base_path: str, top_idx: int) -> str:
    """Insert a _topN suffix before the file extension, e.g. rules_batch.py -> rules_batch_top1.py."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_top{top_idx}{ext}"
