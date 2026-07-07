import logging

from shared_batchs.pipeline.wfo import run_wfo_is
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.utils.plotting import plot_filter_comparison
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE
from shared_batchs.runs.run_correlation import decorrelate_by_profit
from shared_batchs.runs.run_best_wfo_portfolio import find_best_portfolio_combination_wfo

from shared_batchs.rule_mining.rule_generator import generate_all_rules, MAX_DEPTH

logger = logging.getLogger("BOT_batch.rule_mining.runner")

_OP_SLUG = {">": "gt", "<": "lt"}


def _slugify_label(label: str) -> str:
    slug = label
    for op, tag in _OP_SLUG.items():
        slug = slug.replace(op, tag)
    slug = slug.replace(" AND ", "_AND_").replace(" ", "_")
    slug = slug.replace("[", "").replace("]", "").replace("-", "m")
    return slug


def run_rule_mining(
    ohlcv_data: dict,
    timeframe: str,
    param_grid: dict,
    order_amount: int,
    net_gain_th: float,
    dd_th: float,
    dtype,
    data_folder: str,
    n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    max_depth: int = MAX_DEPTH,
    show_plots: bool = False,
    correlation_threshold: float = 0.75,
    run_correlation: bool = True,
    run_best_portfolio: bool = True,
) -> list:

    param_names    = list(param_grid.keys())
    lists_for_grid = [param_grid[k] for k in param_names]

    arr_sample = next(iter(ohlcv_data.values()))
    all_rules  = generate_all_rules({
        "open":  arr_sample["open"],
        "high":  arr_sample["high"],
        "low":   arr_sample["low"],
        "close": arr_sample["close"],
    }, max_depth=max_depth)

    logger.info(f"RULE MINING ── total candidate rules: {len(all_rules)}")

    results                  = []
    approved_wfo_test_trades = []

    for i, rule in enumerate(all_rules):
        rule_id = f"{i:04d}_{rule['side']}_{_slugify_label(rule['label'])}"

        best_params, approved_wfo, wfo_net_gain, wfo_max_dd, _, wfo_test_trades, _ = run_wfo_is(
            ohlcv_data          = ohlcv_data,
            param_names         = param_names,
            lists_for_grid      = lists_for_grid,
            signal_fn           = rule["signal_fn"],
            signal_params_keys  = [],
            order_amount        = order_amount,
            timeframe           = timeframe,
            net_gain_th         = net_gain_th,
            dd_th               = dd_th,
            dtype               = dtype,
            n_jobs              = n_jobs,
            show_progress       = show_progress,
            n_symbols           = n_symbols,
        )

        n_trades = 0 if wfo_test_trades is None else len(wfo_test_trades)

        metrics = None
        if wfo_test_trades is not None and not wfo_test_trades.empty:
            metrics = compute_metrics(wfo_test_trades, capital=INITIAL_BALANCE, name="")

        results.append({
            "rule_id":       rule_id,
            "side":          rule["side"],
            "label":         rule["label"],
            "approved":      approved_wfo,
            "net_gain":      wfo_net_gain,
            "max_dd":        wfo_max_dd,
            "n_trades":      n_trades,
            "win_rate":      metrics["Win_Rate"]      if metrics else 0.0,
            "profit_factor": metrics["Profit_Factor"] if metrics else 0.0,
            "calmar":        metrics["Calmar"]        if metrics else 0.0,
            "best_params":   best_params,
        })

        logger.info(f"[{i + 1}/{len(all_rules)}] {rule['side']:<5} {rule['label']} -> "
                    f"{'PASS' if approved_wfo else 'FAIL'} NetGain={wfo_net_gain:.1f}% DD={wfo_max_dd:.1f}%")

        if approved_wfo and wfo_test_trades is not None and not wfo_test_trades.empty:
            approved_wfo_test_trades.append((rule_id, wfo_test_trades))

            if show_plots:
                plot_filter_comparison(
                    strategy_id        = f"{rule_id}_wfo_test",
                    trades_df_baseline = wfo_test_trades,
                    trades_df_r01      = None,
                    data_folder        = data_folder,
                    initial_balance    = INITIAL_BALANCE,
                    regime_enabled     = False,
                )

    _print_ranking(results)

    validated_wfo_test = approved_wfo_test_trades

    if run_correlation and validated_wfo_test:
        logger.info(f"\n{'─' * 115}\n  CORRELATION ANALYSIS RULE MINING — Profit (threshold={correlation_threshold})\n{'─' * 115}")
        validated_wfo_test = decorrelate_by_profit(
            strategy_trades_wfo_test = validated_wfo_test,
            initial_balance          = INITIAL_BALANCE,
            threshold                = correlation_threshold,
        )

    if run_best_portfolio and validated_wfo_test:
        find_best_portfolio_combination_wfo(
            validated_wfo_trades = validated_wfo_test,
            initial_balance      = INITIAL_BALANCE,
            show_plots           = show_plots,
        )

    return results


def _print_ranking(results: list) -> None:
    approved = [r for r in results if r["approved"]]
    approved.sort(key=lambda r: r["net_gain"], reverse=True)

    print(f"\n{'=' * 130}")
    print(f"  RULE MINING RESULTS ── {len(approved)} approved / {len(results)} tested")
    print(f"{'=' * 130}")
    print(f"{'SIDE':<6}{'NET_GAIN%':<12}{'MAX_DD%':<10}{'WIN%':<8}{'PF':<8}{'CALMAR':<8}{'TRADES':<8}RULE")
    print(f"{'-' * 130}")

    for r in approved:
        print(
            f"{r['side']:<6}{r['net_gain']:<12.1f}{r['max_dd']:<10.1f}"
            f"{r['win_rate']:<8.1f}{r['profit_factor']:<8.2f}{r['calmar']:<8.2f}"
            f"{r['n_trades']:<8}{r['label']}"
        )

    print(f"{'=' * 130}\n")