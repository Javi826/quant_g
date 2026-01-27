# 1. PostgreSQL
sudo systemctl status postgresql

# 2. Config
# - settings.py: POSTGRES_CONFIG, RISK_LIMITS
# - strategies_XX.py: STRATEGIES lista

# 3. Start
python3 main.py --account 00

# 4. Verify
grep "PostgreSQL\|RISK" BOT_orchestator_00.log
# Debe mostrar: "✓ State loaded from PostgreSQL"
# Debe mostrar: "[RISK] ✓" o "[RISK] ⛔ SKIP"

# 5. Check data
psql -U javi -d bot_trading -c "SELECT * FROM bot_state;"
psql -U javi -d bot_trading -c "SELECT * FROM exposure_history ORDER BY date DESC LIMIT 5;"
```

## Arquitectura Final
```
PostgreSQL (source of truth)
    ↓
├─ Bot: PostgreSQL primary + JSON fallback
├─ Dashboard: PostgreSQL direct
├─ Trades: PostgreSQL + Excel dual-write
└─ Exposure: PostgreSQL snapshots diarios ← NUEVO v2.8

Risk Control Flow:
Market Data → Regime Classifier → Position Sizer
    ↓
Risk Limiter (check exposure) ← NUEVO v2.8
    ↓
Strategy Processor (solo si pasa check)
