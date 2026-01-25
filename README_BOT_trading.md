BOT TRADING - DOCUMENTACIÓN TÉCNICA v2.7
Sistema de Trading Automatizado con PostgreSQL y Regime-Based Position Sizing

Versión: 2.7
Fecha: 2026-01-25
Python: 3.12
Framework: Flask + ccxt + Bitget API + PostgreSQL

📋 TABLA DE CONTENIDOS
PARTE 1: VISIÓN GENERAL

Introducción
Arquitectura
Stack Tecnológico

PARTE 2: POSTGRESQL INTEGRATION

Arquitectura de Datos
Estado del Bot
Trades y Analytics

PARTE 3: MARKET REGIME SYSTEM

Clasificación de Mercado
Matriz de Régimen Custom
Position Sizing Adaptativo

PARTE 4: COMPONENTES CORE

BotOrchestrator
Sistema de Estrategias
Dashboard Web

PARTE 5: CONFIGURACIÓN Y OPERACIÓN

Settings.py
Strategies Configuration
Ciclo de Vida

PARTE 6: REFERENCIA RÁPIDA

Troubleshooting
Comandos y Endpoints
Estructuras de Datos


PARTE 1: VISIÓN GENERAL
1. Introducción
1.1 ¿Qué es BOT_trading?
Sistema automatizado de trading en futuros de criptomonedas que opera 24/7. Gestiona múltiples estrategias en diferentes timeframes con position sizing adaptativo según condiciones de mercado.
Características principales:

18+ estrategias multi-timeframe (4H, 1H, 6Hutc)
Position sizing personalizado por estrategia
PostgreSQL como fuente de verdad
Estado persistente con alta disponibilidad
Dashboard web en tiempo real
Tracking completo de condiciones de mercado
Dual-write para redundancia (PostgreSQL + JSON/Excel)

1.2 Novedades v2.7
PostgreSQL Integration Completa:

Estado del bot (posiciones activas) en base de datos
Lectura primaria desde PostgreSQL con fallback a JSON
Dashboard consume directamente de PostgreSQL
Trades históricos en base de datos

Arquitectura Independiente:

PostgreSQL no depende de convenciones de nombres de archivos
Parámetro account_number explícito en toda la cadena
Sistema escalable para replicación/failover


2. Arquitectura
2.1 Estructura de Directorios
bitget/
├── BOT_trading/
│   ├── config/
│   │   ├── settings.py             # POSTGRES_CONFIG, DIRECTION_MATRIX
│   │   ├── strategies_00.py        # Estrategias Python cuenta 00
│   │   ├── strategies_E1.py        # Estrategias Python cuenta E1
│   │   └── strategies_01.py        # Estrategias Python cuenta 01
│   ├── core/
│   │   └── orchestrator.py         # Main loop + regime cache
│   ├── state/
│   │   ├── state_manager.py        # PostgreSQL primary + JSON fallback
│   │   └── candle_tracker.py       # Candle timeouts
│   ├── execution/
│   │   ├── position_tracker.py     # Add/remove positions
│   │   ├── order_manager.py        # TP/SL execution
│   │   └── trade_logger.py         # Dual-write trades
│   ├── market_regime/
│   │   ├── regime_classifier.py    # Detecta régimen/dirección
│   │   └── position_sizer.py       # Calcula multipliers
│   ├── api/
│   │   └── backend.py              # Dashboard Flask + PostgreSQL
│   └── validation/
│       └── validation_module.py    # Config + PostgreSQL validation
2.2 Flujo de Datos
Market Data → Regime Classifier
    ↓
Position Sizer (calcula multipliers)
    ↓
Strategy Processor (abre posiciones)
    ↓
Position Tracker (guarda market_direction)
    ↓
State Manager
├─ PostgreSQL (PRIMARY - source of truth)
└─ JSON (BACKUP - safety net)
    ↓
(Al cerrar posición)
Trade Logger
├─ PostgreSQL (analytics/dashboard)
└─ Excel (visualización rápida)
    ↓
Dashboard Backend (lee PostgreSQL)

3. Stack Tecnológico
3.1 Core

Python 3.12
PostgreSQL 14+ (persistencia primary)
Flask 3.x (dashboard)
ccxt (OHLCV data)
psycopg2 (PostgreSQL driver)

3.2 Librerías Analytics
LibreríaUsopandasProcesamiento datosnumpyArrays numéricosnoldsHurst exponentta (pandas_ta)ATRneurokit2Permutation Entropy
3.3 APIs
Bitget REST API:

Base: https://api.bitget.com
Auth: HMAC SHA256
Product: USDT-FUTURES

Endpoints:

POST /api/v2/mix/order/place
GET /api/v2/mix/position/all-position
GET /api/v2/mix/account/account


PARTE 2: POSTGRESQL INTEGRATION
4. Arquitectura de Datos
4.1 Filosofía
PostgreSQL = Source of Truth

Bot lee estado primero de PostgreSQL
Dashboard consume únicamente PostgreSQL
JSON/Excel como backup automático redundante

¿Por qué dual-write permanente?

Costo marginal: 50KB disco, <1ms latencia
Beneficios enormes: Disaster recovery, debugging, portabilidad
Arquitectura profesional: Zero single points of failure

4.2 Esquema de Tablas
Tabla bot_state:

account (TEXT, PRIMARY KEY)
state_data (JSONB)
updated_at (TIMESTAMP)

Tabla trades:

Histórico completo de trades
Incluye: market_direction, regime_family, multipliers
Índices en: strategy_id, opened_at, closed_at

4.3 Configuración
settings.py:
POSTGRES_CONFIG = {
    'dbname': 'bot_trading',
    'user': 'javi',
    'password': 'xxxx',
    'host': 'localhost',
    'port': 5432,
    'connect_timeout': 5
}
Validación automática en startup:

Conexión PostgreSQL disponible
Tablas existen
Columna state_data es tipo JSONB
Permisos correctos


5. Estado del Bot
5.1 Lectura: PostgreSQL Primary + JSON Fallback
Proceso en state_manager.load_state(account_number, state_file):

Intenta PostgreSQL primero:

Query: SELECT state_data FROM bot_state WHERE account = %s
Si existe: Reconstruye posiciones con tipos Decimal/datetime
Log: ✓ State loaded from PostgreSQL: N positions


Fallback a JSON si falla:

Lee bot_state_XX.json
Mismo procesamiento que PostgreSQL
Log: ✓ State loaded from JSON: N positions


Si ambos fallan:

Retorna estado vacío (posiciones = {}, candles = {})
Bot arranca limpio



Punto crítico: Bot SIEMPRE puede arrancar (con o sin PostgreSQL).
5.2 Escritura: Dual-Write (PostgreSQL + JSON)
Proceso en state_manager.save_state_local():
Siempre escribe a ambos:

JSON: Archivo local bot_state_XX.json
PostgreSQL: UPSERT en tabla bot_state

Comportamiento robusto:

Si JSON falla → Log error, continúa
Si PostgreSQL falla → Log error, continúa
Estado nunca se pierde (al menos uno escribe)

Logs (debug level):
[PG✓ JSON✓] State saved: 3 positions
5.3 Independencia de Archivos
CRÍTICO - Cambio arquitectónico v2.7:
ANTES (v2.6):

account_number extraído del nombre de archivo bot_state_01.json
PostgreSQL dependía de convención de nombres

DESPUÉS (v2.7):

account_number parámetro explícito en todas las funciones
PostgreSQL 100% independiente de archivos JSON
Escalabilidad mejorada (múltiples servidores, naming flexible)

Funciones modificadas:

load_state(account_number, state_file)
save_state_local(open_positions, strategy_candles, account_number, state_file)
sync_broker(..., account_number, state_file)
add_position(..., account_number, ...)
Todas las funciones de state/execution/strategies


6. Trades y Analytics
6.1 Dual-Write Trades
Excel (visualización rápida):

Archivo local TRADES_XX.xlsx
Útil para debugging rápido
Columna MARKET_DIRECTION incluida

PostgreSQL (analytics):

Tabla trades con todos los campos
Dashboard consulta directamente
Sin dependencia de archivos

6.2 Dashboard Integration
Backend lee únicamente PostgreSQL:

GET /api/trades → Query directo a tabla trades
GET /api/positions → Query a tabla bot_state
Sin lectura de Excel/JSON
Queries optimizadas con índices


PARTE 3: MARKET REGIME SYSTEM
7. Clasificación de Mercado
7.1 Métricas Calculadas
Sobre BTCUSDT en ventanas configurables:

Hurst Exponent (trending persistence)

H > 0.5 = trending
H < 0.5 = mean-reverting


Efficiency Ratio (quality direccional)

0 = lateral, 1 = tendencia perfecta


ATR Normalizado (volatilidad)

% sobre precio actual


Permutation Entropy (aleatoriedad)

0 = predecible, 1 = aleatorio



7.2 Reglas de Clasificación
Definidas en settings.py → REGIME_FAMILIES:
Orden first-match-wins:

TRENDING: if Hurst > 0.55 AND ER > 0.4
VOLATILE: elif ATR > 2.0% AND PE > 0.2
RANGING: else

7.3 Detección de Dirección
Basada en precio vs MA50:

BTC price > MA50 → market_direction = 'uptrend'
BTC price < MA50 → market_direction = 'dwtrend'

Se calcula 1 vez por vela cerrada y se cachea.
Logs incluyen contexto:
[REGIME] 1H: REGIME=TRENDING, DIRECTION=DWTREND 
(BTC=$91086.10, MA50=$93446.76, hurst=0.81, er=0.66)

8. Matriz de Régimen Custom
8.1 Sistema Bidimensional
Ajusta position sizing según:

Régimen de mercado: trending/ranging/volatile (detectado)
Dirección de mercado: uptrend/dwtrend (detectado)
Multipliers de estrategia: regime_trending/ranging/volatile (config)
Modo dirección: long_only/short_only/general (config)

Fórmula:
final_mult = regime_mult × direction_mult
adjusted_amount = base_amount × final_mult
8.2 Configuración Matrices
DIRECTION_MATRIX (settings.py):
long_only:  uptrend=1.0, dwtrend=0.0  # Bloquea longs en downtrend
short_only: uptrend=0.0, dwtrend=1.0  # Bloquea shorts en uptrend
general:    uptrend=1.0, dwtrend=1.0  # Sin filtro direccional
REGIME_GENERAL (fallbacks):
trending: 1.0
ranging: 1.0
volatile: 1.0
DIRECTION_GENERAL (fallbacks):
uptrend: 1.0
dwtrend: 1.0
8.3 Ejemplo Práctico
Estrategia Long en Downtrend:

Config: regime_trending=1.8, direction_mode='long_only', base=80 USDT
Mercado: TRENDING + DWTREND (BTC < MA50)
Cálculo:

regime_mult = 1.8
direction_mult = 0.0 (DIRECTION_MATRIX['long_only']['dwtrend'])
final_mult = 1.8 × 0.0 = 0.0
adjusted = 0 USDT


Resultado: Estrategia BLOQUEADA, no busca señales

Estrategia Short en Downtrend:

Config: regime_trending=1.0, direction_mode='short_only', base=80 USDT
Mercado: TRENDING + DWTREND
Cálculo:

regime_mult = 1.0
direction_mult = 1.0 (DIRECTION_MATRIX['short_only']['dwtrend'])
final_mult = 1.0
adjusted = 80 USDT


Resultado: Posición abierta, market_direction='dwtrend' guardado


9. Position Sizing Adaptativo
9.1 Flujo Completo
VELA CIERRA
    ↓
1. SYNC BROKER
    ↓
2. UPDATE REGIME + DIRECTION
   - Calcular métricas en BTCUSDT
   - Clasificar: trending/ranging/volatile
   - Detectar: uptrend/dwtrend (BTC vs MA50)
   - Cachear: regime_cache['1H'], direction_cache['1H']
    ↓
3. PARA CADA ESTRATEGIA:
   - PositionSizer.calculate_adjusted_amount()
     * regime_mult del config
     * direction_mult de DIRECTION_MATRIX
     * metadata incluye market_direction
   - Si blocked → Skip
   - Buscar señales con adjusted_amount
    ↓
4. SI ABRE POSICIÓN:
   - position_tracker.add_position()
     * Guarda market_direction en dict
   - state_manager.save_state()
     * PostgreSQL: UPSERT en bot_state
     * JSON: Escribe bot_state_XX.json
    ↓
5. AL CERRAR:
   - order_manager.close_position()
     * Extrae market_direction
   - trade_logger.log_closed_position()
     * PostgreSQL: INSERT en trades
     * Excel: Append a TRADES_XX.xlsx
9.2 Tracking Market Direction
Se guarda en TODA la cadena:

Cache en orchestrator (por timeframe)
Metadata de PositionSizer
Dict de posición en position_tracker
JSON en state_manager
PostgreSQL tabla bot_state
Excel columna MARKET_DIRECTION
PostgreSQL tabla trades

Resultado: Análisis histórico completo de condiciones de mercado.

PARTE 4: COMPONENTES CORE
10. BotOrchestrator
core/orchestrator.py - Cerebro del sistema
10.1 Responsabilidades

Main loop infinito
Cache de régimen/dirección por timeframe
Coordinación de estrategias
Delegación de sizing a PositionSizer
Sincronización con broker

10.2 Variables Clave
Estado operacional:

open_positions: Dict de posiciones abiertas
strategy_candles: Contadores de velas
strategies: Lista de estrategias (cargadas de Python config)

Caches (NO persisten):

regime_cache: {timeframe: régimen}
direction_cache: {timeframe: dirección}
position_sizer: Instancia de PositionSizer

10.3 Timing Crítico
¿CUÁNDO se calcula régimen/dirección?

Solo al cerrar vela
Una vez por timeframe
Antes de buscar señales
Después de sync broker

Secuencia temporal:
16:00:00 - Vela 1H cerró
16:00:01 - Sync broker
16:00:02 - Calcular regime + direction → cachear
16:00:03 - Para cada estrategia 1H:
           ├─ PositionSizer → metadata con market_direction
           ├─ Check si bloqueada
           └─ Buscar señales

11. Sistema de Estrategias
11.1 Configuración Python
Ubicación: config/strategies_XX.py
Ventajas vs YAML:

Type safety
IDE autocomplete
Syntax errors detectados inmediatamente
Sin problemas de indentación
Backward compatible

11.2 Estructura
Lista STRATEGIES con dicts, cada dict es una estrategia.
Campos clave:

Identificación: id, name
Ejecución: timeframe, direction, order_amount
Risk: tp_pct, sl_pct, sell_after_ncandles
Regime sizing: regime_trending/ranging/volatile
Direction filtering: direction_mode
Estado: active (True/False)
Específicos: lookback, tolerance, etc.

11.3 Validación Automática
En startup (validation/validation_module.py):

PostgreSQL connection disponible
Tablas existen
direction_mode válido
regime_trending/ranging/volatile >= 0
Coherencia direction/direction_mode
Parámetros requeridos presentes

Si falla → Shutdown con error claro

12. Dashboard Web
12.1 Endpoints
Backend Flask en api/backend.py
GET /api/regime/current?timeframe=1H

Retorna: regime, direction, BTC price, BTC MA50, métricas

GET /api/regime/matrix

Retorna: DIRECTION_MATRIX, REGIME_GENERAL, DIRECTION_GENERAL

GET /api/positions

Lee PostgreSQL tabla bot_state
Retorna: Posiciones con market_direction

GET /api/trades?limit=100

Lee PostgreSQL tabla trades
Retorna: Histórico con market_direction, regime_family

GET /api/status

Retorna: Estado general del bot

12.2 PostgreSQL Integration
Dashboard consume únicamente PostgreSQL:

Sin lectura de archivos JSON/Excel
Queries optimizadas con índices
Tiempo real (JSONB queries rápidas)


PARTE 5: CONFIGURACIÓN Y OPERACIÓN
13. Settings.py
config/settings.py - Configuración centralizada
13.1 PostgreSQL
POSTGRES_CONFIG: Dict con parámetros de conexión
13.2 Matrices
DIRECTION_MATRIX: Multipliers por direction_mode
REGIME_GENERAL: Fallbacks por régimen
DIRECTION_GENERAL: Fallbacks por dirección
13.3 Validación
STRATEGY_TYPE_REQUIRED_PARAMS: Parámetros por tipo
COMMON_REQUIRED_PARAMS: Parámetros obligatorios todas
Límites:

Order amount: 40-100 USDT
TP: 1.5-10%
SL: 1.5-15%
Candles: 49-51


14. Strategies Configuration
14.1 Archivos
Por cuenta:

strategies_00.py (18 estrategias)
strategies_E1.py (16 estrategias)
strategies_01.py (2 estrategias)

14.2 Carga Dinámica
strategies/strategy_loader.py:

load_strategies(account_number) → import dinámico
Sin recompilación Docker
Cambios instantáneos (restart bot)

14.3 Alta Nueva Estrategia
Checklist:

Crear función señal en signals/
Registrar en strategy_registry.py
Añadir dict a strategies_XX.py
Definir regime_trending/ranging/volatile
Definir direction_mode
Crear fichero símbolos
Validar en cuenta 01
Verificar logs


15. Ciclo de Vida
15.1 Startup
STARTUP
├─ Cargar config (settings + strategies Python)
├─ Validar PostgreSQL connection
├─ Validar estrategias
├─ Inicializar PositionSizer
├─ Conectar Bitget API
├─ Inicializar dashboard Flask
├─ load_state()
│  ├─ Try PostgreSQL first
│  └─ Fallback JSON if fails
└─ Calcular próximas velas
15.2 Main Loop
LOOP INFINITO:
1. Check vela cerrada
2. Si cerró:
   - Sync broker
   - Update regime + direction → cachear
   - Process estrategias
     * PositionSizer
     * Buscar señales
     * Si abre → save_state() dual-write
3. Si no:
   - Check TP/SL periódico
4. Sleep 50ms
15.3 Shutdown
SHUTDOWN
├─ save_state_local()
│  ├─ PostgreSQL: UPSERT bot_state
│  └─ JSON: Write bot_state_XX.json
├─ Cerrar conexiones
└─ Exit graceful

PARTE 6: REFERENCIA RÁPIDA
16. Troubleshooting
16.1 PostgreSQL Connection Failed
Síntoma: Bot no arranca, error validación PostgreSQL
Solución:
bash# Verificar servicio
sudo systemctl status postgresql

# Iniciar si parado
sudo systemctl start postgresql

# Verificar conexión
psql -U javi -d bot_trading

# Verificar tablas
\dt
16.2 Market Direction No Se Guarda
Síntoma: Posiciones sin market_direction o con 'unknown'
Diagnóstico:
bash# Logs
grep "market_direction\|DIRECTION" BOT_orchestator_XX.log

# PostgreSQL
psql -U javi -d bot_trading -c \
"SELECT market_direction FROM trades ORDER BY id DESC LIMIT 10;"
Solución: Verificar cadena completa (orchestrator → position_sizer → position_tracker → state_manager → trade_logger)
16.3 Estado No Carga de PostgreSQL
Síntoma: Logs muestran "Loading from JSON fallback"
Diagnóstico:
bash# Verificar dato existe
psql -U javi -d bot_trading -c \
"SELECT account FROM bot_state;"
Solución:

Verificar account_number correcto
Verificar tabla bot_state tiene dato
Revisar logs de error PostgreSQL

16.4 Estrategia Bloqueada
Síntoma: Logs "dir=dwtrend(0x) → BLOCKED"
Solución:

Esto es correcto si direction_mode='long_only' y market='dwtrend'
Para permitir con penalización: cambiar 0.0 a 0.5 en DIRECTION_MATRIX


17. Comandos y Endpoints
17.1 CLI
Iniciar:
bashpython3 main.py --account 00
python3 main.py --account E1
python3 main.py --account 01
Logs:
bash# Completos
tail -f persistence/bot_files_00/BOT_orchestator_00.log

# Filtrados
grep "PostgreSQL\|DIRECTION\|SIZING" BOT_orchestator_00.log
grep "REGIME" BOT_orchestator_00.log
grep "ERROR" BOT_orchestator_00.log
Verificar estado:
bash# PostgreSQL state
psql -U javi -d bot_trading -c \
"SELECT account, updated_at FROM bot_state;"

# PostgreSQL trades
psql -U javi -d bot_trading -c \
"SELECT symbol, market_direction, regime_family 
FROM trades ORDER BY id DESC LIMIT 10;"

# JSON backup
cat persistence/bot_state_00.json | python3 -m json.tool
```

### 17.2 API Endpoints

**Base:** `http://localhost:PUERTO`
- Cuenta 00: Puerto 5000
- Cuenta E1: Puerto 5001
- Cuenta 01: Puerto 5099

**Principales:**
- `GET /api/regime/current?timeframe=1H`
- `GET /api/regime/matrix`
- `GET /api/positions`
- `GET /api/trades?limit=100`
- `GET /api/status`

---

## 18. Estructuras de Datos

### 18.1 PositionSizer Metadata

**Retorna dict con:**
- base_amount
- market_regime (trending/ranging/volatile)
- market_direction (uptrend/dwtrend)
- regime_multiplier
- direction_multiplier
- final_multiplier
- adjusted_amount
- blocked (True/False)
- regime_source, direction_source

### 18.2 Position Dict

**Guardado en state:**
- strategy_id, symbol, size
- entry_price, direction
- opened_at, usdt_amount
- market_direction (uptrend/dwtrend)
- regime_family, regime_multiplier
- direction_multiplier
- TP/SL levels, order_id

### 18.3 Caches Orchestrator

**NO persisten, recalculan cada vela:**

**regime_cache:**
```
{'4H': 'ranging', '1H': 'trending', '6Hutc': 'volatile'}
```

**direction_cache:**
```
{'4H': 'dwtrend', '1H': 'uptrend', '6Hutc': 'uptrend'}
18.4 PostgreSQL Schemas
Tabla bot_state:

account (TEXT, PK)
state_data (JSONB)
updated_at (TIMESTAMP)

Tabla trades:

id, strategy_id, symbol
direction, entry_price, exit_price
opened_at, closed_at
market_direction (TEXT)
regime_family (TEXT)
regime_multiplier, direction_multiplier
profit_pct, usdt_amount


🎉 FIN DE DOCUMENTACIÓN
BOT_trading v2.7 - PostgreSQL Integration Complete

📝 RESUMEN EJECUTIVO v2.7
Cambios Principales
PostgreSQL Integration:

Estado del bot: PostgreSQL primary + JSON fallback
Trades: PostgreSQL + Excel dual-write
Dashboard: Consume únicamente PostgreSQL
Independencia total de nombres de archivos

Arquitectura:

account_number parámetro explícito en toda la cadena
Escalabilidad mejorada (multi-servidor ready)
Failover automático (fallback a JSON)
Zero single points of failure

Quick Start
bash# 1. PostgreSQL
sudo systemctl status postgresql

# 2. Config
# - settings.py: POSTGRES_CONFIG
# - strategies_XX.py: STRATEGIES lista

# 3. Start
python3 main.py --account 00

# 4. Verify
grep "PostgreSQL" BOT_orchestator_00.log
# Debe mostrar: "✓ State loaded from PostgreSQL"

# 5. Check data
psql -U javi -d bot_trading -c "SELECT * FROM bot_state;"
```

### Arquitectura Final
```
PostgreSQL (source of truth)
    ↓
├─ Bot: PostgreSQL primary + JSON fallback
├─ Dashboard: PostgreSQL direct
└─ Trades: PostgreSQL + Excel dual-write


