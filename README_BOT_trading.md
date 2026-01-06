# 🤖 BOT TRADING - DOCUMENTACIÓN TÉCNICA v2.0

**Sistema de Trading Automatizado Multi-Estrategia**

---

**Versión:** 2.0  
**Fecha:** 2026-01-06  
**Python:** 3.12  
**Framework:** Flask + ccxt + Bitget API  

---

## 📋 TABLA DE CONTENIDOS

### PARTE 1: VISIÓN GENERAL
1. [Introducción al Sistema](#1-introducción-al-sistema)
2. [Arquitectura General](#2-arquitectura-general)
3. [Stack Tecnológico](#3-stack-tecnológico)

### PARTE 2: COMPONENTES CORE
4. [Estructura de Directorios](#4-estructura-de-directorios)
5. [BotOrchestrator (core/)](#5-botaniquestrator-core)
6. [Sistema de Estrategias ⭐](#6-sistema-de-estrategias-)
7. [Execution: Cliente Bitget](#7-execution-cliente-bitget)
8. [Dashboard Web](#8-dashboard-web)
9. [Funciones de Señales](#9-funciones-de-señales)

### PARTE 3: CONFIGURACIÓN
10. [settings.py - Configuración Central](#10-settingspy---configuración-central)
11. [strategies.yaml - Definiciones](#11-strategiesyaml---definiciones)
12. [strategy_registry.py - Implementaciones ⭐](#12-strategy_registrypy---implementaciones-)

### PARTE 4: LÓGICA DE NEGOCIO
13. [Detección de Señales](#13-detección-de-señales)
14. [Gestión de Posiciones](#14-gestión-de-posiciones)
15. [TP/SL y Timeout](#15-tpsl-y-timeout)
16. [Sincronización con Broker](#16-sincronización-con-broker)

### PARTE 5: FLUJOS CRÍTICOS
17. [Ciclo de Vida del Bot](#17-ciclo-de-vida-del-bot)
18. [Flujos de Operación](#18-flujos-de-operación)

### PARTE 6: DESARROLLO
19. [Añadir Nueva Estrategia ⭐](#19-añadir-nueva-estrategia-)
20. [Sistema de Logging](#20-sistema-de-logging)
21. [Troubleshooting](#21-troubleshooting)

### PARTE 7: REFERENCIA RÁPIDA
22. [Catálogo de Estrategias](#22-catálogo-de-estrategias)
23. [Estructuras de Datos](#23-estructuras-de-datos)
24. [Variables de Estado](#24-variables-de-estado)

---

# PARTE 1: VISIÓN GENERAL

## 1. Introducción al Sistema

### 1.1 ¿Qué es BOT_trading?

BOT_trading es un sistema automatizado de trading en futuros de criptomonedas que opera 24/7 sin intervención humana. Gestiona múltiples estrategias simultáneamente en diferentes timeframes (4H, 1H, 6Hutc, 2m, 5m) y soporta operación en múltiples cuentas independientes.

### 1.2 Características Principales

**Trading:**
- 14 estrategias diferentes implementadas
- Gestión automática de Take Profit (TP) y Stop Loss (SL)
- Timeout automático por número de velas
- Sincronización continua con el broker
- Soporte para posiciones LONG y SHORT

**Arquitectura:**
- Configuración declarativa en YAML
- Separación clara de responsabilidades entre módulos
- Sistema de logging dual (consola + archivo con rotación)
- Estado persistente para recuperación tras crashes
- Validación exhaustiva de configuración al arrancar

**Monitoreo:**
- Dashboard web en tiempo real con Flask
- API REST para integración externa
- Logs streaming en vivo
- Métricas de rendimiento
- Histórico de trades en Excel

### 1.3 Flujo de Alto Nivel

```
INICIALIZACIÓN
├─ Cargar configuración (settings.py)
├─ Cargar estrategias (strategies.yaml)
├─ Conectar a Bitget (API + WebSocket)
└─ Recuperar estado anterior (bot_state_XX.json)

MAIN LOOP (infinito)
├─ ¿Nueva vela cerrada?
│  ├─ SÍ → Procesar estrategias
│  │       ├─ Sync con broker
│  │       ├─ Incrementar candles
│  │       ├─ Check timeout
│  │       └─ Detectar señales
│  └─ NO → Continuar
└─ Cada 10s: Verificar TP/SL

SEÑAL DETECTADA
├─ Verificar balance
├─ Calcular size y TP/SL
├─ Enviar orden market
├─ Registrar posición
└─ Guardar estado

TRACKING POSICIÓN
├─ Verificar TP (cada 10s)
├─ Verificar SL (cada 10s)
├─ Verificar timeout (cada nueva vela)
└─ Cerrar si se alcanza cualquier condición
```

---

## 2. Arquitectura General

### 2.1 Diagrama de Componentes

```
BOT_trading SYSTEM
│
├─ main.py (Entry Point)
│   └─ Parse args, crear BotOrchestrator
│
├─ BotOrchestrator (core/orchestrator.py)
│   ├─ Inicialización del sistema
│   ├─ Gestión del ciclo de vida
│   ├─ Coordinación de estrategias
│   ├─ Control de timeframes
│   └─ Main loop
│
├─ Sistema de Estrategias ⭐
│   ├─ strategies.yaml (configuración)
│   ├─ strategy_registry.py (elif explícito)
│   └─ strategy_loader.py (carga YAML)
│
├─ Execution Manager (execution/bitget_client.py)
│   ├─ Comunicación con Bitget API
│   ├─ Autenticación HMAC
│   └─ Manejo de órdenes
│
├─ Dashboard (api/backend.py)
│   ├─ Servidor Flask
│   ├─ API REST endpoints
│   └─ Templates HTML
│
├─ Funciones de Señales (signals/)
│   ├─ Z_add_signals_double_top.py
│   ├─ Z_add_signals_reversal.py
│   ├─ Z_add_signals_parity.py
│   └─ Z_add_signals_orderblocks.py
│
└─ Persistence (JSON + Excel)
    ├─ bot_state_XX.json (posiciones + candles)
    ├─ bot_trades_XX.xlsx (histórico)
    └─ BOT_orchestator_XX.log (logs)
```

### 2.2 Flujo de Datos

```
Market Data (Bitget)
    ↓
WebSocket/REST Data Fetcher
    ↓
OHLCV Data (DataFrame)
    ↓
Signal Detection (strategy functions)
    ↓
Strategy Processor (orchestrator)
    ↓
├─ Signal detected? → Open position → Track position
├─ Position tracking → Check TP/SL/Timeout
└─ Update state → Save JSON
```

### 2.3 Separación de Responsabilidades

| Módulo | Responsabilidad | Archivos Clave |
|--------|-----------------|----------------|
| **core/** | Orquestación y ciclo de vida | orchestrator.py |
| **strategies/** | Carga y validación de estrategias | strategy_loader.py, strategy_registry.py |
| **signals/** | Funciones de detección | Z_add_signals_*.py |
| **execution/** | Comunicación con broker | bitget_client.py |
| **api/** | Dashboard web | backend.py, templates/ |
| **config/** | Configuración central | settings.py |
| **persistence/** | Estado y logs | bot_state_XX.json, logs/ |
| **validation/** | Validación de configuración | strategy_validator.py |
| **bot_utils/** | Utilidades (logger, etc.) | logger.py |

---

## 3. Stack Tecnológico

### 3.1 Lenguajes y Frameworks

- **Python 3.12:** Lenguaje principal con async/await
- **Flask 3.x:** Framework web para dashboard y API REST
- **ccxt:** Librería unificada para exchanges (usado para OHLCV)

### 3.2 Librerías Clave

| Librería | Uso |
|----------|-----|
| ccxt | Obtención de datos OHLCV |
| requests | HTTP requests a Bitget API |
| pandas | Procesamiento de datos |
| numpy | Arrays numéricos |
| yaml | Parsing de configuración |
| Flask | Dashboard web |
| logging | Sistema de logs |

### 3.3 Formato de Datos

- **YAML (strategies.yaml):** Definición declarativa de estrategias
- **JSON (bot_state_XX.json):** Estado persistente (posiciones + candles)
- **Excel (bot_trades_XX.xlsx):** Histórico de trades
- **Logs (.log files):** Rotación automática, formato dual

### 3.4 APIs Externas

**Bitget API:**
- Base URL: `https://api.bitget.com`
- Autenticación: HMAC SHA256
- Product Type: USDT-FUTURES (futuros perpetuos)

**Endpoints principales:**
- `POST /api/v2/mix/order/place` - Colocar orden
- `GET /api/v2/mix/position/all-position` - Obtener posiciones
- `GET /api/v2/mix/account/account` - Obtener balance
- `GET /api/v2/mix/market/candles` - OHLCV histórico

---

# PARTE 2: COMPONENTES CORE

## 4. Estructura de Directorios

### 4.1 Árbol Completo

```
BOT_trading/
│
├── main.py                              # Entry point del sistema
│
├── config/                              # Configuración
│   ├── __init__.py
│   └── settings.py                      # Settings centralizados
│
├── core/                                # Componentes core
│   ├── __init__.py
│   └── orchestrator.py                  # BotOrchestrator (cerebro)
│
├── strategies/                          # Sistema de estrategias ⭐
│   ├── __init__.py
│   ├── strategies.yaml                  # Definiciones YAML
│   ├── strategy_registry.py             # Elif explícito ⭐
│   ├── strategy_loader.py               # Cargador de YAML
│   └── strategy_processor.py            # Procesador de señales
│
├── signals/                             # Funciones de señales
│   ├── __init__.py
│   ├── Z_add_signals_double_top.py
│   ├── Z_add_signals_reversal.py
│   ├── Z_add_signals_parity.py
│   └── Z_add_signals_orderblocks.py
│
├── execution/                           # Ejecución de órdenes
│   ├── __init__.py
│   └── bitget_client.py                 # Cliente API Bitget
│
├── api/                                 # Dashboard web
│   ├── __init__.py
│   ├── backend.py                       # Flask server + API
│   └── templates/
│       └── dashboard.html
│
├── analytics/                           # Métricas
│   ├── __init__.py
│   └── metrics.py
│
├── persistence/                         # Datos persistentes
│   ├── bot_files_00/                    # Cuenta 00
│   │   ├── BOT_orchestator_00.log
│   │   ├── bot_trades_00.xlsx
│   │   └── bot_state_00.json
│   ├── bot_files_E1/                    # Cuenta E1
│   └── bot_files_01/                    # Cuenta 01
│
├── bot_utils/                           # Utilidades
│   ├── __init__.py
│   └── logger.py
│
├── validation/                          # Validaciones
│   ├── __init__.py
│   └── strategy_validator.py
│
└── utils/
    └── ZZ_connect.py                    # Credenciales (privado)
```

### 4.2 Propósito de Cada Directorio

| Directorio | Propósito | Archivos Importantes |
|------------|-----------|---------------------|
| **config/** | Configuración central | settings.py |
| **core/** | Lógica central de orquestación | orchestrator.py |
| **strategies/** | Sistema de estrategias | strategies.yaml, strategy_registry.py |
| **signals/** | Implementación de funciones de señales | Z_add_signals_*.py |
| **execution/** | Comunicación con broker | bitget_client.py |
| **api/** | Dashboard web y API REST | backend.py, dashboard.html |
| **persistence/** | Estado, logs, trades | JSON, Excel, logs |

---

## 5. BotOrchestrator (core/)

### 5.1 Responsabilidades

El **BotOrchestrator** es el componente central que:

1. Inicializa todo el sistema
2. Coordina las estrategias activas
3. Gestiona el ciclo de vida del bot
4. Controla el main loop infinito
5. Sincroniza con el broker periódicamente

### 5.2 Clase BotOrchestrator

**Constructor (`__init__`):**
- Recibe: account_number, bitget_client, connect_bitget_func, active_strategy_ids
- Inicializa: configuración de cuenta, rutas de archivos, estado global
- Prepara: OPEN_POSITIONS, STRATEGY_CANDLES, strategies, symbols_by_strategy

**Variables de Estado:**
- `self.OPEN_POSITIONS`: Dict de posiciones abiertas por estrategia
- `self.STRATEGY_CANDLES`: Dict de contadores de velas por estrategia
- `self.strategies`: Lista de estrategias cargadas desde YAML
- `self.strategies_by_timeframe`: Estrategias agrupadas por timeframe
- `self.symbols_by_strategy`: Símbolos asignados a cada estrategia

### 5.3 Método `run()`

El método `run()` es el corazón del sistema con dos fases:

**FASE 1: INICIALIZACIÓN**
1. Cargar estrategias desde YAML usando `load_strategies()`
2. Aplicar filtro de `--set-active` si se proporcionó
3. Validar configuración con `validate_strategy_configuration()`
4. Cargar símbolos para cada estrategia con `load_final_symbols()`
5. Agrupar estrategias por timeframe con `group_strategies_by_timeframe()`
6. Inicializar dashboard con `DashboardServer`
7. Cargar estado previo con `load_state()`
8. Sincronizar con broker usando `sync_broker()`
9. Calcular próximas velas con `calculate_next_candle_time()`

**FASE 2: MAIN LOOP**
- Loop infinito con `while True`
- Check 1: Detectar velas cerradas (comparar `now >= next_candle_times[tf]`)
- Si vela cerrada: llamar a `_process_timeframe(tf)`
- Check 2: Verificación periódica TP/SL (cada 10 segundos)
- Sleep de 0.05s para evitar spin de CPU

### 5.4 Método `_process_timeframe()`

Procesa todas las estrategias de un timeframe tras cerrar vela:

**Workflow:**
1. Log: "New {timeframe} candle closed"
2. Ejecutar `sync_broker()` para reconciliar estado
3. Para cada estrategia del timeframe:
   - Si tiene posiciones:
     - Incrementar contador de velas con `increment_strategy_candles()`
     - Verificar timeout con `check_candles_timeout_for_strategy()`
   - Si NO tiene posiciones:
     - Buscar señales con `process_strategy()`
4. Recalcular próxima vela con `calculate_next_candle_time()`

### 5.5 Inicialización desde main.py

**main.py** es el entry point:
- Parsea argumentos: `--account` (requerido), `--set-active` (opcional)
- Mapea clientes Bitget y funciones ccxt por cuenta
- Parsea estrategias activas si se proporcionó `--set-active`
- Crea instancia de `BotOrchestrator`
- Llama a `orchestrator.run()`
- Maneja Ctrl+C para shutdown graceful

---

## 6. Sistema de Estrategias ⭐

### 6.1 Arquitectura del Sistema

El sistema de estrategias separa **configuración** (YAML) de **implementación** (Python):

```
strategies.yaml (configuración)
    ↓
strategy_loader.py (carga YAML)
    ↓
strategy_validator.py (validación)
    ↓
strategy_registry.py (elif explícito) ⭐
    ↓
strategy_processor.py (ejecución)
    ↓
Z_add_signals_*.py (implementación)
```

### 6.2 strategies.yaml - Configuración Declarativa

**Ubicación:** `strategies/strategies.yaml`

**Contenido:** Lista de estrategias con parámetros en formato YAML.

**Parámetros obligatorios por estrategia:**
- `id`: Identificador único (formato `NN_nombre`)
- `name`: Nombre de función (debe existir en registry)
- `timeframe`: Timeframe de operación (4H, 1H, 6Hutc, etc.)
- `active`: true/false para habilitar/deshabilitar
- `direction`: 'long' o 'short'
- `sell_after_ncandles`: Timeout en número de velas
- `order_amount`: USDT por posición (40-100)
- `tp_pct`: Take profit en porcentaje (1.5-10)
- `sl_pct`: Stop loss en porcentaje (1.5-10)

**Parámetros específicos por tipo de estrategia:**
- Double top: `lookback`, `tolerance`, `trend_th`
- Reversal: `lookback`, `tolerance`, `ma_period`
- Parity: `lookback`, `tolerance`, `ma_period`
- Order blocks: `lookback`, `tolerance`, `impulse`

### 6.3 strategy_registry.py - Elif Explícito ⭐

**Ubicación:** `strategies/strategy_registry.py`

**Propósito:** Registro de estrategias con estructura elif visible.

**Función principal: `detect_signals_for_strategy()`**

Esta función es el corazón de la detección de señales. Su estructura:

1. Extrae el nombre de estrategia del dict `strat`
2. Descarga OHLCV para todos los símbolos usando ccxt
3. Para cada símbolo:
   - Normaliza OHLCV con `normalize_live_ohlcv()`
   - Convierte a arrays con `df_to_arrays_live()`
   - Ejecuta **elif explícito** por estrategia:
     - `if strategy_name == 'double_top_long_4H': signals = double_top_long(...)`
     - `elif strategy_name == 'reversal_long_4H': signals = reversal_long(...)`
     - `elif strategy_name == 'parity_short_4H': signals = parity_short(...)`
     - ... (un elif por cada estrategia implementada)
     - `else: logger.warning("Strategy not implemented")`
   - Verifica si `signals[-1] != 0` (señal en última vela)
   - Si hay señal, añade símbolo a lista de resultados
4. Retorna lista de símbolos con señal detectada

**Función auxiliar: `get_implemented_strategies()`**

Retorna un set con los nombres de todas las estrategias implementadas:
- Definido como simple set con todos los nombres
- Usado por el sistema de validación para verificar que estrategias en YAML existen
- Debe actualizarse al añadir nueva estrategia

**Constante: `IMPLEMENTED_STRATEGIES`**

Se define como `IMPLEMENTED_STRATEGIES = get_implemented_strategies()` para compatibilidad con código existente.

**¿Por qué elif explícito?**

- **Visibilidad:** Todas las estrategias visibles en un solo lugar
- **Simplicidad:** No hay magia de registry lookup
- **Mantenibilidad:** Fácil añadir estrategias (copy-paste elif)
- **Debugging:** Stack traces claros sin indirección
- **Copy-paste friendly:** Duplicar elif, cambiar nombre y parámetros

### 6.4 strategy_loader.py - Carga de YAML

**Funciones principales:**

- `load_strategies_from_yaml(yaml_path)`: Lee archivo YAML y retorna lista de estrategias
- `filter_strategies_by_ids(strategies, strategy_ids)`: Filtra estrategias por IDs específicos
- `load_strategies(strategy_ids)`: Función principal que carga YAML, filtra por IDs, valida existencia
- `group_strategies_by_timeframe(strategies)`: Agrupa estrategias por timeframe en dict

### 6.5 strategy_processor.py - Procesamiento

**Clase: `StrategyProcessor`**

**Método principal: `detect_signals(strategy, symbols, exchange)`**

Este método:
1. Obtiene función de señal desde `strategy_registry.detect_signals_for_strategy()`
2. Para cada símbolo en la lista:
   - Descarga OHLCV usando `exchange.fetch_ohlcv()`
   - Convierte a formato esperado por función
   - Ejecuta función de señal
   - Si señal positiva en última vela, añade símbolo a lista
3. Retorna lista de símbolos con señal detectada

**Método auxiliar: `_convert_to_array(ohlcv)`**

Convierte formato OHLCV de ccxt (lista de listas) a formato dict de arrays numpy esperado por funciones de señales.

---

## 7. Execution: Cliente Bitget

### 7.1 Clase BitgetClient

**Ubicación:** `execution/bitget_client.py`

**Constructor:**
- Recibe: api_key, api_secret, api_passphrase
- Inicializa: base_url de Bitget

### 7.2 Autenticación HMAC

**Método: `_sign_request(timestamp, method, request_path, body)`**

Implementa firma HMAC SHA256 según especificación de Bitget:
1. Construye mensaje: `timestamp + method + request_path + body`
2. Firma con HMAC-SHA256 usando secret
3. Codifica resultado en Base64

**Headers requeridos:**
- ACCESS-KEY: API key
- ACCESS-SIGN: Firma HMAC
- ACCESS-TIMESTAMP: Timestamp en milisegundos
- ACCESS-PASSPHRASE: Passphrase
- Content-Type: application/json

### 7.3 Métodos Principales

**`place_order(symbol, side, size, product_type)`**
- Coloca orden market
- Side: 'buy' (long) o 'sell' (short)
- Retorna: orderId, priceAvg, size del response

**`close_position(symbol, size, side, product_type)`**
- Cierra posición existente
- Side: 'buy' (para cerrar short) o 'sell' (para cerrar long)
- tradeSide: 'close'

**`get_all_positions(product_type)`**
- Obtiene todas las posiciones abiertas
- Retorna: lista de posiciones con symbol, holdSide, total, openPriceAvg, etc.

**`get_balance(product_type)`**
- Obtiene balance disponible en USDT
- Retorna: float con USDT disponible

**`get_all_symbols()`**
- Obtiene lista de todos los símbolos disponibles
- Retorna: lista de strings con nombres de símbolos

### 7.4 Manejo de Errores

**Códigos comunes:**
- `00000`: Success
- `40005`: Invalid API key
- `40014`: Insufficient balance
- `43025`: Position not exist

**Retry logic:** Implementado con backoff exponencial para errores recuperables (rate limits, timeouts).

---

## 8. Dashboard Web

### 8.1 DashboardServer

**Ubicación:** `api/backend.py`

**Clase: `DashboardServer`**

Constructor recibe:
- account_number, base_dir, get_current_price_func, get_balance_func
- strategies_config, initial_capital, implemented_strategies, symbols_by_strategy

Inicializa:
- Flask app
- Rutas de archivos (state_file, trades_file, log_file)
- Registra routes con `_register_routes()`

### 8.2 Endpoints API

**Principales endpoints:**

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Sirve dashboard.html |
| `/api/health` | GET | Health check |
| `/api/status` | GET | Estado completo del bot |
| `/api/positions` | GET | Posiciones activas |
| `/api/logs/stream` | GET | Logs nuevos (incremental) |
| `/api/trades` | GET | Histórico de trades desde Excel |

**`/api/status`:**
- Carga estado desde JSON
- Obtiene balance actual
- Calcula profit total
- Agrupa posiciones por símbolo
- Retorna info de estrategias con badges (ACTIVE, DEPRECATING, NOT IMPLEMENTED)

**`/api/logs/stream`:**
- Lee archivo de log desde última posición conocida
- Filtra códigos ANSI y limpia líneas
- Retorna solo líneas nuevas
- Actualiza posición para próxima lectura

### 8.3 Template HTML

**Ubicación:** `api/templates/dashboard.html`

**Componentes:**
- Header con stats (Balance, Profit, Profit %)
- Tabla de posiciones (Symbol, Direction, Size, Strategies)
- Tabla de estrategias (ID, Status, Positions, Symbols)
- Container de logs con auto-scroll

**Actualización:**
- Polling cada 2s con JavaScript
- `updateDashboard()` llama a `/api/status`
- `updateLogs()` llama a `/api/logs/stream`

---

## 9. Funciones de Señales

### 9.1 Estructura General

Todas las funciones en `signals/Z_add_signals_*.py` siguen esta estructura:

**Firma típica:**
- Parámetros: `ohlcv_array, lookback, tolerance, [parámetros específicos], live_trading`
- Retorna: Array numpy de señales (0 = no señal, 1 = señal)

**Lógica:**
1. Extraer arrays de close, high, low del ohlcv_array
2. Inicializar array de señales a ceros
3. Loop sobre velas desde `lookback` hasta `len(close)`
4. Aplicar lógica de detección del patrón
5. Si patrón detectado, asignar `signals[i] = 1`
6. Si `live_trading=True`, retornar solo `signals[-1:]` (última vela)
7. Si `live_trading=False`, retornar array completo

### 9.2 Funciones Implementadas

**Double Top (`Z_add_signals_double_top.py`):**
- `double_top_long()`: Detecta patrón de doble techo para entrar long

**Reversal (`Z_add_signals_reversal.py`):**
- `reversal_long()`: Detecta reversión alcista
- `reversal_short()`: Detecta reversión bajista

**Parity (`Z_add_signals_parity.py`):**
- `parity_long()`: Detecta condición de paridad para long
- `parity_short()`: Detecta condición de paridad para short

**Order Blocks (`Z_add_signals_orderblocks.py`):**
- `orderblocks_long()`: Detecta order block alcista
- `orderblocks_short()`: Detecta order block bajista

### 9.3 Parámetros Comunes

- **lookback:** Número de velas hacia atrás para análisis
- **tolerance:** Tolerancia de precio en porcentaje
- **ma_period:** Período de media móvil (reversal, parity)
- **trend_th:** Threshold de tendencia (double_top)
- **impulse:** Parámetro de impulso (order blocks)
- **live_trading:** Si True, retorna solo última señal

---

# PARTE 3: CONFIGURACIÓN

## 10. settings.py - Configuración Central

### 10.1 Exchange Settings

- `BASE_URL`: URL base de Bitget API
- `PRODUCT_TYPE`: "USDT-FUTURES"
- `MARGIN_MODE`: "crossed"
- `MARGIN_COIN`: "USDT"

### 10.2 General Bot Settings

- `HOUR_ZONE`: ZoneInfo('UTC') para timestamps
- `CHECK_INTERVAL`: 10 segundos entre checks de TP/SL
- `USE_HARDCODED_SIGNALS`: False (usar señales reales)
- `DISPLAY_MODE`: "summary"

### 10.3 Account-Specific Settings

**Constante: `ACCOUNTS`**

Dict con configuración de cada cuenta:

| Cuenta | Capital | Puerto | Descripción |
|--------|---------|--------|-------------|
| 00 | 3671 USDT | 5000 | Main Account |
| E1 | 1761 USDT | 5001 | Elite Account |
| 01 | 117 USDT | 5099 | Testing Account |

### 10.4 Strategy Assignment per Account

**Constante: `ACCOUNT_STRATEGIES`**

Dict que mapea cuentas a listas de strategy IDs:

- **Cuenta 00:** 14 estrategias (todas las validadas)
- **Cuenta E1:** 11 estrategias (subset optimizado, excluye 05, 12, 14)
- **Cuenta 01:** 2 estrategias (testing)

**Razones para separar estrategias por cuenta:**
- Capital diferente → estrategias diferentes
- Rendimiento/riesgo → excluir estrategias malas
- Testing → cuenta 01 solo para nuevas estrategias
- Flexibilidad operativa

### 10.5 Validation Settings

- `MIN_ORDER_AMOUNT`: 40 USDT
- `MAX_ORDER_AMOUNT`: 100 USDT
- `MIN_TP_PCT`: 1.5%
- `MAX_TP_PCT`: 10%
- `MIN_SL_PCT`: 1.5%
- `MAX_SL_PCT`: 10%
- `MIN_CANDLES`: 49
- `MAX_CANDLES`: 51
- `VALID_TIMEFRAMES`: ['1H', '4H', '6Hutc', '2m', '5m', '15m', '30m']

### 10.6 Helper Functions

**`get_account_config(account_number)`**
- Retorna configuración completa de cuenta
- Incluye paths de archivos (base_dir, log_file, state_file, trades_file)

**`get_account_strategies(account_number)`**
- Retorna lista de strategy IDs asignados a cuenta

---

## 11. strategies.yaml - Definiciones

### 11.1 Estructura de Archivo

Archivo YAML con lista de estrategias bajo clave `strategies`.

### 11.2 Ejemplo de Estrategia Completa

Cada estrategia es un dict con:

**Campos obligatorios:**
- `id`: '01_double_top_long_4H'
- `name`: 'double_top_long_4H'
- `timeframe`: '4H'
- `active`: true
- `direction`: 'long'
- `sell_after_ncandles`: 50
- `order_amount`: 40

**Parámetros específicos:**
- `lookback`: 2
- `tolerance`: 15
- `trend_th`: 5 (para double_top)

**TP/SL:**
- `tp_pct`: 4
- `sl_pct`: 10

### 11.3 Estrategias por Timeframe

**4H Timeframe:** 7 estrategias
- 01_double_top_long_4H
- 02_reversal_long_4H
- 03_parity_long_4H
- 04_reversal_short_4H
- 05_parity_short_4H
- 13_orderblocks_short_4H
- 14_orderblocks_long_4H

**1H Timeframe:** 4 estrategias
- 06_reversal_long_1H
- 07_reversal_short_1H
- 10_parity_long_1H
- 11_parity_short_1H

**6Hutc Timeframe:** 3 estrategias
- 08_reversal_long_6Hutc
- 09_reversal_short_6Hutc
- 12_parity_long_6Hutc

### 11.4 Parámetros por Tipo de Estrategia

**Double Top:**
- lookback, tolerance, trend_th

**Reversal:**
- lookback, tolerance, ma_period

**Parity:**
- lookback, tolerance, ma_period

**Order Blocks:**
- lookback, tolerance, impulse

---

## 12. strategy_registry.py - Implementaciones ⭐

### 12.1 Imports

El archivo comienza con imports de todas las funciones de señales desde módulo `signals`:
- `from signals.Z_add_signals_double_top import double_top_long`
- `from signals.Z_add_signals_reversal import reversal_long, reversal_short`
- `from signals.Z_add_signals_parity import parity_long, parity_short`
- `from signals.Z_add_signals_orderblocks import orderblocks_long, orderblocks_short`

### 12.2 Función detect_signals_for_strategy()

**Propósito:** Detectar señales para una estrategia usando estructura elif explícita.

**Parámetros:**
- `strat`: Dict con configuración de estrategia
- `final_symbols`: Lista de símbolos a analizar
- `exchange`: Objeto ccxt exchange
- `use_hardcoded`: Si True, retorna señales hardcoded (testing)

**Workflow:**
1. Extraer `strategy_name` del dict `strat`
2. Fetch OHLCV data para todos los símbolos
3. Para cada símbolo:
   - Normalizar OHLCV
   - Convertir a arrays
   - **Ejecutar elif explícito por estrategia:**
     - Bloque `if strategy_name == 'double_top_long_4H':`
     - Bloques `elif strategy_name == 'reversal_long_4H':` (uno por estrategia)
     - Bloque `else:` para estrategias no implementadas
   - Verificar señal en última vela
   - Si hay señal, añadir a lista
4. Retornar lista de símbolos con señal

**Estructura elif:**
- Un bloque por cada estrategia implementada
- Cada bloque llama a su función correspondiente con parámetros extraídos de `strat`
- Parámetros incluyen: lookback, tolerance, parámetros específicos, live_trading=True

### 12.3 Función get_implemented_strategies()

**Propósito:** Retornar set de todas las estrategias implementadas.

**Retorna:** Set con strings de nombres de estrategias:
- 'double_top_long_4H'
- 'reversal_long_4H'
- 'reversal_short_4H'
- ... (14 estrategias en total)

**Uso:** Sistema de validación verifica que estrategias en YAML existen en este set.

**IMPORTANTE:** Al añadir nueva estrategia, actualizar este set.

### 12.4 Constante IMPLEMENTED_STRATEGIES

Se define como: `IMPLEMENTED_STRATEGIES = get_implemented_strategies()`

Para compatibilidad con código existente que importa esta constante.

---

# PARTE 4: LÓGICA DE NEGOCIO

## 13. Detección de Señales

### 13.1 Workflow Completo

```
Nueva vela cerrada (ej: 4H)
  ↓
Obtener estrategias de ese timeframe
  ↓
Para cada estrategia:
  ↓
¿Tiene posiciones abiertas?
├─ SÍ: Skip (no buscar nuevas)
└─ NO:
    ↓
  Obtener símbolos para esta estrategia
    ↓
  Para cada símbolo:
    ↓
  ├─ Descargar OHLCV (ccxt)
  ├─ Convertir a arrays numpy
  ├─ Ejecutar función de señal
  └─ ¿Señal positiva? (signals[-1] != 0)
      ├─ SÍ: Abrir posición
      └─ NO: Continuar
```

### 13.2 Funciones Involucradas

**`detect_signal_for_strategy(strategy, symbols, exchange)`**
- Crea instancia de `StrategyProcessor`
- Llama a `processor.detect_signals()`
- Retorna lista de símbolos con señal

**`process_strategy(strat, final_symbols, exchange, ...)`**
- Llama a `detect_signal_for_strategy()`
- Para cada símbolo con señal:
  - Verificar balance suficiente
  - Llamar a `open_position_for_signal()`
  - Break tras abrir primera posición (una por estrategia)

---

## 14. Gestión de Posiciones

### 14.1 Apertura de Posición

**Función: `open_position_for_signal(strategy, symbol, exchange, ...)`**

Workflow:
1. Obtener precio actual con `get_current_price(symbol)`
2. Calcular size con `calculate_position_size(symbol, order_amount, current_price)`
3. Calcular TP/SL:
   - Long: TP = entry × (1 + tp_pct/100), SL = entry × (1 - sl_pct/100)
   - Short: TP = entry × (1 - tp_pct/100), SL = entry × (1 + sl_pct/100)
4. Enviar orden con `send_request_func(symbol, side, size)`
5. Parsear response: order_id, fill_price, fill_size
6. Crear dict de posición con todos los datos
7. Añadir a `OPEN_POSITIONS[strat_id]`
8. Inicializar `STRATEGY_CANDLES[strat_id] = 0`
9. Guardar estado con `save_state()`

**Función: `calculate_position_size(symbol, order_amount, current_price)`**
- Formula: `size = order_amount / current_price`
- Redondear a precisión del símbolo (típicamente 4 decimales)

### 14.2 Registro de Posición

Cada posición es un dict con:
- symbol, size, entry_price, direction
- tp, sl (precios absolutos)
- order_id, opened_at (timestamp ISO)
- usdt_amount

---

## 15. TP/SL y Timeout

### 15.1 Verificación TP/SL

**Función: `check_tp_sl_for_strategy(strategy_id, positions, send_request_func)`**

Workflow:
1. Para cada posición en lista:
   - Obtener precio actual
   - Calcular si hit TP o SL:
     - Long: TP hit si `current >= tp`, SL hit si `current <= sl`
     - Short: TP hit si `current <= tp`, SL hit si `current >= sl`
   - Si hit alguno:
     - Calcular profit
     - Llamar a `close_position()`
     - Llamar a `log_trade_to_excel()`
     - Remover de lista
     - Log resultado

**Función: `close_position(position, send_request_func, close_reason)`**

Workflow:
1. Construir orden de cierre:
   - Side opuesto: Long→sell, Short→buy
   - tradeSide: 'close'
2. Ejecutar con API de Bitget
3. Log confirmación

### 15.2 Timeout por Velas

**Función: `increment_strategy_candles(strategy_id, strategy_candles, ...)`**
- Solo incrementa si hay posiciones abiertas
- Incrementa contador en 1
- Guarda estado

**Función: `check_candles_timeout_for_strategy(strategy_id, max_candles, ...)`**

Workflow:
1. Obtener contador actual
2. Si `current_candles < max_candles`: return
3. Si `current_candles >= max_candles`:
   - Para cada posición:
     - Calcular profit
     - Cerrar con `close_position()`
     - Log con reason='TIMEOUT'
     - Remover de lista
   - Reset contador a 0
   - Guardar estado

**Propósito del timeout:**
- Prevenir posiciones abiertas indefinidamente
- Liberar capital para nuevas oportunidades
- Gestión de riesgo temporal

---

## 16. Sincronización con Broker

### 16.1 Propósito

Reconciliar estado local (JSON) con estado real en Bitget:
- Detectar posiciones cerradas externamente (usuario o liquidación)
- Actualizar tracking con datos reales
- Prevenir errores por estado desincronizado

### 16.2 Función sync_broker()

**Parámetros:**
- `open_positions`: Dict local de posiciones
- `strategy_candles`: Dict de contadores
- `state_file`: Path al JSON
- `bitget_client`: Cliente API

**Workflow:**
1. Obtener posiciones reales con `bitget_client.get_all_positions()`
2. Crear set de (symbol, direction) de posiciones reales
3. Para cada posición local:
   - Verificar si existe en set real
   - Si NO existe:
     - Log warning "Position closed externally"
     - Remover de tracking local
     - Reset contador de estrategia
4. Si hubo cambios, guardar estado

**Cuándo se ejecuta:**
- Cada vez que cierra una vela (antes de detectar señales)
- Garantiza que solo se buscan señales si realmente no hay posiciones

---

# PARTE 5: FLUJOS CRÍTICOS

## 17. Ciclo de Vida del Bot

### 17.1 Diagrama Completo

```
INICIO
├─ python3 main.py --account 00
├─ Parse argumentos
├─ Seleccionar cliente Bitget
└─ Crear BotOrchestrator

INICIALIZACIÓN (orchestrator.run())
├─ Cargar estrategias desde YAML
├─ Aplicar filtro --set-active
├─ Validar configuración
├─ Cargar símbolos por estrategia
├─ Agrupar por timeframe
├─ Inicializar dashboard Flask
├─ Cargar estado previo (JSON)
├─ Sync con broker
└─ Calcular próximas velas

MAIN LOOP (infinito)
├─ Detectar velas cerradas
│  └─ Procesar estrategias
├─ Cada 10s: Verificar TP/SL
└─ Sleep 0.05s

SHUTDOWN (Ctrl+C)
├─ Capturar señal
├─ Guardar estado final
└─ Exit
```

### 17.2 Estados del Bot

**STARTING:**
- Cargando configuración
- Validando estrategias
- Conectando a Bitget

**RUNNING:**
- Main loop activo
- Procesando señales
- Verificando TP/SL

**STOPPING:**
- Guardando estado
- Cerrando conexiones

---

## 18. Flujos de Operación

### 18.1 Detección de Vela Cerrada

```
Timeframe: 4H
now = 2026-01-04 20:00:00 UTC
next_candle_time = 2026-01-04 20:00:00 UTC

now >= next_candle_time?
├─ SÍ:
│  ├─ Log: "New 4H candle closed"
│  ├─ Sync con broker
│  ├─ Para cada estrategia 4H:
│  │  └─ Procesar
│  └─ Recalcular: next_candle_time = 2026-01-05 00:00:00
└─ NO: Continuar loop
```

### 18.2 Apertura de Posición

```
Señal detectada: BTCUSDT, estrategia 01_double_top_long_4H

VERIFICACIONES
└─ Balance >= order_amount?
   ├─ SÍ: Continuar
   └─ NO: Skip

CÁLCULOS
├─ Precio actual: 91167.7
├─ Size: 40 / 91167.7 = 0.0004
├─ TP: 91167.7 × 1.04 = 94814.4
└─ SL: 91167.7 × 0.90 = 82051.0

ORDEN MARKET
└─ Bitget API: place_order()

TRACKING
├─ Crear position dict
├─ Añadir a OPEN_POSITIONS[strat_id]
├─ Init STRATEGY_CANDLES[strat_id] = 0
└─ Guardar estado (JSON)
```

### 18.3 Cierre de Posición

```
TP/SL Check (cada 10s)

Posición: BTCUSDT LONG
Entry: 91167.7, TP: 94814.4, SL: 82051.0
Current: 94850.0

VERIFICACIÓN
└─ current >= TP?
   ├─ SÍ: Hit TP
   └─ NO: Check SL

CALCULAR PROFIT
└─ profit_pct = (94850 - 91167.7) / 91167.7 × 100 = 4.04%
└─ profit_usd = 0.0404 × 40 = 1.62$

CERRAR ORDEN
└─ Bitget API: close_position()

LOG Y ACTUALIZAR
├─ log_trade_to_excel()
├─ Remover de OPEN_POSITIONS
├─ Guardar estado
└─ Log: "Position closed (TP) - Profit: 1.62$ (4.04%)"
```

---

# PARTE 6: DESARROLLO

## 19. Añadir Nueva Estrategia ⭐

### 19.1 Workflow Simplificado

**PASO 1: Crear función de señal**
- Ubicación: `signals/Z_add_signals_mi_estrategia.py`
- Implementar función con firma: `mi_estrategia_long(ohlcv_array, lookback, threshold, live_trading)`
- Retornar array de señales

**PASO 2: Añadir elif en strategy_registry.py**
- Abrir: `strategies/strategy_registry.py`
- Añadir import: `from signals.Z_add_signals_mi_estrategia import mi_estrategia_long`
- Añadir bloque elif dentro de `detect_signals_for_strategy()`:
  - `elif strategy_name == 'mi_estrategia_long_4H':`
  - Llamar a función con parámetros de `strat`

**PASO 3: Añadir a get_implemented_strategies()**
- En mismo archivo `strategy_registry.py`
- Dentro de función `get_implemented_strategies()`
- Añadir string `'mi_estrategia_long_4H'` al set

**PASO 4: Definir en strategies.yaml**
- Abrir: `strategies/strategies.yaml`
- Añadir nueva estrategia al final de la lista:
  - id: '15_mi_estrategia_long_4H'
  - name: 'mi_estrategia_long_4H'
  - timeframe, active, direction, etc.
  - Parámetros específicos: lookback, threshold
  - tp_pct, sl_pct

**PASO 5: Asignar a cuenta de testing**
- Abrir: `config/settings.py`
- En dict `ACCOUNT_STRATEGIES`
- Añadir '15_mi_estrategia_long_4H' a lista de cuenta "01"

**PASO 6: Probar**
- Comando: `python3 main.py --account 01 --set-active 15_mi_estrategia_long_4H`
- Verificar logs: `tail -f persistence/bot_files_01/BOT_orchestator_01.log`

### 19.2 Archivos a Modificar

**Total: 3 archivos**
1. `signals/Z_add_signals_mi_estrategia.py` (nuevo)
2. `strategies/strategy_registry.py` (3 cambios: import + elif + set)
3. `strategies/strategies.yaml` (1 estrategia nueva)
4. `config/settings.py` (opcional, asignar a cuenta)

### 19.3 Checklist de Validación

- ✅ Función creada en signals/
- ✅ Import añadido en strategy_registry.py
- ✅ Elif añadido en detect_signals_for_strategy()
- ✅ Nombre añadido en get_implemented_strategies()
- ✅ Definida en strategies.yaml con todos los parámetros
- ✅ TP/SL en rangos válidos (1.5-10%)
- ✅ Asignada a cuenta de testing
- ✅ Bot arranca sin errores
- ✅ Señales se detectan correctamente

---

## 20. Sistema de Logging

### 20.1 Configuración

**Ubicación:** `bot_utils/logger.py`

**Función principal: `setup_logger()`**

Parámetros:
- `log_dir`: Directorio de logs
- `logfile_name`: Nombre del archivo
- `console_level`: Nivel para consola (default: INFO)
- `file_level`: Nivel para archivo (default: DEBUG)
- `max_bytes`: Tamaño máximo antes de rotar (10MB)
- `backup_count`: Número de backups (5)

**Configuración dual:**
1. **Console handler:**
   - Formato limpio (solo mensaje)
   - Para humanos en terminal
   
2. **File handler:**
   - Formato detallado (timestamp + módulo + línea + mensaje)
   - Rotación automática (RotatingFileHandler)
   - Para debugging

### 20.2 Uso en Módulos

En cada módulo Python:
- Import: `import logging`
- Obtener logger: `logger = logging.getLogger('BOT_trading.core')`
- Usar niveles:
  - `logger.debug()`: Detalles muy específicos
  - `logger.info()`: Eventos importantes
  - `logger.warning()`: Advertencias no críticas
  - `logger.error()`: Errores que afectan funcionalidad
  - `logger.critical()`: Errores graves

### 20.3 Rotación de Logs

**Automática:**
- Al alcanzar 10MB, archivo actual se renombra a `.log.1`
- Se crea nuevo archivo `.log`
- Mantiene 5 backups (`.log.1` a `.log.5`)
- El más antiguo se elimina

---

## 21. Troubleshooting

### 21.1 Bot No Arranca

**Síntoma:** Error al ejecutar `python3 main.py --account 00`

**Diagnóstico:**
- Verificar Python: `python3 --version`
- Verificar virtualenv activo
- Probar imports: `python3 -c "import flask, ccxt, yaml"`

**Soluciones:**
- Activar virtualenv
- Reinstalar dependencias
- Verificar permisos de archivos

### 21.2 Estrategia No Carga

**Síntoma:** Error "Strategy not found in YAML"

**Diagnóstico:**
- Ver IDs en YAML: `grep "id:" strategies/strategies.yaml`
- Verificar asignación en settings.py

**Solución:**
- Verificar ID en YAML existe
- Verificar ID en ACCOUNT_STRATEGIES
- Verificar nombre en IMPLEMENTED_STRATEGIES

### 21.3 Dashboard No Carga

**Síntoma:** `http://localhost:5000` no responde

**Diagnóstico:**
- Verificar bot corriendo: `pgrep -f "main.py --account 00"`
- Verificar puerto: `netstat -tlnp | grep 5000`
- Probar health: `curl http://localhost:5000/api/health`

**Solución:**
- Verificar bot en ejecución
- Verificar puerto libre
- Verificar logs de Flask

### 21.4 Posiciones No Cierran

**Síntoma:** Posición no cierra en TP/SL

**Diagnóstico:**
- Ver logs de TP/SL checks
- Ver estado actual: `cat bot_state_XX.json`
- Verificar precio actual alcanza TP/SL

**Solución:**
- Verificar cálculos de TP/SL
- Verificar función `check_tp_sl_for_strategy()` se ejecuta
- Verificar precio actual desde Bitget

### 21.5 Error de Conexión API

**Síntoma:** "Invalid API key" o "Invalid signature"

**Diagnóstico:**
- Verificar credenciales en `utils/ZZ_connect.py`
- Verificar formato de firma HMAC

**Solución:**
- Regenerar API keys en Bitget
- Verificar passphrase correcto
- Verificar timestamp sincronizado

---

# PARTE 7: REFERENCIA RÁPIDA

## 22. Catálogo de Estrategias

### 22.1 Resumen por Timeframe

| Timeframe | Total | Long | Short |
|-----------|-------|------|-------|
| 4H | 7 | 4 | 3 |
| 1H | 4 | 2 | 2 |
| 6Hutc | 3 | 2 | 1 |
| **TOTAL** | **14** | **8** | **6** |

### 22.2 Listado Completo

| ID | Nombre | TF | Dir | TP% | SL% | Cuentas | Estado |
|----|--------|----|----|-----|-----|---------|--------|
| 01 | double_top_long_4H | 4H | LONG | 4 | 10 | 00, E1, 01 | ACTIVE |
| 02 | reversal_long_4H | 4H | LONG | 3 | 10 | 00, E1 | ACTIVE |
| 03 | parity_long_4H | 4H | LONG | 3 | 10 | 00, E1 | ACTIVE |
| 04 | reversal_short_4H | 4H | SHORT | 3 | 9 | 00, E1 | ACTIVE |
| 05 | parity_short_4H | 4H | SHORT | 3 | 9 | 00 | DEPRECATING |
| 06 | reversal_long_1H | 1H | LONG | 2 | 10 | 00, E1 | ACTIVE |
| 07 | reversal_short_1H | 1H | SHORT | 1.9 | 5 | 00, E1 | ACTIVE |
| 08 | reversal_long_6Hutc | 6H | LONG | 4 | 10 | 00, E1 | ACTIVE |
| 09 | reversal_short_6Hutc | 6H | SHORT | 4 | 7.5 | 00, E1 | ACTIVE |
| 10 | parity_long_1H | 1H | LONG | 2 | 10 | 00, E1 | ACTIVE |
| 11 | parity_short_1H | 1H | SHORT | 2 | 7.5 | 00, E1 | ACTIVE |
| 12 | parity_long_6Hutc | 6H | LONG | 3.5 | 10 | 00 | ACTIVE |
| 13 | orderblocks_short_4H | 4H | SHORT | 5 | 10 | 00, E1 | ACTIVE |
| 14 | orderblocks_long_4H | 4H | LONG | 5 | 10 | 00 | ACTIVE |

---

## 23. Estructuras de Datos

### 23.1 Position Dict

**Campos:**
- symbol: String - 'BTCUSDT'
- size: String - '0.0004'
- entry_price: String - '91167.7'
- direction: String - 'long' o 'short'
- tp: String - '94734.408' (precio absoluto)
- sl: String - '82051.03' (precio absoluto)
- order_id: String - '1391784175051902977'
- opened_at: String ISO - '2026-01-04T18:58:46.394725+00:00'
- usdt_amount: Float - 40.0

### 23.2 Strategy Dict (desde YAML)

**Campos:**
- id: String - '01_double_top_long_4H'
- name: String - 'double_top_long_4H'
- timeframe: String - '4H'
- active: Boolean - True/False
- direction: String - 'long' o 'short'
- sell_after_ncandles: Int - 50
- order_amount: Float - 40
- Parámetros específicos: lookback, tolerance, etc.
- tp_pct: Float - 4.0
- sl_pct: Float - 10.0

### 23.3 OHLCV Array (para funciones)

**Estructura dict con arrays numpy:**
- open: np.array([91000, 91100, ...])
- high: np.array([91500, 91600, ...])
- low: np.array([90800, 90900, ...])
- close: np.array([91167, 91250, ...])
- volume: np.array([1000000, 1100000, ...])

---

## 24. Variables de Estado

### 24.1 OPEN_POSITIONS

**En memoria:**
- Tipo: Dict de listas
- Clave: strategy_id (String)
- Valor: Lista de position dicts

**En JSON (bot_state_XX.json):**
- Bajo clave "positions"
- Mismo formato que memoria

### 24.2 STRATEGY_CANDLES

**En memoria:**
- Tipo: Dict de enteros
- Clave: strategy_id (String)
- Valor: Int (contador de velas)

**En JSON:**
- Bajo clave "strategy_candles"
- Mismo formato

### 24.3 bot_state_XX.json

**Ubicación:** `persistence/bot_files_XX/bot_state_XX.json`

**Estructura:**
- Root object con dos claves: "positions" y "strategy_candles"
- Se guarda tras cada cambio de estado (nueva posición, cierre, incremento candles)

### 24.4 bot_trades_XX.xlsx

**Ubicación:** `persistence/bot_files_XX/bot_trades_XX.xlsx`

**Columnas:**
- DATE, SYMBOL, STRATEGY, DIRECTION
- ENTRY_PRICE, EXIT_PRICE, SIZE
- PROFIT, PROFIT_PCT
- CLOSE_REASON (TP/SL/TIMEOUT)
- DURATION (en velas)

---

# 🎉 FIN DEL DOCUMENTO

**Esta es la documentación técnica completa del sistema BOT_trading v2.0.**

---

**Última actualización:** 2026-01-06  
**Versión:** 2.0  
**Autor:** Trading Bot Team  
**Sistema:** BOT_trading

---

## 📝 NOTAS IMPORTANTES

1. **Arquitectura actual:** Sistema con estrategias definidas en YAML y registro explícito mediante elif en strategy_registry.py

2. **Añadir estrategia:** Solo 3 archivos a modificar (signals/nuevo.py, strategy_registry.py, strategies.yaml)

3. **Validación:** Sistema valida automáticamente que estrategias en YAML existen en IMPLEMENTED_STRATEGIES

4. **Estado persistente:** Todas las posiciones y contadores se guardan en JSON para recuperación tras crashes

5. **TP/SL automático:** Sistema verifica cada 10 segundos todas las posiciones abiertas

6. **Timeout por velas:** Posiciones se cierran automáticamente tras N velas configuradas

7. **Sync con broker:** Cada vela cerrada se sincroniza con Bitget para detectar cierres externos

8. **Dashboard en tiempo real:** Flask server en puerto 5000/5001/5099 según cuenta

9. **Logging dual:** Consola limpia para humanos, archivo detallado para debugging

10. **Multi-cuenta:** Soporta 3 cuentas independientes (00, E1, 01) con configuración separada
