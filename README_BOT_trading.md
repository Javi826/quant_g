# 🤖 BOT TRADING - DOCUMENTACIÓN TÉCNICA v2.2

**Sistema de Trading Automatizado Multi-Estrategia con Market Regime**

---

**Versión:** 2.2  
**Fecha:** 2026-01-13  
**Python:** 3.12  
**Framework:** Flask + ccxt + Bitget API  
**Nuevo:** Market Regime Classifier con Position Sizing Adaptativo

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
10. [**Market Regime Classifier ⭐ NUEVO**](#10-market-regime-classifier--nuevo)

### PARTE 3: CONFIGURACIÓN
11. [settings.py - Configuración Central](#11-settingspy---configuración-central)
12. [strategies.yaml - Definiciones](#12-strategiesyaml---definiciones)
13. [strategy_registry.py - Implementaciones ⭐](#13-strategy_registrypy---implementaciones-)
14. [**Configuración de Market Regime ⭐ NUEVO**](#14-configuración-de-market-regime--nuevo)

### PARTE 4: LÓGICA DE NEGOCIO
15. [Detección de Señales](#15-detección-de-señales)
16. [Gestión de Posiciones](#16-gestión-de-posiciones)
17. [TP/SL y Timeout](#17-tpsl-y-timeout)
18. [Sincronización con Broker](#18-sincronización-con-broker)
19. [**Position Sizing Adaptativo ⭐ NUEVO**](#19-position-sizing-adaptativo--nuevo)

### PARTE 5: FLUJOS CRÍTICOS
20. [Ciclo de Vida del Bot](#20-ciclo-de-vida-del-bot)
21. [Flujos de Operación](#21-flujos-de-operación)
22. [**Flujo de Market Regime ⭐ NUEVO**](#22-flujo-de-market-regime--nuevo)

### PARTE 6: DESARROLLO
23. [Añadir Nueva Estrategia ⭐](#23-añadir-nueva-estrategia-)
24. [Sistema de Logging](#24-sistema-de-logging)
25. [Troubleshooting](#25-troubleshooting)

### PARTE 7: REFERENCIA RÁPIDA
26. [Catálogo de Estrategias](#26-catálogo-de-estrategias)
27. [Estructuras de Datos](#27-estructuras-de-datos)
28. [Variables de Estado](#28-variables-de-estado)

---

# PARTE 1: VISIÓN GENERAL

## 1. Introducción al Sistema

### 1.1 ¿Qué es BOT_trading?

BOT_trading es un sistema automatizado de trading en futuros de criptomonedas que opera 24/7 sin intervención humana. Gestiona múltiples estrategias simultáneamente en diferentes timeframes (4H, 1H, 6Hutc, 2m, 5m) y soporta operación en múltiples cuentas independientes.

**NUEVO EN v2.2:** Sistema de Market Regime que ajusta automáticamente el tamaño de posición según las condiciones de mercado detectadas (volátil, ranging, trending).

### 1.2 Características Principales

**Trading:**
- 14 estrategias diferentes implementadas
- **Position sizing adaptativo según market regime** ⭐ NUEVO
- Gestión automática de Take Profit (TP) y Stop Loss (SL)
- Timeout automático por número de velas
- Sincronización continua con el broker
- Soporte para posiciones LONG y SHORT

**Arquitectura:**
- Configuración declarativa en YAML
- Separación clara de responsabilidades entre módulos
- **Market Regime Classifier con métricas técnicas** ⭐ NUEVO
- Sistema de logging dual (consola + archivo con rotación)
- Estado persistente para recuperación tras crashes
- Validación exhaustiva de configuración al arrancar

**Monitoreo:**
- Dashboard web en tiempo real con Flask
- **Visualización de market regime en dashboard** ⭐ NUEVO
- API REST para integración externa
- Logs streaming en vivo
- Métricas de rendimiento
- Histórico de trades en Excel

### 1.3 Flujo de Alto Nivel

```
INICIALIZACIÓN
├─ Cargar configuración (settings.py)
├─ Cargar estrategias (strategies.yaml)
├─ Cargar configuración de market regime ⭐ NUEVO
├─ Conectar a Bitget (API + WebSocket)
└─ Recuperar estado anterior (bot_state_XX.json)

MAIN LOOP (infinito)
├─ ¿Nueva vela cerrada?
│  ├─ SÍ → Procesar estrategias
│  │       ├─ Sync con broker
│  │       ├─ Calcular market regime ⭐ NUEVO
│  │       ├─ Ajustar multiplicador de posición ⭐ NUEVO
│  │       ├─ Incrementar candles
│  │       ├─ Check timeout
│  │       └─ Detectar señales
│  └─ NO → Continuar
└─ Cada 10s: Verificar TP/SL

SEÑAL DETECTADA
├─ Verificar balance
├─ Calcular size con multiplicador de régimen ⭐ NUEVO
├─ Calcular TP/SL
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
│   ├─ Market regime cache ⭐ NUEVO
│   └─ Main loop
│
├─ Sistema de Estrategias 
│   ├─ strategies.yaml (configuración)
│   ├─ strategy_registry.py (elif explícito)
│   ├─ strategy_processor.py (con adjusted_order_amount) ⭐ NUEVO
│   └─ strategy_loader.py (carga YAML)
│
├─ Market Regime System ⭐ NUEVO
│   ├─ regime_classifier.py (detección y cálculo)
│   ├─ REGIME_FAMILIES (configuración de thresholds)
│   └─ REGIME_FAMILY_SIZING (multiplicadores)
│
├─ Execution Manager (execution/bitget_client.py)
│   ├─ Comunicación con Bitget API
│   ├─ Autenticación HMAC
│   └─ Manejo de órdenes
│
├─ Dashboard (api/backend.py)
│   ├─ Servidor Flask
│   ├─ API REST endpoints
│   ├─ Market Regime visualization ⭐ NUEVO
│   └─ Templates HTML
│
├─ Funciones de Señales (signals/) ⭐ COMPARTIDO
│   ├─ add_signals_double_top.py
│   ├─ add_signals_reversal.py
│   ├─ add_signals_parity.py
│   └─ add_signals_orderblocks.py
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
├─ Market Regime Classifier ⭐ NUEVO
│   ├─ Calculate metrics (ATR, ER, PE)
│   ├─ Detect regime (volatile/ranging/trending)
│   └─ Get multiplier (0x, 1.0x, 1.5x, etc.)
│
└─ Signal Detection (strategy functions)
    ↓
Strategy Processor (orchestrator)
    ↓
├─ Signal detected? → Adjust order amount by multiplier ⭐ NUEVO
├─ Open position with adjusted size
├─ Position tracking → Check TP/SL/Timeout
└─ Update state → Save JSON
```

### 2.3 Separación de Responsabilidades

| Módulo | Responsabilidad | Archivos Clave |
|--------|-----------------|----------------|
| **core/** | Orquestación y ciclo de vida | orchestrator.py |
| **strategies/** | Carga y validación de estrategias | strategy_loader.py, strategy_registry.py, strategy_processor.py |
| **signals/** (compartido) | Funciones de detección | add_signals_*.py |
| **market_regime/** ⭐ NUEVO | Clasificación de mercado | regime_classifier.py |
| **execution/** | Comunicación con broker | bitget_client.py |
| **api/** | Dashboard web | backend.py, templates/ |
| **config/** | Configuración central | settings.py |
| **persistence/** | Estado y logs | bot_state_XX.json, logs/ |
| **validation/** | Validación de configuración | strategy_validator.py |
| **bot_utils/** | Utilidades (logger, etc.) | logger.py |

### 2.4 Organización de Carpetas

El proyecto se organiza en 3 áreas principales:

**bitget/BOT_trading/** (Producción)
- Bot autónomo de trading en vivo
- Contiene: core, strategies, execution, api, config, persistence, **market_regime** ⭐ NUEVO
- **NO contiene** las funciones de señales (están en signals/)

**bitget/signals/** (Compartido)
- Funciones de detección de señales
- Indicadores técnicos
- Usado por BOT_trading (producción) Y development (backtesting)

**bitget/development/** (Desarrollo)
- Backtesting, análisis, optimización
- Extracción de datos históricos (parquet_process)
- Herramientas de desarrollo

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
| **scipy** ⭐ NUEVO | Cálculo de permutation entropy |
| **ta-lib/pandas_ta** ⭐ NUEVO | Indicadores técnicos (ATR, ER) |

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
bitget/                                  # Directorio raíz del proyecto
│
├── BOT_trading/                         # 🤖 Bot de producción
│   │
│   ├── main.py                          # Entry point del sistema
│   │
│   ├── config/                          # Configuración
│   │   ├── __init__.py
│   │   ├── settings.py                  # Settings centralizados + regime config ⭐ NUEVO
│   │   └── connect_pass.py              # Credenciales (privado)
│   │
│   ├── core/                            # Componentes core
│   │   ├── __init__.py
│   │   └── orchestrator.py              # BotOrchestrator (cerebro) + regime cache ⭐ NUEVO
│   │
│   ├── strategies/                      # Sistema de estrategias 
│   │   ├── __init__.py
│   │   ├── strategies.yaml              # Definiciones YAML
│   │   ├── strategy_registry.py         # Elif explícito 
│   │   ├── strategy_loader.py           # Cargador de YAML
│   │   └─ processor.py        # Procesador con adjusted_order_amount ⭐ NUEVO
│   │
│   ├── market_regime/                   # ⭐ NUEVO - Sistema de régimen
│   │   ├── __init__.py
│   │   └── regime_classifier.py         # Clasificador de mercado
│   │
│   ├── market_data/                     # Datos de mercado
│   │   ├── __init__.py
│   │   ├── api_client.py                # Cliente API de Bitget
│   │   ├── data_utils.py                # Utilidades de datos
│   │   └── websocket_manager.py         # WebSocket manager
│   │
│   ├── execution/                       # Ejecución de órdenes
│   │   ├── __init__.py
│   │   └── bitget_client.py             # Cliente API Bitget
│   │
│   ├── api/                             # Dashboard web
│   │   ├── __init__.py
│   │   ├── backend.py                   # Flask server + API + regime endpoints ⭐ NUEVO
│   │   └── templates/
│   │       └── dashboard.html           # Con visualización de regime ⭐ NUEVO
│   │
│   ├── analytics/                       # Métricas
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── persistence/                     # Datos persistentes
│   │   ├── bot_files_00/                # Cuenta 00
│   │   │   ├── BOT_orchestator_00.log
│   │   │   ├── bot_trades_00.xlsx
│   │   │   └── bot_state_00.json
│   │   ├── bot_files_E1/                # Cuenta E1
│   │   └── bot_files_01/                # Cuenta 01
│   │
│   ├── bot_utils/                       # Utilidades
│   │   ├── __init__.py
│   │   └── logger.py
│   │
│   └── validation/                      # Validaciones
│       ├── __init__.py
│       └── strategy_validator.py
│
├── signals/                             # 🔄 Módulo compartido
│   ├── add_signals_double_top.py
│   ├── add_signals_reversal.py
│   ├── add_signals_parity.py
│   ├── add_signals_orderblocks.py
│   └── indicators.py                    # Indicadores técnicos
│
└── development/                         # 🛠️ Desarrollo y análisis
    ├── parquet_process/                 # Extracción de datos históricos
    ├── backtesters/                     # Backtesting
    ├── analysis/                        # Análisis de resultados
    ├── live_trading/                    # Versiones antiguas
    ├── testing/                         # Tests
    └── tools/                           # Herramientas
```

### 4.2 Propósito de Cada Directorio

| Directorio | Propósito | Archivos Importantes |
|------------|-----------|---------------------|
| **config/** | Configuración central | settings.py (+ regime config), connect_pass.py |
| **core/** | Lógica central de orquestación | orchestrator.py (+ regime_cache) |
| **strategies/** | Sistema de estrategias | strategies.yaml, strategy_registry.py, processor.py |
| **market_regime/** ⭐ NUEVO | Clasificación de mercado | regime_classifier.py |
| **signals/** (compartido) | Funciones de detección compartidas | add_signals_*.py, indicators.py |
| **market_data/** | Datos de mercado y API | api_client.py, data_utils.py |
| **execution/** | Comunicación con broker | bitget_client.py |
| **api/** | Dashboard web y API REST | backend.py (+ regime endpoints), dashboard.html |
| **persistence/** | Estado, logs, trades | JSON, Excel, logs |
| **development/** | Backtesting y análisis | parquet_process/, backtesters/ |

---

## 5. BotOrchestrator (core/)

### 5.1 Responsabilidades

El **BotOrchestrator** es el componente central que:

1. Inicializa todo el sistema
2. Coordina las estrategias activas
3. Gestiona el ciclo de vida del bot
4. Controla el main loop infinito
5. Sincroniza con el broker periódicamente
6. **Mantiene cache de market regime por timeframe** ⭐ NUEVO
7. **Ajusta position sizing según régimen detectado** ⭐ NUEVO

### 5.2 Clase BotOrchestrator

**Constructor (`__init__`):**
- Recibe: account_number, bitget_client, connect_bitget_func, active_strategy_ids
- Inicializa: configuración de cuenta, rutas de archivos, estado global
- Prepara: OPEN_POSITIONS, STRATEGY_CANDLES, strategies, symbols_by_strategy
- **Inicializa: regime_cache dict (vacío)** ⭐ NUEVO

**Variables de Estado:**
- `self.OPEN_POSITIONS`: Dict de posiciones abiertas por estrategia
- `self.STRATEGY_CANDLES`: Dict de contadores de velas por estrategia
- `self.strategies`: Lista de estrategias cargadas desde YAML
- `self.strategies_by_timeframe`: Estrategias agrupadas por timeframe
- `self.symbols_by_strategy`: Símbolos asignados a cada estrategia
- **`self.regime_cache`: Dict de multiplicadores por timeframe** ⭐ NUEVO

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
9. **Inicializar regime_cache como dict vacío** ⭐ NUEVO
10. Calcular próximas velas con `calculate_next_candle_time()`

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
3. **Calcular market regime con `_update_regime_for_timeframes()`** ⭐ NUEVO
4. Para cada estrategia del timeframe:
   - Si tiene posiciones:
     - Incrementar contador de velas con `increment_strategy_candles()`
     - Verificar timeout con `check_candles_timeout_for_strategy()`
   - Si NO tiene posiciones:
     - **Obtener multiplicador de regime_cache** ⭐ NUEVO
     - **Calcular adjusted_order_amount** ⭐ NUEVO
     - Buscar señales con `process_strategy(adjusted_order_amount)` ⭐ NUEVO
5. Recalcular próxima vela con `calculate_next_candle_time()`

### 5.5 Método `_update_regime_for_timeframes()` ⭐ NUEVO

**Propósito:** Calcular y cachear multiplicador de market regime para timeframes cerrados.

**Workflow:**
1. Para cada timeframe en lista de cerrados:
   - Llamar a `get_regime_multiplier(symbol='BTCUSDT', timeframe=tf)`
   - Recibir tupla: (multiplier, family_name)
   - Guardar en `self.regime_cache[tf] = multiplier`
   - Log: `[REGIME] {tf}: {family_name} (multiplier={multiplier}x)`
2. En caso de error:
   - Usar fallback multiplier=1.0
   - Log error y continuar

**Ejemplo de logs:**
```
[REGIME] Updating regime for timeframes: ['1H']
[REGIME] 1H: RANGING (multiplier=1.0x)
```

### 5.6 Inicialización desde main.py

**main.py** es el entry point:
- Parsea argumentos: `--account` (requerido), `--set-active` (opcional)
- Mapea clientes Bitget y funciones ccxt por cuenta
- Parsea estrategias activas si se proporcionó `--set-active`
- Crea instancia de `BotOrchestrator`
- Llama a `orchestrator.run()`
- Maneja Ctrl+C para shutdown graceful

---

## 6. Sistema de Estrategias ⭐

*(Contenido sin cambios)*

---

## 7. Execution: Cliente Bitget

*(Contenido sin cambios)*

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
| **`/api/regime/current`** ⭐ NUEVO | GET | Market regime actual |
| **`/api/correlation-matrix`** ⭐ NUEVO | POST | Matriz de correlación |

**`/api/status`:**
- Carga estado desde JSON
- Obtiene balance actual
- Calcula profit total
- Agrupa posiciones por símbolo
- **Incluye market regime actual en respuesta** ⭐ NUEVO
- Retorna info de estrategias con badges (ACTIVE, DEPRECATING, NOT IMPLEMENTED)

**`/api/regime/current`:** ⭐ NUEVO
- Query param: `timeframe` (ej: '4H', '1H', '6Hutc')
- Llama a `get_regime_info(timeframe)` desde regime_classifier
- Retorna JSON con:
  - `family`: 'volatile', 'ranging', o 'trending'
  - `multiplier`: Float (0, 1.0, 1.5, etc.)
  - `metrics`: Dict con ATR, ER, PE
  - `thresholds`: Dict con umbrales configurados
  - `all_families`: REGIME_FAMILY_SIZING completo
  - `all_thresholds`: REGIME_FAMILIES completo

**`/api/logs/stream`:**
- Lee archivo de log desde última posición conocida
- Filtra códigos ANSI y limpia líneas
- Retorna solo líneas nuevas
- Actualiza posición para próxima lectura

### 8.3 Template HTML

**Ubicación:** `api/templates/dashboard.html`

**Componentes:**
- Header con stats (Balance, Profit, Profit %, **Market Regime** ⭐ NUEVO)
- **Tab Market Regime con visualización** ⭐ NUEVO:
  - Selector de timeframe (4H, 1H, 6Hutc)
  - Card de régimen actual con métricas
  - Barras de progreso para ATR, ER, PE
  - Reglas de clasificación por familia
- Tabla de posiciones (Symbol, Direction, Size, Strategies)
- Tabla de estrategias (ID, Status, Positions, Symbols)
- **Tab Analytics con Correlation Heatmap** ⭐ NUEVO
- Container de logs con auto-scroll

**Actualización:**
- Polling cada 2s con JavaScript
- `updateDashboard()` llama a `/api/status`
- `updateLogs()` llama a `/api/logs/stream`
- **`updateRegime()` llama a `/api/regime/current`** ⭐ NUEVO

---

## 9. Funciones de Señales

*(Contenido sin cambios)*

---

## 10. Market Regime Classifier ⭐ NUEVO

### 10.1 Propósito

El **Market Regime Classifier** analiza las condiciones actuales del mercado y clasifica el estado en una de tres familias:

- **VOLATILE:** Alta volatilidad, mercado errático → Reducir o bloquear trading
- **RANGING:** Mercado lateral, consolidación → Position sizing normal
- **TRENDING:** Tendencia clara, momentum → Aumentar position sizing

Basándose en esta clasificación, el sistema ajusta automáticamente el tamaño de las posiciones multiplicando el `order_amount` base por un **multiplicador** específico.

### 10.2 Ubicación

**Archivo:** `market_regime/regime_classifier.py`

### 10.3 Métricas Utilizadas

El clasificador calcula tres métricas técnicas en los últimos 50 períodos:

**1. ATR (Average True Range) Normalizado**
- **Fórmula:** `ATR_normalized = (ATR_50 / close[-1]) * 100`
- **Significado:** Volatilidad como porcentaje del precio actual
- **Rango:** Típicamente 0-15%
- **Uso:** Detectar volatilidad extrema

**2. ER (Efficiency Ratio)**
- **Fórmula:** `ER = abs(close[-1] - close[-50]) / sum(abs(price_changes))`
- **Significado:** Eficiencia del movimiento direccional (0=lateral, 1=tendencia perfecta)
- **Rango:** 0.0 - 1.0
- **Uso:** Detectar mercados trending vs ranging

**3. PE (Permutation Entropy)**
- **Fórmula:** Entropía de permutaciones de orden 3 en retornos logarítmicos
- **Significado:** Complejidad/aleatoriedad del movimiento de precios (0=predictible, 1=aleatorio)
- **Rango:** 0.0 - 1.0
- **Uso:** Detectar estructura vs caos en el mercado

### 10.4 Lógica de Clasificación

El régimen se determina mediante reglas basadas en umbrales configurados en `REGIME_FAMILIES`:

```python
# Pseudocódigo de clasificación

if ATR > volatile_atr_min AND PE > volatile_pe_min:
    return 'volatile'  # Alta volatilidad + alta aleatoriedad
    
elif ER < ranging_er_max:
    return 'ranging'   # Baja eficiencia direccional = lateral
    
elif ER >= trending_er_min:
    return 'trending'  # Alta eficiencia direccional = tendencia
    
else:
    return 'default'   # Caso por defecto
```

### 10.5 Funciones Principales

**`get_current_regime(timeframe: str) -> Tuple[str, Dict]`**

Función principal que:
1. Descarga OHLCV de símbolo de referencia (BTCUSDT) con ccxt
2. Calcula ATR, ER, PE en últimos 50 períodos
3. Aplica reglas de clasificación
4. Retorna tupla: (family_name, metrics_dict)

**Parámetros:**
- `timeframe`: String ('4H', '1H', '6Hutc', etc.)

**Retorna:**
```python
('volatile', {
    'atr': 3.45,
    'er': 0.25,
    'pe': 0.82
})
```

**`get_regime_multiplier(symbol: str, timeframe: str) -> Tuple[float, str]`**

Función de alto nivel que:
1. Llama a `get_current_regime(timeframe)`
2. Obtiene family name
3. Busca multiplicador en `REGIME_FAMILY_SIZING[family]`
4. Retorna tupla: (multiplier, family_name)

**Parámetros:**
- `symbol`: String (actualmente solo 'BTCUSDT' se usa)
- `timeframe`: String

**Retorna:**
```python
(1.0, 'ranging')  # multiplier=1.0, family='ranging'
```

**`get_regime_info(timeframe: str) -> Dict`**

Función auxiliar para dashboard que:
1. Llama a `get_current_regime()`
2. Obtiene multiplier y thresholds
3. Construye dict completo con toda la info
4. Retorna dict listo para JSON API

**Retorna:**
```python
{
    'family': 'ranging',
    'multiplier': 1.0,
    'metrics': {'atr': 2.1, 'er': 0.15, 'pe': 0.65},
    'thresholds': {...},
    'timeframe': '4H'
}
```

### 10.6 Manejo de Errores

**Fallback automático:**
- Si cálculo falla (API error, datos insuficientes, etc.)
- Sistema usa `multiplier = 1.0` (sin ajuste)
- Log error pero continúa operación
- **Nunca bloquea el bot por error en regime**

**Logging:**
```
[REGIME] Error calculating regime for 1H: Connection timeout
[REGIME] 1H: using fallback multiplier=1.0x (error)
```

### 10.7 Símbolo de Referencia

**`REGIME_REFERENCE_SYMBOL = 'BTCUSDT'`**

Todas las estrategias usan BTCUSDT como referencia para calcular régimen, independientemente del símbolo que tradeen. 

**Razones:**
- BTCUSDT tiene mayor liquidez y volumen
- Representa mejor el sentimiento general del mercado crypto
- Evita ruido de símbolos de baja liquidez
- Simplifica cálculo (solo 1 fetch por timeframe)

---

# PARTE 3: CONFIGURACIÓN

## 11. settings.py - Configuración Central

### 11.1 Exchange Settings

- `BASE_URL`: URL base de Bitget API
- `PRODUCT_TYPE`: "USDT-FUTURES"
- `MARGIN_MODE`: "crossed"
- `MARGIN_COIN`: "USDT"

### 11.2 API Request Settings

- `API_TIMEOUT`: 10 segundos
- `API_MAX_RETRIES`: 3 reintentos
- `API_LIMIT_LIVE`: 180 velas (límite para trading en vivo)

### 11.3 General Bot Settings

- `HOUR_ZONE`: ZoneInfo('UTC') para timestamps
- `CHECK_INTERVAL`: 10 segundos entre checks de TP/SL
- `USE_HARDCODED_SIGNALS`: False (usar señales reales)
- `DISPLAY_MODE`: "summary"

### 11.4 Account-Specific Settings

**Constante: `ACCOUNTS`**

Dict con configuración de cada cuenta:

| Cuenta | Capital | Puerto | Descripción |
|--------|---------|--------|-------------|
| 00 | 3671 USDT | 5000 | Main Account |
| E1 | 1761 USDT | 5001 | Elite Account |
| 01 | 117 USDT | 5099 | Testing Account |

### 11.5 Strategy Assignment per Account

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

### 11.6 Validation Settings

- `MIN_ORDER_AMOUNT`: 40 USDT
- `MAX_ORDER_AMOUNT`: 100 USDT
- `MIN_TP_PCT`: 1.5%
- `MAX_TP_PCT`: 10%
- `MIN_SL_PCT`: 1.5%
- `MAX_SL_PCT`: 10%
- `MIN_CANDLES`: 49
- `MAX_CANDLES`: 51
- `VALID_TIMEFRAMES`: ['1H', '4H', '6Hutc', '2m', '5m', '15m', '30m']

### 11.7 Helper Functions

**`get_account_config(account_number)`**
- Retorna configuración completa de cuenta
- Incluye paths de archivos (base_dir, log_file, state_file, trades_file)

**`get_account_strategies(account_number)`**
- Retorna lista de strategy IDs asignados a cuenta

---

## 12. strategies.yaml - Definiciones

*(Contenido sin cambios)*

---

## 13. strategy_registry.py - Implementaciones ⭐

*(Contenido sin cambios)*

---

## 14. Configuración de Market Regime ⭐ NUEVO

### 14.1 Ubicación

**Archivo:** `config/settings.py`

Las configuraciones de market regime se definen como constantes globales en settings.py junto con el resto de la configuración del sistema.

### 14.2 REGIME_REFERENCE_SYMBOL

```python
REGIME_REFERENCE_SYMBOL = 'BTCUSDT'
```

**Propósito:** Símbolo usado como referencia para calcular régimen de mercado.

**Por qué BTCUSDT:**
- Mayor liquidez y volumen en el mercado
- Mejor representación del sentimiento general
- Evita ruido de alt-coins
- Un solo fetch por timeframe (eficiencia)

### 14.3 REGIME_FAMILIES

```python
REGIME_FAMILIES = {
    'volatile': {
        'atr_min': 3.5,
        'pe_min': 0.70
    },
    'ranging': {
        'er_max': 0.45
    },
    'trending': {
        'er_min': 0.45
    },
    'default': {}
}
```

**Propósito:** Define los umbrales (thresholds) para clasificar el régimen de mercado.

**Explicación de cada familia:**

**VOLATILE:**
- `atr_min: 3.5` → ATR normalizado > 3.5% (alta volatilidad)
- `pe_min: 0.70` → Permutation Entropy > 0.70 (alta aleatoriedad)
- **Condición:** ATR alto Y PE alto = mercado errático y volátil

**RANGING:**
- `er_max: 0.45` → Efficiency Ratio < 0.45 (baja eficiencia direccional)
- **Condición:** Movimiento sin dirección clara = mercado lateral

**TRENDING:**
- `er_min: 0.45` → Efficiency Ratio >= 0.45 (alta eficiencia direccional)
- **Condición:** Movimiento direccional claro = mercado en tendencia

**DEFAULT:**
- Sin umbrales
- Usado como fallback si ninguna regla se cumple

**Orden de evaluación:**
1. VOLATILE (if ATR > 3.5 AND PE > 0.70)
2. RANGING (elif ER < 0.45)
3. TRENDING (elif ER >= 0.45)
4. DEFAULT (else)

### 14.4 REGIME_FAMILY_SIZING

```python
REGIME_FAMILY_SIZING = {
    'volatile': 0.5,
    'ranging': 1.0,
    'trending': 1.5,
    'default': 1.0
}
```

**Propósito:** Define el multiplicador de position sizing para cada familia de régimen.

**Multiplicadores:**

| Familia | Multiplicador | Significado |
|---------|---------------|-------------|
| **volatile** | 0.5x | Reducir posiciones a la mitad (o bloquear con 0) |
| **ranging** | 1.0x | Position sizing normal |
| **trending** | 1.5x | Aumentar posiciones 50% |
| **default** | 1.0x | Fallback sin ajuste |

**Ejemplo de cálculo:**
```
Estrategia: order_amount = 40 USDT
Régimen: TRENDING (multiplier = 1.5)

adjusted_order_amount = 40 * 1.5 = 60 USDT
→ Se abre posición con 60 USDT en vez de 40 USDT
```

**Caso especial: Bloquear trading**
```python
REGIME_FAMILY_SIZING = {
    'volatile': 0,  # ← multiplier=0 bloquea trading
    ...
}
```

Si `multiplier = 0`, el sistema **NO abre posiciones** en ese régimen:
```python
if regime_multiplier == 0:
    logger.info(f"[REGIME] Skipping {strat_id}: multiplier=0 (regime blocks trading)")
    continue  # No buscar señales
```

### 14.5 Modificar Configuración

**Para ajustar umbrales:**
1. Abrir `config/settings.py`
2. Modificar valores en `REGIME_FAMILIES`
3. Reiniciar bot

**Ejemplo: Hacer régimen volatile más restrictivo**
```python
'volatile': {
    'atr_min': 4.0,  # Antes 3.5, ahora requiere más volatilidad
    'pe_min': 0.75   # Antes 0.70, ahora requiere más aleatoriedad
}
```

**Para ajustar multiplicadores:**
1. Abrir `config/settings.py`
2. Modificar valores en `REGIME_FAMILY_SIZING`
3. Reiniciar bot

**Ejemplo: Ser más agresivo en trending**
```python
'trending': 2.0,  # Antes 1.5x, ahora 2.0x (doble tamaño)
```

**Ejemplo: Bloquear trading en volatile**
```python
'volatile': 0,  # Antes 0.5x, ahora 0x (no trading)
```

### 14.6 Recomendaciones

**Umbrales conservadores (menos trades):**
- volatile: atr_min=4.0, pe_min=0.75
- ranging: er_max=0.35
- trending: er_min=0.55

**Umbrales agresivos (más trades):**
- volatile: atr_min=3.0, pe_min=0.65
- ranging: er_max=0.55
- trending: er_min=0.35

**Multiplicadores conservadores:**
- volatile: 0 (bloquear)
- ranging: 1.0
- trending: 1.2

**Multiplicadores agresivos:**
- volatile: 0.5
- ranging: 1.0
- trending: 2.0

---

# PARTE 4: LÓGICA DE NEGOCIO

## 15. Detección de Señales

### 15.1 Workflow Completo

```
Nueva vela cerrada (ej: 4H)
  ↓
Obtener estrategias de ese timeframe
  ↓
Calcular market regime para 4H ⭐ NUEVO
Cachear multiplier en regime_cache['4H'] ⭐ NUEVO
  ↓
Para cada estrategia:
  ↓
¿Tiene posiciones abiertas?
├─ SÍ: Skip (no buscar nuevas)
└─ NO:
    ↓
  Obtener multiplier de regime_cache ⭐ NUEVO
  ¿multiplier == 0? ⭐ NUEVO
  ├─ SÍ: Skip (régimen bloquea trading) ⭐ NUEVO
  └─ NO:
      ↓
    Calcular adjusted_order_amount = base * multiplier ⭐ NUEVO
      ↓
    Obtener símbolos para esta estrategia
      ↓
    Para cada símbolo:
      ↓
    ├─ Descargar OHLCV (ccxt)
    ├─ Convertir a arrays numpy
    ├─ Ejecutar función de señal
    └─ ¿Señal positiva? (signals[-1] != 0)
        ├─ SÍ: Abrir posición con adjusted_order_amount ⭐ NUEVO
        └─ NO: Continuar
```

### 15.2 Funciones Involucradas

**`detect_signal_for_strategy(strategy, symbols, exchange)`**
- Crea instancia de `StrategyProcessor`
- Llama a `processor.detect_signals()`
- Retorna lista de símbolos con señal

**`process_strategy(strat, final_symbols, exchange, ..., adjusted_order_amount)` ⭐ NUEVO**
- **Ahora recibe `adjusted_order_amount` como parámetro** ⭐ NUEVO
- Llama a `detect_signal_for_strategy()`
- Para cada símbolo con señal:
  - Verificar balance suficiente
  - Llamar a `open_position_for_signal()` **con adjusted_order_amount** ⭐ NUEVO
  - Break tras abrir primera posición (una por estrategia)

---

## 16. Gestión de Posiciones

### 16.1 Apertura de Posición

**Función: `open_position_for_signal(strategy, symbol, exchange, ..., order_amount)`**

**Cambio en v2.2:** Ahora recibe `order_amount` que puede ser el ajustado por régimen. ⭐ NUEVO

Workflow:
1. Obtener precio actual con `get_current_price(symbol)`
2. **Calcular size con order_amount (que puede ser adjusted_order_amount)** ⭐ NUEVO
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
- **El order_amount ahora puede ser el base o el adjusted** ⭐ NUEVO
- Redondear a precisión del símbolo (típicamente 4 decimales)

### 16.2 Registro de Posición

Cada posición es un dict con:
- symbol, size, entry_price, direction
- tp, sl (precios absolutos)
- order_id, opened_at (timestamp ISO)
- **usdt_amount (refleja el monto ajustado por régimen)** ⭐ NUEVO

---

## 17. TP/SL y Timeout

*(Contenido sin cambios)*

---

## 18. Sincronización con Broker

*(Contenido sin cambios)*

---

## 19. Position Sizing Adaptativo ⭐ NUEVO

### 19.1 Flujo Completo

```
VELA CIERRA (ej: 1H)
    ↓
CALCULAR RÉGIMEN
├─ Fetch OHLCV de BTCUSDT para 1H
├─ Calcular métricas (ATR, ER, PE)
├─ Aplicar reglas de clasificación
├─ Determinar familia: 'ranging'
└─ Obtener multiplier: 1.0
    ↓
CACHEAR MULTIPLIER
└─ regime_cache['1H'] = 1.0
    ↓
PROCESAR ESTRATEGIA
├─ Estrategia: 06_reversal_long_1H
├─ order_amount base: 40 USDT
├─ Obtener multiplier de cache: 1.0
├─ Verificar multiplier != 0 ✓
├─ Calcular adjusted: 40 * 1.0 = 40 USDT
└─ Buscar señales con adjusted_order_amount=40
    ↓
SEÑAL DETECTADA
└─ Abrir posición con 40 USDT (sin ajuste en ranging)

---

EJEMPLO CON AJUSTE:

VELA CIERRA (1H)
    ↓
RÉGIMEN: TRENDING (multiplier=1.5)
    ↓
PROCESAR ESTRATEGIA
├─ order_amount base: 40 USDT
├─ multiplier: 1.5
├─ adjusted: 40 * 1.5 = 60 USDT
└─ Buscar señales con adjusted_order_amount=60
    ↓
SEÑAL DETECTADA
└─ Abrir posición con 60 USDT (+50% por trending)

---

EJEMPLO BLOQUEANDO:

VELA CIERRA (1H)
    ↓
RÉGIMEN: VOLATILE (multiplier=0)
    ↓
PROCESAR ESTRATEGIA
├─ order_amount base: 40 USDT
├─ multiplier: 0
├─ Verificar multiplier != 0 ✗
└─ Skip estrategia (no buscar señales)
    ↓
LOG: "[REGIME] Skipping 06_reversal_long_1H: multiplier=0 (regime blocks trading)"
```

### 19.2 Código en Orchestrator

**En `_search_signals()` (orchestrator.py):**

```python
for strat in strategies_to_process:
    strat_id = strat['id']
    
    # Skip si tiene posiciones
    if len(self.open_positions.get(strat_id, [])) > 0:
        continue
    
    # ⭐ NUEVO: Obtener multiplier y calcular adjusted
    timeframe = strat['timeframe']
    regime_multiplier = self.regime_cache.get(timeframe, 1.0)
    
    # ⭐ NUEVO: Bloquear si multiplier=0
    if regime_multiplier == 0:
        self.logger.info(
            f"[REGIME] Skipping {strat_id}: "
            f"multiplier=0 (regime blocks trading)"
        )
        continue
    
    base_order_amount = strat['order_amount']
    adjusted_order_amount = base_order_amount * regime_multiplier
    
    self.logger.debug(
        f"[REGIME] {strat_id}: TF={timeframe}, "
        f"Base=${base_order_amount:.2f}, "
        f"Multiplier={regime_multiplier}x, "
        f"Adjusted=${adjusted_order_amount:.2f}"
    )
    
    # Buscar señales con adjusted_order_amount
    self.strategy_processor.process(
        strat=strat,
        final_symbols=self.final_by_strat.get(strat['id'], []),
        exchange=self.exchange,
        open_positions=self.open_positions,
        strategy_candles=self.strategy_candles,
        adjusted_order_amount=adjusted_order_amount  # ⭐ NUEVO
    )
```

### 19.3 Modificación en Strategy Processor

**En `strategy_processor.py`:**

**Método `process()` ahora recibe `adjusted_order_amount`:** ⭐ NUEVO

```python
def process(
    self,
    strat,
    final_symbols,
    exchange,
    open_positions,
    strategy_candles,
    adjusted_order_amount=None  # ⭐ NUEVO: parámetro opcional
):
    # Si no se proporciona, usar order_amount de estrategia
    order_amount = adjusted_order_amount or strat['order_amount']
    
    # ... resto del código usando order_amount
    
    # Pasar order_amount a open_position_for_signal
    open_position_for_signal(
        strategy=strat,
        symbol=symbol,
        order_amount=order_amount,  # ⭐ USA EL AJUSTADO
        ...
    )
```

**Backward compatible:** Si `adjusted_order_amount=None`, usa `strat['order_amount']` (comportamiento anterior).

### 19.4 Logs de Régimen

**Logs informativos (INFO level):**
```
[REGIME] Updating regime for timeframes: ['1H']
[REGIME] 1H: RANGING (multiplier=1.0x)
[REGIME] Skipping 06_reversal_long_1H: multiplier=0 (regime blocks trading)
```

**Logs de debug (DEBUG level):**
```
[REGIME] 06_reversal_long_1H: TF=1H, Base=$40.00, Multiplier=1.5x, Adjusted=$60.00
```

### 19.5 Ejemplo Completo de Ejecución

```
2026-01-13 16:00:00 - INFO - New candles 2026-01-13 16:00:00 UTC
2026-01-13 16:00:00 - INFO - Timeframes: 1H
2026-01-13 16:00:01 - INFO - Sync with broker completed
2026-01-13 16:00:02 - INFO - [REGIME] Updating regime for timeframes: ['1H']
2026-01-13 16:00:03 - INFO - [REGIME] 1H: TRENDING (multiplier=1.5x)
2026-01-13 16:00:04 - INFO - Searching Signals...
2026-01-13 16:00:05 - DEBUG - [REGIME] 06_reversal_long_1H: TF=1H, Base=$40.00, Multiplier=1.5x, Adjusted=$60.00
2026-01-13 16:00:06 - INFO - Processing strategy: 06_reversal_long_1H
2026-01-13 16:00:08 - INFO - Signals detected: 2
2026-01-13 16:00:09 - INFO - LONG BTCUSDT | Amount: $60.00 | Price: 94356.8
2026-01-13 16:00:10 - INFO - Position opened successfully
```

**Notas:**
- Base amount era 40 USDT
- Régimen TRENDING detectado con multiplier=1.5x
- Posición abierta con 60 USDT (50% más)
- Todo automático sin intervención manual

---

# PARTE 5: FLUJOS CRÍTICOS

## 20. Ciclo de Vida del Bot

### 20.1 Diagrama Completo

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
├─ Inicializar regime_cache (vacío) ⭐ NUEVO
└─ Calcular próximas velas

MAIN LOOP (infinito)
├─ Detectar velas cerradas
│  ├─ Calcular market regime ⭐ NUEVO
│  ├─ Cachear multipliers ⭐ NUEVO
│  └─ Procesar estrategias con adjusted sizing ⭐ NUEVO
├─ Cada 10s: Verificar TP/SL
└─ Sleep 0.05s

SHUTDOWN (Ctrl+C)
├─ Capturar señal
├─ Guardar estado final
└─ Exit
```

### 20.2 Estados del Bot

**STARTING:**
- Cargando configuración
- Validando estrategias
- Conectando a Bitget
- **Inicializando regime classifier** ⭐ NUEVO

**RUNNING:**
- Main loop activo
- **Calculando regime en cada vela** ⭐ NUEVO
- **Ajustando position sizing** ⭐ NUEVO
- Procesando señales
- Verificando TP/SL

**STOPPING:**
- Guardando estado
- Cerrando conexiones

---

## 21. Flujos de Operación

### 21.1 Detección de Vela Cerrada

```
Timeframe: 4H
now = 2026-01-04 20:00:00 UTC
next_candle_time = 2026-01-04 20:00:00 UTC

now >= next_candle_time?
├─ SÍ:
│  ├─ Log: "New 4H candle closed"
│  ├─ Sync con broker
│  ├─ Calcular regime para 4H ⭐ NUEVO
│  │  ├─ Fetch OHLCV de BTCUSDT
│  │  ├─ Calcular ATR, ER, PE
│  │  ├─ Clasificar: TRENDING
│  │  └─ Cachear multiplier=1.5
│  ├─ Para cada estrategia 4H:
│  │  ├─ Obtener multiplier del cache
│  │  ├─ Calcular adjusted_order_amount
│  │  └─ Procesar con adjusted amount
│  └─ Recalcular: next_candle_time = 2026-01-05 00:00:00
└─ NO: Continuar loop
```

### 21.2 Apertura de Posición

```
Señal detectada: BTCUSDT, estrategia 01_double_top_long_4H
Régimen: TRENDING (multiplier=1.5) ⭐ NUEVO

VERIFICACIONES
└─ Balance >= adjusted_order_amount? ⭐ NUEVO
   ├─ SÍ: Continuar
   └─ NO: Skip

CÁLCULOS
├─ order_amount base: 40 USDT
├─ Multiplier: 1.5x ⭐ NUEVO
├─ adjusted_order_amount: 60 USDT ⭐ NUEVO
├─ Precio actual: 91167.7
├─ Size: 60 / 91167.7 = 0.000658 ⭐ NUEVO (antes era 0.000439)
├─ TP: 91167.7 × 1.04 = 94814.4
└─ SL: 91167.7 × 0.90 = 82051.0

ORDEN MARKET
└─ Bitget API: place_order() con size=0.000658 ⭐ NUEVO

TRACKING
├─ Crear position dict
├─ usdt_amount: 60 (refleja ajuste) ⭐ NUEVO
├─ Añadir a OPEN_POSITIONS[strat_id]
├─ Init STRATEGY_CANDLES[strat_id] = 0
└─ Guardar estado (JSON)
```

### 21.3 Cierre de Posición

*(Sin cambios - TP/SL no se ajustan por régimen)*

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
└─ profit_usd = 0.0404 × 60 = 2.42$ ⭐ NUEVO (antes 1.62$)

CERRAR ORDEN
└─ Bitget API: close_position()

LOG Y ACTUALIZAR
├─ log_trade_to_excel()
├─ Remover de OPEN_POSITIONS
├─ Guardar estado
└─ Log: "Position closed (TP) - Profit: 2.42$ (4.04%)" ⭐ NUEVO
```

**Nota:** Profit en USD es mayor porque se abrió con 60 USDT en vez de 40 USDT.

---

## 22. Flujo de Market Regime ⭐ NUEVO

### 22.1 Diagrama Completo

```
TRIGGER: VELA CIERRA
    ↓
ORCHESTRATOR._process_timeframe(tf)
    ↓
SYNC CON BROKER
    ↓
LLAMAR: _update_regime_for_timeframes([tf])
    ↓
PARA CADA TIMEFRAME:
    ↓
┌─────────────────────────────────────┐
│ MARKET REGIME CLASSIFIER            │
├─────────────────────────────────────┤
│ 1. Fetch OHLCV                      │
│    ├─ Symbol: BTCUSDT               │
│    ├─ Timeframe: tf                 │
│    └─ Limit: 180 velas              │
│                                     │
│ 2. Calcular métricas                │
│    ├─ ATR_normalized                │
│    ├─ Efficiency Ratio (ER)         │
│    └─ Permutation Entropy (PE)      │
│                                     │
│ 3. Aplicar reglas                   │
│    ├─ if ATR>3.5 AND PE>0.70:       │
│    │   → VOLATILE                    │
│    ├─ elif ER<0.45:                 │
│    │   → RANGING                     │
│    ├─ elif ER>=0.45:                │
│    │   → TRENDING                    │
│    └─ else:                          │
│        → DEFAULT                     │
│                                     │
│ 4. Obtener multiplier               │
│    └─ REGIME_FAMILY_SIZING[family]  │
└─────────────────────────────────────┘
    ↓
CACHEAR: regime_cache[tf] = multiplier
    ↓
LOG: [REGIME] {tf}: {family} (multiplier={mult}x)
    ↓
CONTINUAR CON ESTRATEGIAS
    ↓
PARA CADA ESTRATEGIA DE TIMEFRAME:
    ↓
LEER: regime_multiplier = regime_cache[tf]
    ↓
¿multiplier == 0?
├─ SÍ: Skip estrategia (log warning)
└─ NO:
    ↓
  adjusted = base × multiplier
    ↓
  Buscar señales con adjusted
    ↓
  Si señal → Abrir posición con adjusted
```

### 22.2 Timing Crítico

**¿CUÁNDO se calcula el régimen?**
- **DESPUÉS** del sync con broker
- **ANTES** de buscar señales
- **SOLO** cuando cierra vela del timeframe
- **UNA VEZ** por timeframe por vela

**¿POR QUÉ este timing?**
- Necesita vela completa/cerrada para cálculo preciso
- Calcula ANTES de señales para tener multiplier listo
- Sync ANTES para evitar conflictos de estado

**Ejemplo temporal:**
```
15:59:58 - Esperando vela de 1H...
16:00:00 - ¡Vela 1H cerrada!
16:00:01 - Sync con broker (reconciliar posiciones)
16:00:02 - Calcular regime para 1H
16:00:03 - Regime detectado: TRENDING (1.5x)
16:00:04 - Cachear multiplier
16:00:05 - Buscar señales con adjusted_order_amount
16:00:10 - Señal detectada → Abrir posición (60 USDT)
```

### 22.3 Cache de Multiplicadores

**Estructura:**
```python
self.regime_cache = {
    '4H': 1.0,    # RANGING
    '1H': 1.5,    # TRENDING
    '6Hutc': 0.5  # VOLATILE
}
```

**Propiedades:**
- Diccionario en memoria (no persiste en JSON)
- Se actualiza solo cuando cierra vela de ese timeframe
- Valor se mantiene hasta próxima vela cerrada
- Si no existe clave, `get(tf, 1.0)` retorna 1.0 (sin ajuste)

**Beneficios del cache:**
- Evita recalcular régimen múltiples veces por vela
- Un solo fetch de OHLCV por timeframe por vela
- Múltiples estrategias del mismo timeframe usan mismo multiplier
- Eficiente: O(1) lookup por estrategia

### 22.4 Manejo de Errores

**Errores posibles:**
- API Bitget no responde (timeout)
- OHLCV insuficiente (< 50 velas)
- Error en cálculo de métrica (división por cero, etc.)
- Excepción inesperada

**Comportamiento ante error:**
```python
try:
    multiplier, family = get_regime_multiplier('BTCUSDT', tf)
    self.regime_cache[tf] = multiplier
    logger.info(f"[REGIME] {tf}: {family} (multiplier={mult}x)")
except Exception as e:
    logger.error(f"[REGIME] Error calculating regime for {tf}: {e}")
    self.regime_cache[tf] = 1.0  # Fallback
    logger.debug(f"[REGIME] {tf}: using fallback multiplier=1.0x (error)")
```

**Principio:** **Nunca bloquear el bot por error en regime**
- Siempre usa fallback (1.0x = sin ajuste)
- Log error pero continúa
- Bot sigue operando normalmente

### 22.5 Logs Completos

**Ejemplo de logs en operación normal:**

```
2026-01-13 16:00:00 - INFO - ================================================
2026-01-13 16:00:00 - INFO - New candles 2026-01-13 16:00:00 UTC
2026-01-13 16:00:00 - INFO - Timeframes: 1H
2026-01-13 16:00:01 - INFO - Sync with broker completed.
2026-01-13 16:00:02 - INFO - [REGIME] Updating regime for timeframes: ['1H']
2026-01-13 16:00:03 - INFO - [REGIME] 1H: TRENDING (multiplier=1.5x)
2026-01-13 16:00:04 - INFO - Searching Signals... - 2026-01-13 16:00:04
2026-01-13 16:00:05 - INFO - ------------------------------------------------
2026-01-13 16:00:06 - INFO - Processing strategy: 06_reversal_long_1H
2026-01-13 16:00:06 - INFO - ------------------------------------------------
2026-01-13 16:00:08 - INFO - Signals detected: 2
2026-01-13 16:00:09 - INFO - Opening position: BTCUSDT LONG
2026-01-13 16:00:09 - INFO - Amount: $60.00 (adjusted by regime)
2026-01-13 16:00:10 - INFO - Entry: 94356.8, TP: 98131.07, SL: 84921.12
2026-01-13 16:00:11 - INFO - Position opened successfully
2026-01-13 16:00:12 - INFO - Signal cycle completed
2026-01-13 16:00:12 - INFO - ================================================
```

**Ejemplo con multiplicador = 0 (bloqueando):**

```
2026-01-13 16:00:02 - INFO - [REGIME] Updating regime for timeframes: ['1H']
2026-01-13 16:00:03 - INFO - [REGIME] 1H: VOLATILE (multiplier=0x)
2026-01-13 16:00:04 - INFO - Searching Signals...
2026-01-13 16:00:05 - INFO - [REGIME] Skipping 06_reversal_long_1H: multiplier=0 (regime blocks trading)
2026-01-13 16:00:06 - INFO - [REGIME] Skipping 07_reversal_short_1H: multiplier=0 (regime blocks trading)
2026-01-13 16:00:07 - INFO - Signal cycle completed
```

---

# PARTE 6: DESARROLLO

## 23. Añadir Nueva Estrategia ⭐

*(Contenido sin cambios - las estrategias no necesitan cambios para usar regime)*

---

## 24. Sistema de Logging

*(Contenido sin cambios)*

---

## 25. Troubleshooting

### 25.1 Bot No Arranca

*(Contenido sin cambios)*

### 25.2 Estrategia No Carga

*(Contenido sin cambios)*

### 25.3 Dashboard No Carga

*(Contenido sin cambios)*

### 25.4 Posiciones No Cierran

*(Contenido sin cambios)*

### 25.5 Error de Conexión API

*(Contenido sin cambios)*

### 25.6 Market Regime No Se Calcula ⭐ NUEVO

**Síntoma:** No aparecen logs de `[REGIME]` o multiplier siempre es 1.0

**Diagnóstico:**
- Verificar logs: `grep REGIME bot_orchestator_XX.log`
- Verificar imports en orchestrator.py
- Verificar que regime_classifier.py existe
- Probar cálculo manual:
```python
from market_regime.regime_classifier import get_current_regime
family, metrics = get_current_regime('1H')
print(family, metrics)
```

**Solución:**
- Verificar `market_regime/` existe y tiene `__init__.py`
- Verificar import en orchestrator: `from market_regime.regime_classifier import get_regime_multiplier`
- Verificar configuración en settings.py: `REGIME_FAMILIES`, `REGIME_FAMILY_SIZING`
- Reinstalar dependencias: scipy, pandas_ta

### 25.7 Posiciones con Tamaño Incorrecto ⭐ NUEVO

**Síntoma:** Posiciones abiertas con monto diferente al esperado

**Diagnóstico:**
- Ver logs DEBUG: `grep "adjusted_order_amount" bot_orchestator_XX.log`
- Verificar régimen activo: `grep "REGIME.*multiplier" bot_orchestator_XX.log`
- Calcular manual: `base × multiplier = adjusted`

**Ejemplo:**
```
Base: 40 USDT
Multiplier: 1.5x
Esperado: 60 USDT
Obtenido: 40 USDT ← ERROR
```

**Solución:**
- Verificar strategy_processor.py recibe `adjusted_order_amount`
- Verificar orchestrator pasa `adjusted_order_amount` a `process()`
- Verificar `adjusted_order_amount` no es None
- Verificar backward compatibility: si None, usa `strat['order_amount']`

### 25.8 Dashboard No Muestra Régimen ⭐ NUEVO

**Síntoma:** Tab "Market Regime" vacío o no responde

**Diagnóstico:**
- Probar endpoint: `curl http://localhost:5000/api/regime/current?timeframe=1H`
- Verificar logs de Flask en consola
- Verificar JavaScript en browser console (F12)

**Solución:**
- Verificar backend.py tiene endpoint `/api/regime/current`
- Verificar dashboard.html tiene tab con id `tab-regime`
- Verificar JavaScript tiene función `updateRegime()`
- Hard refresh: Ctrl+Shift+R

---

# PARTE 7: REFERENCIA RÁPIDA

## 26. Catálogo de Estrategias

*(Contenido sin cambios)*

---

## 27. Estructuras de Datos

### 27.1 Position Dict

**Campos:**
- symbol: String - 'BTCUSDT'
- size: String - '0.000658' ⭐ NUEVO (puede ser mayor por regime)
- entry_price: String - '91167.7'
- direction: String - 'long' o 'short'
- tp: String - '94734.408' (precio absoluto)
- sl: String - '82051.03' (precio absoluto)
- order_id: String - '1391784175051902977'
- opened_at: String ISO - '2026-01-04T18:58:46.394725+00:00'
- **usdt_amount: Float - 60.0** ⭐ NUEVO (refleja ajuste por regime)

### 27.2 Strategy Dict (desde YAML)

*(Contenido sin cambios)*

### 27.3 OHLCV Array (para funciones)

*(Contenido sin cambios)*

### 27.4 Regime Cache ⭐ NUEVO

**Estructura dict en memoria:**
```python
{
    '4H': 1.0,     # Float - multiplier para 4H
    '1H': 1.5,     # Float - multiplier para 1H
    '6Hutc': 0.5   # Float - multiplier para 6Hutc
}
```

**Características:**
- En memoria (no persiste)
- Clave: timeframe (String)
- Valor: multiplier (Float)
- Se actualiza cada vela cerrada
- Lookup: `regime_cache.get(tf, 1.0)` (fallback 1.0 si no existe)

### 27.5 Regime Info Dict ⭐ NUEVO

**Estructura retornada por `get_regime_info()`:**
```python
{
    'family': 'trending',           # String
    'multiplier': 1.5,              # Float
    'metrics': {
        'atr': 2.34,                # Float
        'er': 0.67,                 # Float
        'pe': 0.45                  # Float
    },
    'thresholds': {
        'atr_min': 3.5,
        'pe_min': 0.70,
        ...
    },
    'timeframe': '1H'               # String
}
```

---

## 28. Variables de Estado

### 28.1 OPEN_POSITIONS

**En memoria:**
- Tipo: Dict de listas
- Clave: strategy_id (String)
- Valor: Lista de position dicts

**En JSON (bot_state_XX.json):**
- Bajo clave "positions"
- Mismo formato que memoria

### 28.2 STRATEGY_CANDLES

**En memoria:**
- Tipo: Dict de enteros
- Clave: strategy_id (String)
- Valor: Int (contador de velas)

**En JSON:**
- Bajo clave "strategy_candles"
- Mismo formato

### 28.3 regime_cache ⭐ NUEVO

**En memoria solamente (NO persiste en JSON):**
- Tipo: Dict de floats
- Clave: timeframe (String)
- Valor: Float (multiplier)
- Se inicializa vacío en `__init__()`
- Se actualiza cada vela cerrada
- **NO se guarda en bot_state_XX.json**

**Razón de no persistir:**
- Régimen puede cambiar entre sesiones
- Siempre se recalcula en primera vela tras restart
- Evita usar multiplier desactualizado
- Cache solo útil durante sesión activa

### 28.4 bot_state_XX.json

**Ubicación:** `persistence/bot_files_XX/bot_state_XX.json`

**Estructura:**
- Root object con dos claves: "positions" y "strategy_candles"
- **NO incluye regime_cache** ⭐ NUEVO
- Se guarda tras cada cambio de estado (nueva posición, cierre, incremento candles)

### 28.5 bot_trades_XX.xlsx

**Ubicación:** `persistence/bot_files_XX/bot_trades_XX.xlsx`

**Columnas:**
- DATE, SYMBOL, STRATEGY, DIRECTION
- ENTRY_PRICE, EXIT_PRICE, SIZE
- PROFIT, PROFIT_PCT
- CLOSE_REASON (TP/SL/TIMEOUT)
- DURATION (en velas)
- **USDT_AMOUNT refleja monto ajustado por regime** ⭐ NUEVO

---

# 🎉 FIN DEL DOCUMENTO

**Esta es la documentación técnica completa del sistema BOT_trading v2.2 con Market Regime Classifier.**

---

**Última actualización:** 2026-01-13  
**Versión:** 2.2  
**Autor:** Trading Bot Team  
**Sistema:** BOT_trading  
**Nueva Feature:** Market Regime Classifier con Position Sizing Adaptativo

---

## 📝 NOTAS IMPORTANTES

1. **Arquitectura actual:** Sistema con estrategias definidas en YAML y registro explícito mediante elif en strategy_registry.py

2. **Estructura de carpetas:** Separación clara entre BOT_trading (producción), signals (compartido) y development (desarrollo)

3. **Añadir estrategia:** Solo 3 archivos a modificar (bitget/signals/nuevo.py, strategy_registry.py, strategies.yaml)

4. **Validación:** Sistema valida automáticamente que estrategias en YAML existen en IMPLEMENTED_STRATEGIES

5. **Estado persistente:** Todas las posiciones y contadores se guardan en JSON para recuperación tras crashes

6. **TP/SL automático:** Sistema verifica cada 10 segundos todas las posiciones abiertas

7. **Timeout por velas:** Posiciones se cierran automáticamente tras N velas configuradas

8. **Sync con broker:** Cada vela cerrada se sincroniza con Bitget para detectar cierres externos

9. **Dashboard en tiempo real:** Flask server en puerto 5000/5001/5099 según cuenta

10. **Logging dual:** Consola limpia para humanos, archivo detallado para debugging

11. **Multi-cuenta:** Soporta 3 cuentas independientes (00, E1, 01) con configuración separada

12. **Módulo signals compartido:** Las funciones de detección están en bitget/signals/ y son usadas tanto por BOT_trading como por development

13. **⭐ NUEVO - Market Regime:** Sistema adaptativo que ajusta position sizing (0x, 0.5x, 1.0x, 1.5x) según volatilidad, eficiencia direccional y entropía del mercado

14. **⭐ NUEVO - Métricas técnicas:** ATR normalizado, Efficiency Ratio, Permutation Entropy calculados en BTCUSDT cada vela

15. **⭐ NUEVO - Configuración flexible:** REGIME_FAMILIES (thresholds) y REGIME_FAMILY_SIZING (multipliers) completamente configurables en settings.py

16. **⭐ NUEVO - Dashboard regime:** Visualización en tiempo real del régimen actual con métricas y reglas de clasificación

17. **⭐ NUEVO - Protección multiplier=0:** Sistema puede bloquear trading completamente en mercados volátiles si multiplier=0

18. **⭐ NUEVO - Fallback robusto:** Si cálculo de regime falla, usa multiplier=1.0 sin bloquear el bot

19. **⭐ NUEVO - Cache eficiente:** Un solo cálculo de regime por timeframe por vela, compartido entre todas las estrategias

20. **⭐ NUEVO - Backward compatible:** Strategy processor acepta tanto adjusted_order_amount como None (usa base)