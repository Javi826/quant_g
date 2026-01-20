# 🤖 BOT TRADING - DOCUMENTACIÓN TÉCNICA v2.5

**Sistema de Trading Automatizado con Regime-Based Position Sizing**

---

**Versión:** 2.5  
**Fecha:** 2026-01-20  
**Python:** 3.12  
**Framework:** Flask + ccxt + Bitget API

---

## 📋 TABLA DE CONTENIDOS

### PARTE 1: VISIÓN GENERAL
1. [Introducción](#1-introducción)
2. [Arquitectura](#2-arquitectura)
3. [Stack Tecnológico](#3-stack-tecnológico)

### PARTE 2: MARKET REGIME SYSTEM
4. [Clasificación de Mercado](#4-clasificación-de-mercado)
5. [Matriz de Régimen Custom](#5-matriz-de-régimen-custom)
6. [Configuración de Régimen](#6-configuración-de-régimen)

### PARTE 3: COMPONENTES CORE
7. [BotOrchestrator](#7-botaniquestrator)
8. [Sistema de Estrategias](#8-sistema-de-estrategias)
9. [Dashboard Web](#9-dashboard-web)

### PARTE 4: CONFIGURACIÓN
10. [Settings.py](#10-settingspy)
11. [Strategies.yaml](#11-strategiesyaml)
12. [Alta de Estrategias](#12-alta-de-estrategias)

### PARTE 5: FLUJOS Y OPERACIÓN
13. [Ciclo de Vida](#13-ciclo-de-vida)
14. [Position Sizing Adaptativo](#14-position-sizing-adaptativo)
15. [Troubleshooting](#15-troubleshooting)

### PARTE 6: REFERENCIA RÁPIDA
16. [Estructuras de Datos](#16-estructuras-de-datos)
17. [Comandos y Endpoints](#17-comandos-y-endpoints)

---

# PARTE 1: VISIÓN GENERAL

## 1. Introducción

### 1.1 ¿Qué es BOT_trading?

BOT_trading es un sistema automatizado de trading en futuros de criptomonedas que opera 24/7 sin intervención humana. Gestiona múltiples estrategias simultáneamente en diferentes timeframes (4H, 1H, 6Hutc, 2m, 5m) con **position sizing adaptativo personalizado por estrategia** según las condiciones del mercado.

**Características principales:**
- 14+ estrategias multi-timeframe
- **Custom multipliers: cada estrategia define su comportamiento según régimen de mercado**
- Gestión automática de TP/SL/Timeout
- Multi-cuenta (00, E1, 01)
- Dashboard web en tiempo real
- Estado persistente (recuperación tras crash)
- Tracking completo de condiciones de mercado en cada trade

### 1.2 Flujo Simplificado
```
VELA CIERRA → DETECTAR RÉGIMEN Y DIRECCIÓN MERCADO
    ↓
PARA CADA ESTRATEGIA:
├─ Leer regime_trending/ranging/volatile (del YAML)
├─ Leer direction_mode (del YAML)
├─ PositionSizer calcula multipliers
├─ Calcular: adjusted_amount = base × regime_mult × direction_mult
├─ Guardar market_direction en posición
└─ Si multiplier != 0 → Buscar señales con adjusted_amount
```

---

## 2. Arquitectura

### 2.1 Estructura de Directorios
```
bitget/
├── BOT_trading/                    # 🤖 Producción
│   ├── config/
│   │   └── settings.py             # DIRECTION_MATRIX, REGIME_GENERAL
│   ├── core/
│   │   └── orchestrator.py         # Orquestación
│   ├── strategies/
│   │   ├── strategies.yaml         # regime_trending/ranging/volatile + direction_mode
│   │   └── strategy_registry.py
│   ├── market_regime/
│   │   ├── regime_classifier.py    # Detecta régimen y dirección
│   │   ├── regime_metrics.py       # Calcula métricas técnicas
│   │   └── position_sizer.py       # Cálculo multipliers
│   ├── execution/
│   │   ├── position_tracker.py     # Guarda market_direction
│   │   ├── order_manager.py        # Extrae market_direction
│   │   └── trade_logger.py         # Escribe MARKET_DIRECTION a Excel
│   ├── state/
│   │   └── state_manager.py        # Persiste market_direction en JSON
│   ├── api/
│   │   ├── backend.py
│   │   └── templates/
│   │       └── dashboard.html
│   └── signals/                    # ← Symlink a ../signals/
│
├── signals/                        # 🔄 Compartido
│   ├── add_signals_double_top.py
│   ├── add_signals_reversal.py
│   └── ...
│
└── development/                    # 🛠️ Desarrollo
    └── backtesters/
```

### 2.2 Flujo de Datos
```
Market Data (Bitget API)
    ↓
Market Regime Classifier
├─ Calcular Hurst, ER, ATR, PE
├─ Clasificar régimen: trending/ranging/volatile
├─ Detectar dirección: uptrend/dwtrend (price vs MA50)
└─ Cachear en orchestrator
    ↓
PositionSizer
├─ Lookup regime_trending/ranging/volatile (del YAML)
├─ Lookup DIRECTION_MATRIX[direction_mode][market_direction]
├─ Calcular: final_mult = regime_mult × direction_mult
└─ Retornar adjusted_order_amount + metadata (incluye market_direction)
    ↓
Strategy Processor
├─ Detectar señales
├─ Abrir posición con adjusted_order_amount
└─ Pasar market_direction a position_tracker
    ↓
Position Tracker
├─ Guardar market_direction en diccionario de posición
└─ Persistir en state_manager
    ↓
State Manager
└─ Guardar market_direction en bot_state.json
    ↓
(Al cerrar posición)
Order Manager
├─ Extraer market_direction de posición
└─ Pasar a trade_logger
    ↓
Trade Logger
└─ Escribir columna MARKET_DIRECTION en Excel
```

### 2.3 Separación de Responsabilidades

| Módulo | Responsabilidad |
|--------|-----------------|
| **config/** | DIRECTION_MATRIX, REGIME_GENERAL, configuración |
| **core/** | Orquestación, cache de régimen/dirección |
| **market_regime/** | Clasificación mercado + cálculo sizing |
| **strategies/** | Definición regime_trending/ranging/volatile + direction_mode |
| **execution/** | API Bitget + tracking de market_direction |
| **state/** | Persistencia de market_direction |
| **api/** | Dashboard + endpoints |
| **signals/** | Funciones de señales (compartido) |

---

## 3. Stack Tecnológico

### 3.1 Lenguajes y Frameworks

- **Python 3.12**
- **Flask 3.x** (Dashboard)
- **ccxt** (OHLCV data)
- **requests** (Bitget API)

### 3.2 Librerías Clave

| Librería | Uso |
|----------|-----|
| pandas | Procesamiento datos |
| numpy | Arrays numéricos |
| yaml | Parsing estrategias |
| nolds | Hurst exponent |
| ta | ATR (pandas_ta) |
| neurokit2 | Permutation Entropy |

### 3.3 APIs

**Bitget API:**
- Base URL: `https://api.bitget.com`
- Auth: HMAC SHA256
- Product: USDT-FUTURES

**Endpoints:**
- `POST /api/v2/mix/order/place`
- `GET /api/v2/mix/position/all-position`
- `GET /api/v2/mix/account/account`
- `GET /api/v2/mix/market/candles`

---

# PARTE 2: MARKET REGIME SYSTEM

## 4. Clasificación de Mercado

### 4.1 Métricas Calculadas

El sistema calcula 4 métricas en los últimos N períodos de BTCUSDT:

**1. Hurst Exponent**
```python
H = hurst_rs(log_returns, window=100)
```
- **Rango:** 0.0 - 1.0
- **Uso:** Detectar persistencia tendencial
- H > 0.5 = trending (tendencia persistente)
- H < 0.5 = mean-reverting (reversión a la media)

**2. Efficiency Ratio (ER)**
```python
ER = abs(close[-1] - close[-window]) / sum(abs(price_changes))
```
- **Rango:** 0.0 - 1.0
- **Uso:** Medir calidad direccional
- 0 = completamente lateral
- 1 = tendencia perfecta

**3. ATR Normalizado**
```python
ATR_normalized = (ATR_14 / close[-1]) * 100
```
- **Rango:** 0-15%
- **Uso:** Detectar volatilidad extrema

**4. Permutation Entropy (PE)**
```python
PE = entropy(permutations(log_returns, order=3))
```
- **Rango:** 0.0 - 1.0
- **Uso:** Detectar aleatoriedad
- 0 = predecible
- 1 = aleatorio

### 4.2 Reglas de Clasificación
```python
# config/settings.py
REGIME_FAMILIES = {
    'trending': {
        'hurst': ('>', 0.55),
        'efficiency_ratio': ('>', 0.4)
    },
    'volatile': {
        'atr_pct': ('>', 2.0),
        'permutation_entropy': ('>', 0.2)
    },
    'ranging': {}  # Default
}
```

**Orden de evaluación (first-match-wins):**
1. TRENDING: if Hurst > 0.55 AND ER > 0.4
2. VOLATILE: elif ATR > 2.0% AND PE > 0.2
3. RANGING: else (fallback)

### 4.3 Detección de Dirección

**Basada en precio vs MA50:**
```python
if BTC_price > MA50:
    market_direction = 'uptrend'
else:
    market_direction = 'dwtrend'
```

**Logs incluyen contexto completo:**
```
[REGIME] 1H: REGIME=TRENDING, DIRECTION=UPTREND (BTC=$94356.80, MA50=$92145.23, hurst=0.67, er=0.58)
```

### 4.4 Símbolo de Referencia

**`REGIME_REFERENCE_SYMBOL = 'BTCUSDT'`**

Todas las estrategias usan BTCUSDT como referencia, independientemente del símbolo que tradeen.

**Razones:**
- Mayor liquidez y volumen
- Representa sentimiento general del mercado crypto
- Evita ruido de símbolos de baja liquidez
- Eficiencia: 1 solo fetch por timeframe

---

## 5. Matriz de Régimen Custom

### 5.1 Concepto

Sistema bidimensional que ajusta position sizing basado en:
- **Régimen de mercado:** trending/ranging/volatile (calculado del mercado)
- **Dirección de mercado:** uptrend/dwtrend (calculado del mercado)
- **Multipliers de estrategia:** regime_trending/ranging/volatile (definidos en YAML)
- **Modo dirección de estrategia:** long_only/short_only/general (definido en YAML)

**Fórmula:**
```python
final_multiplier = regime_multiplier × direction_multiplier
adjusted_amount = base_amount × final_multiplier
```

**El `market_direction` se guarda en cada posición** para tracking histórico.

### 5.2 Definición de Matrices
```python
# config/settings.py

# Matriz de Dirección
DIRECTION_MATRIX = {
    'long_only': {
        'uptrend': 1.0,    # Long en uptrend = Favorable
        'dwtrend': 0.0     # Long en downtrend = BLOQUEAR
    },
    'short_only': {
        'uptrend': 0.0,    # Short en uptrend = BLOQUEAR
        'dwtrend': 1.0     # Short en downtrend = Favorable
    }
}

# Fallbacks globales
REGIME_GENERAL = {
    'trending': 1.0,
    'ranging': 1.0,
    'volatile': 1.0,
}

DIRECTION_GENERAL = {
    'uptrend': 1.0,
    'dwtrend': 1.0
}
```

### 5.3 Lógica de Aplicación (PositionSizer)
```python
# En market_regime/position_sizer.py

class PositionSizer:
    def calculate_adjusted_amount(
        self,
        base_amount: float,
        regime_trending: float,        # Del YAML
        regime_ranging: float,         # Del YAML
        regime_volatile: float,        # Del YAML
        direction_mode: str,           # Del YAML: 'long_only'/'short_only'/'general'
        market_regime: str,            # Del clasificador: 'trending'/'ranging'/'volatile'
        market_direction: str          # Del clasificador: 'uptrend'/'dwtrend'
    ):
        # 1. Regime multiplier (usa valores del YAML según market_regime)
        if market_regime == 'trending':
            regime_mult = regime_trending
        elif market_regime == 'ranging':
            regime_mult = regime_ranging
        elif market_regime == 'volatile':
            regime_mult = regime_volatile
        
        # Si no tiene valores definidos, usa REGIME_GENERAL
        if not (regime_trending or regime_ranging or regime_volatile):
            regime_mult = REGIME_GENERAL[market_regime]
        
        # 2. Direction multiplier
        if direction_mode == 'general':
            direction_mult = DIRECTION_GENERAL[market_direction]
        elif direction_mode:
            direction_mult = DIRECTION_MATRIX[direction_mode][market_direction]
        else:
            direction_mult = 1.0
        
        # 3. Combined
        final_mult = regime_mult * direction_mult
        adjusted_amount = base_amount * final_mult
        
        # 4. Build metadata (incluye market_direction para tracking)
        metadata = {
            'market_direction': market_direction,
            'direction_multiplier': direction_mult,
            'regime_multiplier': regime_mult,
            'final_multiplier': final_mult,
            'adjusted_amount': adjusted_amount,
            'blocked': (final_mult == 0)
        }
        
        return adjusted_amount, metadata
```

### 5.4 Ejemplos Prácticos

**Ejemplo 1: Estrategia Long en Mercado Trending Downtrend**
```
Estrategia: 06_reversal_long_1H
- regime_trending: 1.8
- direction_mode: 'long_only'
- order_amount: 80 USDT

Mercado: TRENDING + DWTREND
- BTC: $91,086, MA50: $93,446 (BTC < MA50 → dwtrend)

Cálculo:
- regime_mult = regime_trending = 1.8
- direction_mult = DIRECTION_MATRIX['long_only']['dwtrend'] = 0.0
- final_mult = 1.8 × 0.0 = 0.0
- adjusted = 80 × 0.0 = 0 USDT

→ Estrategia BLOQUEADA (no se buscan señales)
→ Log: "[SIZING] Skip 06_...: regime=trending(1.8x), dir=dwtrend(0x), final=0x → BLOCKED"
→ market_direction = 'dwtrend' (se guardaría si se abriera)
```

**Ejemplo 2: Estrategia Short en Mercado Trending Downtrend**
```
Estrategia: 07_reversal_short_1H
- regime_trending: 1.0
- direction_mode: 'short_only'
- order_amount: 80 USDT

Mercado: TRENDING + DWTREND

Cálculo:
- regime_mult = regime_trending = 1.0
- direction_mult = DIRECTION_MATRIX['short_only']['dwtrend'] = 1.0
- final_mult = 1.0 × 1.0 = 1.0
- adjusted = 80 × 1.0 = 80 USDT

→ Posición abierta con 80 USDT
→ Log: "[SIZING] 07_...: Market=[trending, dwtrend] | Base=$80 × regime(1.0x) × dir(1.0x) = $80"
→ market_direction = 'dwtrend' (se guarda en posición)
```

**Ejemplo 3: Sin Valores Específicos = Fallback Global**
```
Estrategia: 12_legacy_strategy_4H
- regime_trending: null
- regime_ranging: null
- regime_volatile: null
- direction_mode: null
- order_amount: 50 USDT

Mercado: TRENDING + UPTREND

Cálculo:
- regime_mult = REGIME_GENERAL['trending'] = 1.0
- direction_mult = DIRECTION_GENERAL['uptrend'] = 1.0
- final_mult = 1.0 × 1.0 = 1.0
- adjusted = 50 × 1.0 = 50 USDT

→ Usa multiplicadores globales (backward compatible)
→ market_direction = 'uptrend' (se guarda en posición)
```

### 5.5 Tracking de Market Direction

**Flujo completo:**
```
orchestrator → calcula market_direction ('uptrend'/'dwtrend')
    ↓
position_sizer → incluye en metadata['market_direction']
    ↓
strategy_processor → pasa market_direction a position_tracker
    ↓
position_tracker → guarda en posición: {'market_direction': 'dwtrend'}
    ↓
state_manager → persiste en JSON: "market_direction": "dwtrend"
    ↓
(al cerrar posición)
order_manager → extrae position_data.get('market_direction')
    ↓
trade_logger → escribe Excel: columna 'MARKET_DIRECTION'
```

**Resultado:** Cada trade en el Excel tiene registrado el `market_direction` que había cuando se abrió la posición.

---

## 6. Configuración de Régimen

### 6.1 Archivo settings.py

**Ubicación:** `config/settings.py`
```python
# ===== MARKET REGIME CONFIGURATION =====

# Símbolo de referencia
REGIME_REFERENCE_SYMBOL = 'BTCUSDT'

# Windows para métricas
REGIME_HURST_WINDOW = 100
REGIME_ER_WINDOW = 14
REGIME_ATR_WINDOW = 14
REGIME_PE_WINDOW = 50
REGIME_PE_ORDER = 3

# Thresholds para clasificación
REGIME_FAMILIES = {
    'trending': {
        'hurst': ('>', 0.55),
        'efficiency_ratio': ('>', 0.4)
    },
    'volatile': {
        'atr_pct': ('>', 2.0),
        'permutation_entropy': ('>', 0.2)
    },
    'ranging': {}
}

# Multiplicadores globales (fallback)
REGIME_GENERAL = {
    'trending': 1.0,
    'ranging': 1.0,
    'volatile': 1.0,
}

DIRECTION_GENERAL = {
    'uptrend': 1.0,
    'dwtrend': 1.0
}

# Matriz de Dirección
DIRECTION_MATRIX = {
    'long_only': {
        'uptrend': 1.0,
        'dwtrend': 0.0
    },
    'short_only': {
        'uptrend': 0.0,
        'dwtrend': 1.0
    }
}
```

### 6.2 Modificar Configuración

**Para ajustar thresholds:**
```python
# Más restrictivo en trending
'trending': {
    'hurst': ('>', 0.60),
    'efficiency_ratio': ('>', 0.50)
}
```

**Para ajustar multiplicadores de dirección:**
```python
# Más agresivo en dirección favorable
'long_only': {
    'uptrend': 1.5,  # Antes 1.0
    'dwtrend': 0.0
}
```

### 6.3 Validación

El sistema valida automáticamente en startup:
- `direction_mode` existe en DIRECTION_MATRIX
- `regime_trending/ranging/volatile` son números válidos (>= 0)
- Coherencia entre `direction` y `direction_mode`
- Todos los campos requeridos presentes

Si validación falla → Error crítico + shutdown

---

# PARTE 3: COMPONENTES CORE

## 7. BotOrchestrator

### 7.1 Responsabilidades

**core/orchestrator.py** es el cerebro del sistema:

1. Inicialización y configuración
2. Main loop infinito
3. Coordinación de estrategias
4. **Cache de régimen y dirección por timeframe**
5. **Delegación de sizing a PositionSizer**
6. **Paso de market_direction a strategy_processor**
7. Sincronización con broker

### 7.2 Variables de Estado
```python
class BotOrchestrator:
    def __init__(self, ...):
        self.open_positions = {}           # {strategy_id: [positions]}
        self.strategy_candles = {}         # {strategy_id: counter}
        self.strategies = []               # Cargadas desde YAML
        self.regime_cache = {}             # {timeframe: regime_str}
        self.direction_cache = {}          # {timeframe: 'uptrend'/'dwtrend'}
        self.position_sizer = None         # PositionSizer instance
```

**Caches:** Dict en memoria (NO persisten)
- Se recalculan cada vela cerrada
- Compartidos entre estrategias del mismo timeframe
- `direction_cache` almacena 'uptrend' o 'dwtrend'

### 7.3 Método `_update_regime_for_timeframes()`

Calcula y cachea régimen + dirección tras cerrar vela:
```python
def _update_regime_for_timeframes(self, timeframes):
    for tf in timeframes:
        try:
            # 1. Calculate REGIME
            family, metrics = get_current_regime(tf)
            self.regime_cache[tf] = family
            
            # 2. Calculate DIRECTION (returns price + MA50)
            direction, btc_price, btc_ma50 = get_current_direction(tf)
            self.direction_cache[tf] = direction  # 'uptrend' o 'dwtrend'
            
            # Format for logging
            price_str = f"${btc_price:.2f}" if btc_price else "N/A"
            ma50_str = f"${btc_ma50:.2f}" if btc_ma50 else "N/A"
            
            self.logger.info(
                f"[REGIME] {tf}: REGIME={family.upper()}, "
                f"DIRECTION={direction.upper()} "
                f"(BTC={price_str}, MA50={ma50_str}, "
                f"hurst={metrics.get('hurst', 0):.2f}, "
                f"er={metrics.get('efficiency_ratio', 0):.2f})"
            )
        except Exception as e:
            self.regime_cache[tf] = 'ranging'
            self.direction_cache[tf] = 'uptrend'
```

### 7.4 Método `_search_signals()`

Delegación a PositionSizer y paso de market_direction:
```python
def _search_signals(self, strategies_to_process):
    for strat in strategies_to_process:
        # Skip checks...
        
        # Get market state from cache
        timeframe = strat['timeframe']
        market_regime = self.regime_cache.get(timeframe, 'ranging')
        market_direction = self.direction_cache.get(timeframe, 'uptrend')
        
        # Calculate adjusted amount using PositionSizer
        adjusted_amount, metadata = self.position_sizer.calculate_adjusted_amount(
            base_amount=strat['order_amount'],
            regime_trending=strat.get('regime_trending', 1.0),
            regime_ranging=strat.get('regime_ranging', 1.0),
            regime_volatile=strat.get('regime_volatile', 1.0),
            direction_mode=strat.get('direction_mode', 'general'),
            market_regime=market_regime,
            market_direction=market_direction
        )
        
        # Check if blocked
        if metadata['blocked']:
            log_msg = self.position_sizer.format_log_message(strat['id'], metadata)
            self.logger.info(log_msg)
            continue
        
        # Log sizing decision
        log_msg = self.position_sizer.format_log_message(strat['id'], metadata)
        self.logger.info(log_msg)
        
        # Process strategy (pasa market_direction)
        self.strategy_processor.process(
            strat=strat,
            adjusted_order_amount=adjusted_amount,
            regime_family=metadata['market_regime'],
            regime_multiplier=metadata['regime_multiplier'],
            direction=metadata['market_direction'],           # ← PASA market_direction
            direction_multiplier=metadata['direction_multiplier'],
            ...
        )
```

### 7.5 Timing Crítico

**¿CUÁNDO se calcula régimen y dirección?**
- DESPUÉS del sync con broker
- ANTES de buscar señales
- SOLO cuando cierra vela
- UNA VEZ por timeframe por vela

**Flujo temporal:**
```
15:59:58 - Esperando vela 1H...
16:00:00 - ¡Vela cerrada!
16:00:01 - Sync con broker
16:00:02 - Calcular regime + direction
16:00:03 - Cachear en orchestrator (direction_cache['1H'] = 'dwtrend')
16:00:04 - Para cada estrategia 1H:
           ├─ PositionSizer.calculate_adjusted_amount()
           ├─ metadata incluye market_direction='dwtrend'
           ├─ Check if blocked
           ├─ Buscar señales
           └─ Si abre posición → guardar market_direction='dwtrend'
```

---

## 8. Sistema de Estrategias

### 8.1 Definición en YAML

**strategies/strategies.yaml**
```yaml
strategies:
  # ===== TRENDING STRATEGIES =====
  - id: "06_reversal_long_1H"
    name: "reversal_long_1H"
    function_name: "add_signals_reversal_long"
    timeframe: "1H"
    direction: "long"
    order_amount: 80
    tp_pct: 2.0
    sl_pct: 10.0
    sell_after_ncandles: 50
    symbols: "multi"
    active: true
    regime_trending: 1.8     # Multiplier cuando market_regime='trending'
    regime_ranging: 0        # Multiplier cuando market_regime='ranging' (BLOQUEA)
    regime_volatile: 1.0     # Multiplier cuando market_regime='volatile'
    direction_mode: "long_only"  # Usa DIRECTION_MATRIX['long_only']
    
  # ===== RANGING STRATEGIES =====
  - id: "07_reversal_short_1H"
    name: "reversal_short_1H"
    function_name: "add_signals_reversal_short"
    timeframe: "1H"
    direction: "short"
    order_amount: 80
    tp_pct: 1.9
    sl_pct: 5.0
    sell_after_ncandles: 50
    symbols: "multi"
    active: true
    regime_trending: 0       # BLOQUEA en trending
    regime_ranging: 1.5      # Favorece ranging
    regime_volatile: 1.0
    direction_mode: "short_only"  # Usa DIRECTION_MATRIX['short_only']
    
  # ===== LEGACY (sin custom multipliers - usa fallback) =====
  - id: "01_double_top_long_4H"
    name: "double_top_long_4H"
    function_name: "add_signals_double_top_long"
    timeframe: "4H"
    direction: "long"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    sell_after_ncandles: 50
    symbols: "multi"
    active: true
    # regime_trending: null    # Sin definir → usa REGIME_GENERAL
    # regime_ranging: null
    # regime_volatile: null
    # direction_mode: null     # Sin definir → usa DIRECTION_GENERAL
```

### 8.2 Campos Disponibles

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `id` | String | Sí | Identificador único (NN_name) |
| `name` | String | Sí | Nombre estrategia |
| `function_name` | String | Sí | Nombre en strategy_registry |
| `timeframe` | String | Sí | 4H, 1H, 6Hutc, etc. |
| `direction` | String | Sí | long o short |
| `order_amount` | Float | Sí | Monto base en USDT |
| `tp_pct` | Float | Sí | Take Profit % |
| `sl_pct` | Float | Sí | Stop Loss % |
| `sell_after_ncandles` | Int | Sí | Velas hasta timeout |
| `symbols` | String | Sí | "multi" o lista |
| `active` | Bool | Sí | true o false |
| `regime_trending` | Float | No | Multiplier para mercado trending |
| `regime_ranging` | Float | No | Multiplier para mercado ranging |
| `regime_volatile` | Float | No | Multiplier para mercado volatile |
| `direction_mode` | String | No | long_only/short_only/general |

### 8.3 Validación

El sistema valida en startup:
- `direction_mode` existe en DIRECTION_MATRIX (si está definido)
- `regime_trending/ranging/volatile` son números >= 0 (si están definidos)
- `direction` coherente con `direction_mode` y nombre
- `function_name` existe en IMPLEMENTED_STRATEGIES
- Todos los campos requeridos presentes

Si falla → Error crítico + shutdown

---

## 9. Dashboard Web

### 9.1 Endpoints de Régimen

**api/backend.py**
```python
@app.route('/api/regime/matrix', methods=['GET'])
def get_regime_matrix():
    return jsonify({
        'direction_matrix': DIRECTION_MATRIX,
        'regime_general': REGIME_GENERAL,
        'direction_general': DIRECTION_GENERAL
    })

@app.route('/api/regime/current', methods=['GET'])
def get_current_regime():
    tf = request.args.get('timeframe', '1H')
    info = get_regime_info(tf)
    
    return jsonify({
        'timeframe': tf,
        'regime': info['family'],
        'direction': info['btc_trend'],  # 'uptrend' o 'downtrend'
        'btc_price': info['btc_price'],
        'btc_ma50': info['btc_ma50'],
        'metrics': info['metrics'],
        'timestamp': datetime.now().isoformat()
    })
```

### 9.2 Visualización

Dashboard muestra en tiempo real:
- Régimen actual por timeframe
- Dirección de mercado (uptrend/dwtrend)
- BTC price vs MA50
- Matriz de multiplicadores
- Estrategias con multipliers aplicados
- Posiciones abiertas con su market_direction

---

# PARTE 4: CONFIGURACIÓN

## 10. Settings.py

**Ubicación:** `config/settings.py`

### 10.1 Configuración de Cuentas
```python
ACCOUNTS = {
    '00': {
        'initial_capital': 3671,
        'dashboard_port': 5000,
        'description': 'Main Account'
    },
    'E1': {
        'initial_capital': 1761,
        'dashboard_port': 5001,
        'description': 'Elite Account'
    },
    '01': {
        'initial_capital': 117,
        'dashboard_port': 5099,
        'description': 'Testing Account'
    }
}
```

### 10.2 Validación Settings
```python
MIN_ORDER_AMOUNT = 40
MAX_ORDER_AMOUNT = 100
MIN_TP_PCT = 1.5
MAX_TP_PCT = 10
MIN_SL_PCT = 1.5
MAX_SL_PCT = 15
MIN_CANDLES = 49
MAX_CANDLES = 51
VALID_TIMEFRAMES = ['1H', '4H', '6Hutc']
```

---

## 11. Strategies.yaml

Ver sección 8.1 para estructura completa.

---

## 12. Alta de Estrategias

### 12.1 Proceso Completo

**Paso 1: Crear función de señal**
```python
# bitget/signals/add_signals_mi_estrategia.py
import numpy as np

def add_signals_mi_estrategia_long(data):
    close = data['close']
    signals = np.zeros(len(close))
    
    # Lógica de señal
    for i in range(50, len(close)):
        if condition_long:
            signals[i] = 1
    
    return signals
```

**Paso 2: Registrar en strategy_registry.py**
```python
IMPLEMENTED_STRATEGIES = {
    'add_signals_mi_estrategia_long': add_signals_mi_estrategia_long,
}
```

**Paso 3: Añadir a strategies.yaml**
```yaml
- id: "18_mi_estrategia_long_1H"
  function_name: "add_signals_mi_estrategia_long"
  timeframe: "1H"
  order_amount: 80
  regime_trending: 1.5
  regime_ranging: 0.5
  regime_volatile: 1.0
  direction_mode: "long_only"
  # ... otros campos
```

**Paso 4: Crear fichero de símbolos**
```
symbols_live/symbols_live_18_mi_estrategia_long_1H_1H.xlsx
```

**Paso 5: Validar**
```bash
python3 main.py --account 01
# Verificar logs de validación
```

### 12.2 Checklist

- [ ] Función creada en `signals/`
- [ ] Registrada en `strategy_registry.py`
- [ ] Añadida a `strategies.yaml`
- [ ] `regime_trending/ranging/volatile` definidos
- [ ] `direction_mode` definido
- [ ] Fichero símbolos creado
- [ ] Validación pasa en startup
- [ ] Probada en cuenta 01

---

# PARTE 5: FLUJOS Y OPERACIÓN

## 13. Ciclo de Vida

### 13.1 Inicialización
```
STARTUP
├─ Parse args
├─ Cargar configuración
├─ Validar estrategias (incluye direction_mode)
├─ Validar símbolos por estrategia
├─ Inicializar PositionSizer
├─ Conectar a API
├─ Inicializar dashboard
├─ Cargar estado previo (incluye market_direction de posiciones)
└─ Calcular próximas velas

RUNNING
└─ Main loop infinito

SHUTDOWN
├─ Guardar estado (incluye market_direction)
└─ Exit graceful
```

### 13.2 Main Loop
```python
while True:
    now = datetime.now(tz=UTC)
    
    closed_timeframes = check_closed_candles()
    
    if closed_timeframes:
        sync_broker()
        update_regime_for_timeframes(closed_timeframes)  # Calcula direction_cache
        process_strategies(closed_timeframes)            # Usa direction_cache
    else:
        check_tp_sl_periodic()
    
    time.sleep(0.05)
```

---

## 14. Position Sizing Adaptativo

### 14.1 Flujo Completo con Market Direction
```
VELA CIERRA
    ↓
1. SYNC BROKER
    ↓
2. UPDATE REGIME + DIRECTION
├─ Fetch BTCUSDT OHLCV
├─ Calcular métricas
├─ Clasificar régimen
├─ Calcular dirección (BTC vs MA50)
├─ Cachear regime_cache['1H'] = 'trending'
└─ Cachear direction_cache['1H'] = 'dwtrend'
    ↓
3. PARA CADA ESTRATEGIA:
├─ Obtener market_regime del cache
├─ Obtener market_direction del cache
├─ PositionSizer.calculate_adjusted_amount()
│   ├─ regime_mult = regime_trending (del YAML)
│   ├─ direction_mult = DIRECTION_MATRIX[direction_mode][market_direction]
│   └─ metadata['market_direction'] = 'dwtrend'
├─ Si blocked (mult=0) → Skip
└─ Buscar señales con adjusted_amount
    ↓
4. SI SE ABRE POSICIÓN:
├─ position_tracker.add_position()
│   └─ Guarda: {'market_direction': 'dwtrend', ...}
├─ state_manager.save_state()
│   └─ Persiste en JSON: "market_direction": "dwtrend"
    ↓
5. AL CERRAR POSICIÓN:
├─ order_manager.close_position()
│   └─ Extrae: position_data.get('market_direction')
└─ trade_logger.log_closed_position()
    └─ Escribe Excel: columna 'MARKET_DIRECTION' = 'dwtrend'
```

### 14.2 Logs de Régimen y Dirección

**Formato de logs:**
```
[REGIME] Updating regime & direction for: ['1H']
[REGIME] 1H: REGIME=TRENDING, DIRECTION=DWTREND (BTC=$91086.10, MA50=$93446.76, hurst=0.81, er=0.66)
[SIZING] 06_reversal_long_1H: Market=[trending, dwtrend] | Base=$80 × regime(1.8x) × dir(0.0x) = $0
[SIZING] Skip 06_reversal_long_1H: regime=trending(1.8x), dir=dwtrend(0x), final=0x → BLOCKED
[SIZING] Skip 07_reversal_short_1H: regime=trending(0x), dir=dwtrend(1.0x), final=0x → BLOCKED
```

**Explicación:**
- `REGIME=TRENDING`: Mercado clasificado como trending
- `DIRECTION=DWTREND`: BTC por debajo de MA50
- `BTC=$91086.10, MA50=$93446.76`: Precio y media móvil
- `regime(1.8x)`: Multiplier del YAML (regime_trending)
- `dir(0x)`: Multiplier de DIRECTION_MATRIX (long_only + dwtrend = 0)
- `final=0x → BLOCKED`: Estrategia bloqueada

---

## 15. Troubleshooting

### 15.1 Market Direction No Se Guarda

**Síntoma:** Posiciones en Excel sin columna `MARKET_DIRECTION` o con valor `unknown`

**Diagnóstico:**
```bash
# Verificar que se pasa correctamente
grep "market_direction" persistence/bot_files_XX/BOT_orchestator_XX.log

# Verificar JSON de estado
cat persistence/bot_state_XX.json | python3 -m json.tool | grep market_direction
```

**Solución:** Verificar que toda la cadena está actualizada (ver sección 5.5)

### 15.2 Direction Cache Vacío

**Síntoma:** Logs "direction_cache.get() returned None"

**Diagnóstico:**
```bash
grep "direction_cache\|DIRECTION=" persistence/bot_files_XX/BOT_orchestator_XX.log
```

**Solución:** Verificar que `get_current_direction()` se ejecuta correctamente

### 15.3 Estrategia Bloqueada por Dirección

**Síntoma:** Logs "dir=dwtrend(0x) → BLOCKED"

**Diagnóstico:** Verificar DIRECTION_MATRIX

**Solución:** Esto es correcto si quieres bloquear long en downtrend. Para cambiar:
```python
# En config/settings.py
'long_only': {
    'uptrend': 1.0,
    'dwtrend': 0.5  # Permitir con penalización
}
```

### 15.4 Excel Sin Columna MARKET_DIRECTION

**Síntoma:** Archivo Excel no tiene columna `MARKET_DIRECTION`

**Solución:** Verificar en `trade_logger.py` línea ~168:
```python
'MARKET_DIRECTION': market_direction if market_direction else 'unknown',
```

### 15.5 Validación Falla

**Síntoma:** Bot no arranca, error de validación

**Solución:**
1. Verificar `direction_mode` es válido (long_only/short_only/general)
2. Verificar `regime_trending/ranging/volatile` son números >= 0
3. Verificar coherencia direction/direction_mode
4. Corregir YAML y reiniciar

---

# PARTE 6: REFERENCIA RÁPIDA

## 16. Estructuras de Datos

### 16.1 Strategy Dict
```python
{
    'id': '06_reversal_long_1H',
    'name': 'reversal_long_1H',
    'function_name': 'add_signals_reversal_long',
    'timeframe': '1H',
    'direction': 'long',
    'order_amount': 80.0,
    'tp_pct': 2.0,
    'sl_pct': 10.0,
    'sell_after_ncandles': 50,
    'symbols': 'multi',
    'active': True,
    'regime_trending': 1.8,
    'regime_ranging': 0.0,
    'regime_volatile': 1.0,
    'direction_mode': 'long_only'
}
```

### 16.2 PositionSizer Metadata
```python
{
    'base_amount': 80.0,
    'market_regime': 'trending',
    'market_direction': 'dwtrend',           # ← TRACKING
    'regime_multiplier': 1.8,
    'regime_source': 'strategy YAML',
    'direction_multiplier': 0.0,
    'direction_source': 'MATRIX[long_only][dwtrend]',
    'final_multiplier': 0.0,
    'adjusted_amount': 0.0,
    'blocked': True
}
```

### 16.3 Position Dict
```python
{
    'strategy_id': '06_reversal_long_1H',
    'symbol': 'BTCUSDT',
    'size': 0.05,
    'entry_price': 91086.10,
    'direction': 'long',
    'opened_at': '2026-01-20 09:00:45',
    'usdt_amount': 80.0,
    'market_direction': 'dwtrend',           # ← GUARDADO
    'direction_multiplier': 0.0,
    'regime_family': 'trending',
    'regime_multiplier': 1.8,
    # ... otros campos
}
```

### 16.4 Caches
```python
# Orchestrator
self.regime_cache = {
    '4H': 'ranging',
    '1H': 'trending',
    '6Hutc': 'volatile'
}

self.direction_cache = {
    '4H': 'dwtrend',
    '1H': 'uptrend',
    '6Hutc': 'uptrend'
}
```

### 16.5 Excel Output
```
| MARKET_DIRECTION | DIRECTION_MULTIPLIER | REGIME_FAMILY | REGIME_MULTIPLIER |
|------------------|---------------------|---------------|-------------------|
| dwtrend          | 0.0                 | trending      | 1.8               |
| uptrend          | 1.5                 | ranging       | 1.0               |
```

---

## 17. Comandos y Endpoints

### 17.1 Comandos CLI
```bash
# Iniciar bot
python3 main.py --account 00

# Ver logs completos
tail -f persistence/bot_files_00/BOT_orchestator_00.log

# Filtrar logs de sizing y dirección
grep "SIZING\|DIRECTION" persistence/bot_files_00/BOT_orchestator_00.log

# Verificar market_direction en estado
cat persistence/bot_state_00.json | python3 -m json.tool | grep market_direction
```

### 17.2 API Endpoints
```
GET http://localhost:5000/api/regime/current?timeframe=1H
  → Retorna: regime, direction ('uptrend'/'downtrend'), btc_price, btc_ma50

GET http://localhost:5000/api/regime/matrix
  → Retorna: DIRECTION_MATRIX, REGIME_GENERAL, DIRECTION_GENERAL

GET http://localhost:5000/api/positions
  → Incluye market_direction por posición

GET http://localhost:5000/api/status
```

### 17.3 Testing Rápido
```python
# Test position sizer con market_direction
from market_regime import PositionSizer
import logging

logger = logging.getLogger('test')
sizer = PositionSizer(logger)

amount, meta = sizer.calculate_adjusted_amount(
    base_amount=80.0,
    regime_trending=1.8,
    regime_ranging=0.0,
    regime_volatile=1.0,
    direction_mode='long_only',
    market_regime='trending',
    market_direction='dwtrend'
)

print(f"Adjusted: ${amount:.2f}")
print(f"Market direction: {meta['market_direction']}")
print(f"Direction mult: {meta['direction_multiplier']:.1f}x")
print(f"Final mult: {meta['final_multiplier']:.1f}x")
print(f"Blocked: {meta['blocked']}")
```

---

# 🎉 FIN DE DOCUMENTACIÓN

**BOT_trading v2.5 - Sistema de Trading Automatizado**

---

**Última actualización:** 2026-01-20  
**Autor:** Trading Bot Team  
**Nueva Feature:** Market Direction Tracking (uptrend/dwtrend)

---

## 📝 RESUMEN EJECUTIVO

### Cambios Principales v2.5

1. **Market Direction Tracking:** Sistema unificado de `market_direction` en toda la cadena
2. **Persistencia completa:** market_direction se guarda en JSON y Excel
3. **Logs mejorados:** Incluyen BTC price, MA50 y dirección en cada ciclo
4. **Cadena de datos verificada:** 7 pasos desde orchestrator hasta Excel

### Quick Start
```bash
# 1. Actualizar strategies.yaml
regime_trending: 1.8
regime_ranging: 0
regime_volatile: 1.0
direction_mode: "long_only"

# 2. Reiniciar bot
python3 main.py --account 00

# 3. Verificar logs
grep "DIRECTION\|SIZING" persistence/bot_files_00/BOT_orchestator_00.log

# 4. Verificar Excel
# Columna MARKET_DIRECTION debe tener 'uptrend' o 'dwtrend'
```

---

**¿Preguntas? Consulta [Troubleshooting](#15-troubleshooting) o [Position Sizing Adaptativo](#14-position-sizing-adaptativo).**
