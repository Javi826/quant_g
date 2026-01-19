<artifact identifier="bot-trading-doc-v24-complete" type="text/markdown" title="BOT_trading_Documentation_v2.4.md">
# 🤖 BOT TRADING - DOCUMENTACIÓN TÉCNICA v2.4

**Sistema de Trading Automatizado con Custom Regime Multipliers por Estrategia**

---

**Versión:** 2.4  
**Fecha:** 2026-01-19  
**Python:** 3.12  
**Framework:** Flask + ccxt + Bitget API  
**Nuevo:** Arquitectura modular de position sizing y logs mejorados

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
- **Custom multipliers: cada estrategia define su comportamiento según régimen de mercado** ⭐
- Gestión automática de TP/SL/Timeout
- Multi-cuenta (00, E1, 01)
- Dashboard web en tiempo real
- Estado persistente (recuperación tras crash)

### 1.2 Novedades v2.4

**Arquitectura modular de position sizing:**

El sistema ahora separa la lógica de cálculo de multiplicadores en un módulo independiente (`market_regime/position_sizer.py`), mejorando:
- Testabilidad independiente
- Reutilización en backtester
- Mantenibilidad del código
- Separación de responsabilidades

**Logs mejorados con contexto de mercado:**

Los logs de dirección ahora incluyen precio de BTC y MA50:
```
[REGIME] 1H: REGIME=TRENDING, DIRECTION=UPTREND (BTC=$94356.80, MA50=$92145.23)
```

### 1.3 Flujo Simplificado
```
VELA CIERRA → DETECTAR RÉGIMEN MERCADO
    ↓
PARA CADA ESTRATEGIA:
├─ Leer strategy.regime_family (del YAML)
├─ Leer strategy.dir_mode (del YAML)
├─ PositionSizer calcula multipliers
├─ Calcular: adjusted_amount = base * regime_mult * dir_mult
└─ Si multiplier != 0 → Buscar señales con adjusted_amount
```

---

## 2. Arquitectura

### 2.1 Estructura de Directorios
```
bitget/
├── BOT_trading/                    # 🤖 Producción
│   ├── config/
│   │   └── settings.py             # REGIME_MATRIX, DIRECTION_MATRIX
│   ├── core/
│   │   └── orchestrator.py         # Orquestación (570 líneas)
│   ├── strategies/
│   │   ├── strategies.yaml         # regime_family + dir_mode ⭐
│   │   └── strategy_registry.py
│   ├── market_regime/              # ⭐ Módulo cohesivo
│   │   ├── regime_classifier.py    # Detecta régimen y dirección
│   │   ├── regime_metrics.py       # Calcula métricas técnicas
│   │   ├── position_sizer.py       # ⭐ NUEVO: Cálculo multipliers
│   │   └── __init__.py             # Exporta funciones públicas
│   ├── execution/
│   │   └── bitget_client.py
│   ├── api/
│   │   ├── backend.py              # Endpoints regime
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
PositionSizer (módulo especializado)
├─ Lookup REGIME_MATRIX[family][regime]
├─ Lookup DIRECTION_MATRIX[dir_mode][direction]
├─ Calcular: final_mult = regime_mult × direction_mult
└─ Retornar adjusted_order_amount + metadata
    ↓
Strategy Processor
├─ Detectar señales
└─ Abrir posición con adjusted_order_amount
```

### 2.3 Separación de Responsabilidades

| Módulo | Responsabilidad |
|--------|-----------------|
| **config/** | REGIME_MATRIX, DIRECTION_MATRIX, configuración |
| **core/** | Orquestación, cache, coordinación |
| **market_regime/** | Clasificación mercado + cálculo sizing |
| **strategies/** | Definición regime_family + dir_mode en YAML |
| **execution/** | API Bitget |
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
    'volatile': {
        ('atr_pct', '>'): 3.5,
        ('permutation_entropy', '>'): 0.70
    },
    'ranging': {
        ('efficiency_ratio', '<'): 0.45
    },
    'trending': {
        ('efficiency_ratio', '>='): 0.45
    },
    'default': {}
}
```

**Orden de evaluación:**
1. VOLATILE: if ATR > 3.5% AND PE > 0.70
2. RANGING: elif ER < 0.45
3. TRENDING: elif ER >= 0.45
4. DEFAULT: else (fallback)

### 4.3 Detección de Dirección

**Basada en precio vs MA50:**
```python
if BTC_price > MA50:
    direction = 'uptrend'
else:
    direction = 'dwtrend'
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
- **Régimen:** trending/ranging/volatile (del mercado)
- **Dirección:** uptrend/dwtrend (del mercado)
- **Familia estrategia:** trending/ranging/volatile (del YAML)
- **Modo dirección:** long_only/short_only/general (del YAML)

**Fórmula:**
```python
final_multiplier = regime_multiplier × direction_multiplier
adjusted_amount = base_amount × final_multiplier
```

### 5.2 Definición de Matrices
```python
# config/settings.py

# Matriz de Régimen
REGIME_MATRIX = {
    'trending': {
        'trending': 1.8,   # Trending + Trending = MUY agresivo
        'ranging': 1.0,    # Trending + Ranging = Normal
        'volatile': 0.0    # Trending + Volatile = BLOQUEAR
    },
    'ranging': {
        'trending': 1.0,
        'ranging': 1.8,
        'volatile': 0.0
    },
    'volatile': {
        'trending': 0.5,
        'ranging': 0.5,
        'volatile': 1.5
    }
}

# Matriz de Dirección
DIRECTION_MATRIX = {
    'long_only': {
        'uptrend': 1.5,    # Long en uptrend = Favorable
        'dwtrend': 0.0     # Long en downtrend = BLOQUEAR
    },
    'short_only': {
        'uptrend': 0.0,    # Short en uptrend = BLOQUEAR
        'dwtrend': 1.5     # Short en downtrend = Favorable
    }
}

# Fallbacks globales
REGIME_GENERAL = {
    'volatile': 0.5,
    'ranging': 1.0,
    'trending': 1.5,
    'default': 1.0
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
        strategy_family: str,      # 'trending' del YAML
        dir_mode: str,             # 'long_only' del YAML
        market_regime: str,        # 'trending' del clasificador
        market_direction: str      # 'uptrend' del clasificador
    ):
        # 1. Regime multiplier
        if strategy_family == 'general':
            regime_mult = REGIME_GENERAL[market_regime]
        elif strategy_family:
            regime_mult = REGIME_MATRIX[strategy_family][market_regime]
        else:
            regime_mult = REGIME_GENERAL[market_regime]
        
        # 2. Direction multiplier
        if dir_mode == 'general':
            direction_mult = DIRECTION_GENERAL[market_direction]
        elif dir_mode:
            direction_mult = DIRECTION_MATRIX[dir_mode][market_direction]
        else:
            direction_mult = 1.0
        
        # 3. Combined
        final_mult = regime_mult * direction_mult
        adjusted_amount = base_amount * final_mult
        
        return adjusted_amount, metadata
```

### 5.4 Ejemplos Prácticos

**Ejemplo 1: Estrategia Long Trending en Mercado Trending Uptrend**
```
Estrategia: 06_reversal_long_1H
- regime_family: 'trending'
- dir_mode: 'long_only'
- order_amount: 40 USDT

Mercado: TRENDING + UPTREND
- ER: 0.67, BTC: $94356, MA50: $92145

Cálculo:
- regime_mult = MATRIX['trending']['trending'] = 1.8x
- direction_mult = MATRIX['long_only']['uptrend'] = 1.5x
- final_mult = 1.8 × 1.5 = 2.7x
- adjusted = 40 × 2.7 = 108 USDT

→ Posición abierta con 108 USDT (170% más)
```

**Ejemplo 2: Estrategia Long en Mercado Downtrend = Bloqueado**
```
Estrategia: 06_reversal_long_1H
- dir_mode: 'long_only'
- order_amount: 40 USDT

Mercado: DOWNTREND
- BTC: $94356, MA50: $95200

Cálculo:
- regime_mult = 1.8x
- direction_mult = MATRIX['long_only']['dwtrend'] = 0.0x
- final_mult = 1.8 × 0.0 = 0.0x

→ Estrategia bloqueada (no se buscan señales)
→ Log: "[SIZING] Skip 06_...: final=0x → BLOCKED"
```

**Ejemplo 3: Sin Familia = Fallback Global**
```
Estrategia: 12_legacy_strategy_4H
- regime_family: null
- dir_mode: null
- order_amount: 50 USDT

Mercado: TRENDING + UPTREND

Cálculo:
- regime_mult = REGIME_GENERAL['trending'] = 1.5x
- direction_mult = DIRECTION_GENERAL['uptrend'] = 1.0x
- final_mult = 1.5 × 1.0 = 1.5x
- adjusted = 50 × 1.5 = 75 USDT

→ Usa multiplicadores globales (backward compatible)
```

### 5.5 Fallback Global

Para estrategias que NO tienen `regime_family` o `dir_mode` definidos:
```python
# config/settings.py
REGIME_GENERAL = {
    'volatile': 0.5,
    'ranging': 1.0,
    'trending': 1.5,
    'default': 1.0
}

DIRECTION_GENERAL = {
    'uptrend': 1.0,
    'dwtrend': 1.0
}
```

Esto permite **backward compatibility**: estrategias antiguas siguen funcionando.

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
    'volatile': {
        ('atr_pct', '>'): 3.5,
        ('permutation_entropy', '>'): 0.70
    },
    'ranging': {
        ('efficiency_ratio', '<'): 0.45
    },
    'trending': {
        ('efficiency_ratio', '>='): 0.45
    },
    'default': {}
}

# Multiplicadores globales (fallback)
REGIME_GENERAL = {
    'volatile': 0.5,
    'ranging': 1.0,
    'trending': 1.5,
    'default': 1.0
}

DIRECTION_GENERAL = {
    'uptrend': 1.0,
    'dwtrend': 1.0
}

# Matriz de Régimen
REGIME_MATRIX = {
    'trending': {
        'trending': 1.8,
        'ranging': 1.0,
        'volatile': 0.0
    },
    'ranging': {
        'trending': 1.0,
        'ranging': 1.8,
        'volatile': 0.0
    },
    'volatile': {
        'trending': 0.5,
        'ranging': 0.5,
        'volatile': 1.5
    }
}

# Matriz de Dirección
DIRECTION_MATRIX = {
    'long_only': {
        'uptrend': 1.5,
        'dwtrend': 0.0
    },
    'short_only': {
        'uptrend': 0.0,
        'dwtrend': 1.5
    }
}
```

### 6.2 Modificar Configuración

**Para ajustar thresholds:**
```python
# Más restrictivo en volatile
'volatile': {
    ('atr_pct', '>'): 4.0,
    ('permutation_entropy', '>'): 0.75
}
```

**Para ajustar multiplicadores:**
```python
# Más agresivo en trending
'trending': {
    'trending': 2.0,  # Antes 1.8x
    ...
}

# Bloquear combinación específica
'trending': {
    'ranging': 0.0,  # Antes 1.0x
    ...
}
```

### 6.3 Validación

El sistema valida automáticamente en startup:
- Todas las familias en MATRIX existen
- Todos los regímenes en MATRIX existen
- Todos los dir_modes en DIRECTION_MATRIX existen
- Multiplicadores son números válidos (>= 0)

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
6. Sincronización con broker

### 7.2 Variables de Estado
```python
class BotOrchestrator:
    def __init__(self, ...):
        self.open_positions = {}           # {strategy_id: [positions]}
        self.strategy_candles = {}         # {strategy_id: counter}
        self.strategies = []               # Cargadas desde YAML
        self.regime_cache = {}             # {timeframe: regime_str}
        self.direction_cache = {}          # {timeframe: direction_str}
        self.position_sizer = None         # PositionSizer instance
```

**Caches:** Dict en memoria (NO persisten)
- Se recalculan cada vela cerrada
- Compartidos entre estrategias del mismo timeframe

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
            self.direction_cache[tf] = direction
            
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

Delegación a PositionSizer:
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
            strategy_family=strat.get('regime_family'),
            dir_mode=strat.get('dir_mode'),
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
        
        # Process strategy
        self.strategy_processor.process(
            strat=strat,
            adjusted_order_amount=adjusted_amount,
            ...
        )
```

### 7.5 Timing Crítico

**¿CUÁNDO se calcula régimen?**
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
16:00:03 - Cachear en orchestrator
16:00:04 - Para cada estrategia 1H:
           ├─ PositionSizer.calculate_adjusted_amount()
           ├─ Check if blocked
           └─ Buscar señales
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
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    sell_after_ncandles: 50
    symbols: "multi"
    active: true
    regime_family: "trending"    # ⭐ Declara familia
    dir_mode: "long_only"        # ⭐ Declara modo dirección
    
  # ===== RANGING STRATEGIES =====
  - id: "10_parity_reversal_short_1H"
    name: "parity_reversal_short_1H"
    function_name: "add_signals_parity_reversal_short"
    timeframe: "1H"
    direction: "short"
    order_amount: 45
    tp_pct: 3.5
    sl_pct: 10.0
    sell_after_ncandles: 50
    symbols: "multi"
    active: true
    regime_family: "ranging"     # ⭐ Familia ranging
    dir_mode: "short_only"       # ⭐ Solo opera short
    
  # ===== LEGACY (sin family/dir_mode - usa fallback) =====
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
    # regime_family: null  # ⭐ Sin definir → usa REGIME_GENERAL
    # dir_mode: null       # ⭐ Sin definir → usa DIRECTION_GENERAL
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
| `regime_family` | String | No | trending/ranging/volatile/general ⭐ |
| `dir_mode` | String | No | long_only/short_only/general ⭐ |

### 8.3 Validación

El sistema valida en startup:
- `regime_family` existe en REGIME_MATRIX (si está definido)
- `dir_mode` existe en DIRECTION_MATRIX (si está definido)
- `direction` coherente con `dir_mode` y nombre
- `function_name` existe en IMPLEMENTED_STRATEGIES
- Todos los campos requeridos presentes

Si falla → Error crítico + shutdown

---

## 9. Dashboard Web

### 9.1 Nuevos Endpoints

**api/backend.py**
```python
@app.route('/api/regime/matrix', methods=['GET'])
def get_regime_matrix():
    return jsonify({
        'regime_matrix': REGIME_MATRIX,
        'direction_matrix': DIRECTION_MATRIX,
        'regime_general': REGIME_GENERAL,
        'direction_general': DIRECTION_GENERAL
    })

@app.route('/api/regime/current', methods=['GET'])
def get_current_regime():
    tf = request.args.get('timeframe', '1H')
    regime, metrics = get_regime_info(tf)
    
    return jsonify({
        'timeframe': tf,
        'regime': regime,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    })
```

### 9.2 Visualización

Dashboard muestra en tiempo real:
- Régimen actual por timeframe
- Dirección de mercado (BTC price vs MA50)
- Matriz de multiplicadores
- Estrategias agrupadas por familia
- Multipliers aplicados en última ejecución

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
MAX_SL_PCT = 10
MIN_CANDLES = 49
MAX_CANDLES = 51
VALID_TIMEFRAMES = ['1H', '4H', '6Hutc', '2m', '5m', '15m', '30m']
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
  order_amount: 40
  regime_family: "trending"
  dir_mode: "long_only"
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
- [ ] `regime_family` + `dir_mode` definidos
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
├─ Validar estrategias
├─ Validar símbolos por estrategia
├─ Inicializar PositionSizer
├─ Conectar a API
├─ Inicializar dashboard
├─ Cargar estado previo
└─ Calcular próximas velas

RUNNING
└─ Main loop infinito

SHUTDOWN
├─ Guardar estado
└─ Exit graceful
```

### 13.2 Main Loop
```python
while True:
    now = datetime.now(tz=UTC)
    
    closed_timeframes = check_closed_candles()
    
    if closed_timeframes:
        sync_broker()
        update_regime_for_timeframes(closed_timeframes)
        process_strategies(closed_timeframes)
    else:
        check_tp_sl_periodic()
    
    time.sleep(0.05)
```

---

## 14. Position Sizing Adaptativo

### 14.1 Flujo Completo
```
VELA CIERRA
    ↓
1. SYNC BROKER
    ↓
2. UPDATE REGIME + DIRECTION
├─ Fetch BTCUSDT OHLCV
├─ Calcular métricas
├─ Clasificar régimen + dirección
└─ Cachear
    ↓
3. PARA CADA ESTRATEGIA:
├─ Obtener market_regime + market_direction del cache
├─ PositionSizer.calculate_adjusted_amount()
├─ Si blocked (mult=0) → Skip
└─ Buscar señales con adjusted_amount
```

### 14.2 Logs de Régimen

**Formato de logs:**
```
[REGIME] Updating regime & direction for: ['1H']
[REGIME] 1H: REGIME=TRENDING, DIRECTION=UPTREND (BTC=$94356.80, MA50=$92145.23, hurst=0.67, er=0.58)
[SIZING] 06_reversal_long_1H: Market=[trending, uptrend] | Base=$40 × regime(1.8) × dir(1.5) = $108
[SIZING] Skip 07_reversal_short_1H: regime=trending(1.8x), dir=uptrend(0.0x), final=0x → BLOCKED
```

---

## 15. Troubleshooting

### 15.1 PositionSizer No Inicializado

**Síntoma:** `AttributeError: 'NoneType' object has no attribute 'calculate_adjusted_amount'`

**Solución:** Verificar que `self.position_sizer = PositionSizer(self.logger)` se ejecuta en `_load_and_validate_strategies()`

### 15.2 Dirección No Muestra BTC Price

**Síntoma:** Logs muestran "BTC=N/A, MA50=N/A"

**Diagnóstico:**
```bash
grep "BTC=" persistence/bot_files_XX/BOT_orchestator_XX.log
```

**Solución:** Verificar que `get_current_direction()` retorna tupla `(direction, price, ma50)`

### 15.3 Estrategia Bloqueada

**Síntoma:** Logs "Skip ... final=0x → BLOCKED"

**Diagnóstico:** Verificar que combinación estrategia/mercado tiene multiplier=0

**Solución:** Esto es correcto si quieres bloquear. Para cambiar, editar MATRIX en settings.py

### 15.4 Símbolos No Cargados

**Síntoma:** `FileNotFoundError: Symbol file not found`

**Solución:** Crear fichero `symbols_live/symbols_live_{strategy_id}_{timeframe}.xlsx`

### 15.5 Validación Falla

**Síntoma:** Bot no arranca, error de validación

**Solución:**
1. Verificar `regime_family` es válido (trending/ranging/volatile/general)
2. Verificar `dir_mode` es válido (long_only/short_only/general)
3. Verificar coherencia direction/dir_mode
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
    'order_amount': 40.0,
    'tp_pct': 4.0,
    'sl_pct': 10.0,
    'sell_after_ncandles': 50,
    'symbols': 'multi',
    'active': True,
    'regime_family': 'trending',
    'dir_mode': 'long_only'
}
```

### 16.2 PositionSizer Metadata
```python
{
    'base_amount': 40.0,
    'market_regime': 'trending',
    'market_direction': 'uptrend',
    'regime_multiplier': 1.8,
    'regime_source': 'MATRIX[trending][trending]',
    'direction_multiplier': 1.5,
    'direction_source': 'MATRIX[long_only][uptrend]',
    'final_multiplier': 2.7,
    'adjusted_amount': 108.0,
    'blocked': False
}
```

### 16.3 Caches
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

---

## 17. Comandos y Endpoints

### 17.1 Comandos CLI
```bash
# Iniciar bot
python3 main.py --account 00

# Con estrategias específicas
python3 main.py --account 00 --set-active "06,07,10,11"

# Ver logs
tail -f persistence/bot_files_00/BOT_orchestator_00.log

# Filtrar logs de sizing
grep SIZING persistence/bot_files_00/BOT_orchestator_00.log
```

### 17.2 API Endpoints
```
GET http://localhost:5000/api/regime/current?timeframe=1H
GET http://localhost:5000/api/regime/matrix
GET http://localhost:5000/api/strategies/regime
GET http://localhost:5000/api/positions
GET http://localhost:5000/api/status
```

### 17.3 Testing Rápido
```python
# Test position sizer
from market_regime import PositionSizer
import logging

logger = logging.getLogger('test')
sizer = PositionSizer(logger)

amount, meta = sizer.calculate_adjusted_amount(
    base_amount=40.0,
    strategy_family='trending',
    dir_mode='long_only',
    market_regime='trending',
    market_direction='uptrend'
)

print(f"Adjusted: ${amount:.2f}")
print(f"Final mult: {meta['final_multiplier']:.1f}x")
```

---

# 🎉 FIN DE DOCUMENTACIÓN

**BOT_trading v2.4 - Sistema de Trading Automatizado**

---

**Última actualización:** 2026-01-19  
**Autor:** Trading Bot Team  
**Nueva Feature:** Arquitectura modular de position sizing

---

## 📝 RESUMEN EJECUTIVO

### Cambios Principales v2.4

1. **Módulo PositionSizer:** Lógica de sizing separada y testeable
2. **Logs mejorados:** Incluyen BTC price y MA50 en dirección
3. **Arquitectura limpia:** Orchestrator delegado, responsabilidades claras
4. **Backward compatible:** Estrategias antiguas siguen funcionando

### Quick Start
```bash
# 1. Actualizar strategies.yaml
regime_family: "trending"
dir_mode: "long_only"

# 2. Reiniciar bot
python3 main.py --account 00

# 3. Verificar logs
grep SIZING persistence/bot_files_00/BOT_orchestator_00.log
```

---

**¿Preguntas? Consulta [Troubleshooting](#15-troubleshooting) o [Position Sizing Adaptativo](#14-position-sizing-adaptativo).**
</artifact>
