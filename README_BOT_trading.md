# 🤖 BOT TRADING - DOCUMENTACIÓN TÉCNICA v2.3

**Sistema de Trading Automatizado con Custom Regime Multipliers por Estrategia**

---

**Versión:** 2.3  
**Fecha:** 2026-01-15  
**Python:** 3.12  
**Framework:** Flask + ccxt + Bitget API  
**Nuevo:** Custom Regime Multipliers - Matriz de régimen personalizada por estrategia

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

### 1.2 Novedades v2.3

**REGIME_FAMILY_MATRIX:** Sistema de multiplicadores personalizados por estrategia

Cada estrategia puede declarar su "familia de régimen" (trending/ranging/volatile) y el sistema ajustará automáticamente el tamaño de posición usando una matriz bidimensional:

```
MATRIZ[estrategia.regime_family][mercado.current_regime] → multiplier

Ejemplo:
- Estrategia: regime_family='trending' 
- Mercado: current_regime='trending'
- Lookup: MATRIX['trending']['trending'] = 1.8x
→ Posición abierta con 1.8x el monto base
```

**Ventajas:**
- Estrategias trending operan más agresivas en mercados trending
- Estrategias ranging operan más agresivas en mercados laterales
- Estrategias volatile aprovechan la volatilidad
- Bloqueo automático cuando estrategia/mercado no alinean

### 1.3 Flujo Simplificado

```
VELA CIERRA → DETECTAR RÉGIMEN MERCADO
    ↓
PARA CADA ESTRATEGIA:
├─ Leer strategy.regime_family (del YAML)
├─ Leer mercado.current_regime (del clasificador)
├─ Lookup: MATRIX[family][regime] → multiplier
├─ Calcular: adjusted_amount = base * multiplier
└─ Si multiplier != 0 → Buscar señales con adjusted_amount
```

---

## 2. Arquitectura

### 2.1 Estructura de Directorios

```
bitget/
├── BOT_trading/                    # 🤖 Producción
│   ├── config/
│   │   └── settings.py             # REGIME_FAMILY_MATRIX ⭐
│   ├── core/
│   │   └── orchestrator.py         # Lookup matriz + cache
│   ├── strategies/
│   │   ├── strategies.yaml         # regime_family: 'trending' ⭐
│   │   └── strategy_registry.py    # Registro elif
│   ├── market_regime/
│   │   └── regime_classifier.py    # Detecta mercado
│   ├── execution/
│   │   └── bitget_client.py
│   ├── api/
│   │   ├── backend.py              # /api/regime/matrix ⭐
│   │   └── templates/
│   │       └── dashboard.html      # Visualización matriz ⭐
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
├─ Calcular ATR, ER, PE
├─ Clasificar: volatile/ranging/trending
└─ Cachear regime por timeframe
    ↓
Orchestrator (para cada estrategia)
├─ Leer strategy.regime_family
├─ Lookup: MATRIX[family][current_regime]
├─ Calcular: adjusted_order_amount
└─ Pasar a Strategy Processor
    ↓
Strategy Processor
├─ Detectar señales
└─ Abrir posición con adjusted_order_amount
```

### 2.3 Separación de Responsabilidades

| Módulo | Responsabilidad |
|--------|-----------------|
| **config/** | REGIME_FAMILY_MATRIX, configuración |
| **core/** | Lookup matriz, cache, orquestación |
| **strategies/** | Definición regime_family en YAML |
| **market_regime/** | Clasificación de mercado |
| **execution/** | API Bitget |
| **api/** | Dashboard + endpoints matriz |
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
| scipy | Permutation Entropy |
| pandas_ta | ATR, ER |

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

El sistema calcula 3 métricas en los últimos 50 períodos de BTCUSDT:

**1. ATR Normalizado (Average True Range)**
```python
ATR_normalized = (ATR_50 / close[-1]) * 100
```
- **Rango:** 0-15%
- **Uso:** Detectar volatilidad extrema

**2. ER (Efficiency Ratio)**
```python
ER = abs(close[-1] - close[-50]) / sum(abs(price_changes))
```
- **Rango:** 0.0 - 1.0
- **Uso:** Detectar trending vs ranging
- 0 = completamente lateral
- 1 = tendencia perfecta

**3. PE (Permutation Entropy)**
```python
PE = entropy(permutations(log_returns, order=3))
```
- **Rango:** 0.0 - 1.0
- **Uso:** Detectar aleatoriedad/estructura
- 0 = predecible
- 1 = completamente aleatorio

### 4.2 Reglas de Clasificación

```python
# config/settings.py
REGIME_FAMILIES = {
    'volatile': {
        'atr_min': 3.5,  # ATR > 3.5%
        'pe_min': 0.70   # PE > 0.70
    },
    'ranging': {
        'er_max': 0.45   # ER < 0.45
    },
    'trending': {
        'er_min': 0.45   # ER >= 0.45
    },
    'default': {}
}
```

**Orden de evaluación:**
1. VOLATILE: if ATR > 3.5 AND PE > 0.70
2. RANGING: elif ER < 0.45
3. TRENDING: elif ER >= 0.45
4. DEFAULT: else (fallback)

### 4.3 Símbolo de Referencia

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

La **REGIME_FAMILY_MATRIX** es una matriz bidimensional que mapea:
- **Fila:** Familia de la estrategia (trending/ranging/volatile)
- **Columna:** Régimen actual del mercado (trending/ranging/volatile)
- **Valor:** Multiplicador de position sizing

### 5.2 Definición de la Matriz

```python
# config/settings.py
REGIME_FAMILY_MATRIX = {
    'trending': {
        'trending': 1.8,   # Trending + Trending = MUY agresivo
        'ranging': 1.0,    # Trending + Ranging = Normal
        'volatile': 0.0    # Trending + Volatile = BLOQUEAR
    },
    'ranging': {
        'trending': 1.0,   # Ranging + Trending = Normal
        'ranging': 1.8,    # Ranging + Ranging = MUY agresivo
        'volatile': 0.0    # Ranging + Volatile = BLOQUEAR
    },
    'volatile': {
        'trending': 0.5,   # Volatile + Trending = Reducir
        'ranging': 0.5,    # Volatile + Ranging = Reducir
        'volatile': 1.5    # Volatile + Volatile = Aprovechar
    }
}
```

### 5.3 Lógica de Aplicación

```python
# En orchestrator.py (_search_signals)

# 1. Detectar régimen actual del mercado
market_regime = get_current_regime(timeframe)  # → 'trending'

# 2. Leer familia de la estrategia
strategy_family = strat.get('regime_family')   # → 'ranging' (del YAML)

# 3. Lookup multiplier
if strategy_family:
    # Custom: usar matriz
    multiplier = REGIME_FAMILY_MATRIX[strategy_family][market_regime]
    # → MATRIX['ranging']['trending'] = 1.0x
else:
    # Global fallback: usar sizing genérico
    multiplier = REGIME_FAMILY_SIZING[market_regime]
    # → SIZING['trending'] = 1.5x

# 4. Calcular adjusted amount
adjusted_order_amount = strat['order_amount'] * multiplier

# 5. Si multiplier == 0 → Skip estrategia (bloquear)
if multiplier == 0:
    logger.info(f"Skipping {strat_id}: multiplier=0 (blocked)")
    continue
```

### 5.4 Ejemplos Prácticos

**Ejemplo 1: Estrategia Trending en Mercado Trending**
```
Estrategia: 06_reversal_long_1H
- regime_family: 'trending'
- order_amount: 40 USDT

Mercado: TRENDING (ER=0.67)

Lookup: MATRIX['trending']['trending'] = 1.8x
Adjusted: 40 * 1.8 = 72 USDT

→ Posición abierta con 72 USDT (80% más)
```

**Ejemplo 2: Estrategia Trending en Mercado Volatile**
```
Estrategia: 06_reversal_long_1H
- regime_family: 'trending'
- order_amount: 40 USDT

Mercado: VOLATILE (ATR=4.2%, PE=0.78)

Lookup: MATRIX['trending']['volatile'] = 0.0x
Adjusted: 40 * 0.0 = 0 USDT

→ Estrategia bloqueada (no se buscan señales)
```

**Ejemplo 3: Estrategia Sin Familia (Fallback Global)**
```
Estrategia: 12_legacy_strategy_4H
- regime_family: null (no definido)
- order_amount: 50 USDT

Mercado: TRENDING

Lookup: REGIME_FAMILY_SIZING['trending'] = 1.5x
Adjusted: 50 * 1.5 = 75 USDT

→ Usa multiplicador global (backward compatible)
```

### 5.5 Fallback Global

Para estrategias que NO tienen `regime_family` definido en YAML:

```python
# config/settings.py
REGIME_FAMILY_SIZING = {
    'volatile': 0.5,   # Reducir a la mitad
    'ranging': 1.0,    # Sin ajuste
    'trending': 1.5,   # Aumentar 50%
    'default': 1.0     # Fallback
}
```

Esto permite **backward compatibility**: estrategias antiguas siguen funcionando con multiplicadores globales.

---

## 6. Configuración de Régimen

### 6.1 Archivo settings.py

**Ubicación:** `config/settings.py`

Todas las configuraciones de régimen se centralizan aquí:

```python
# ===== MARKET REGIME CONFIGURATION =====

# Símbolo de referencia
REGIME_REFERENCE_SYMBOL = 'BTCUSDT'

# Thresholds para clasificación
REGIME_FAMILIES = {
    'volatile': {'atr_min': 3.5, 'pe_min': 0.70},
    'ranging': {'er_max': 0.45},
    'trending': {'er_min': 0.45},
    'default': {}
}

# Multiplicadores globales (fallback)
REGIME_FAMILY_SIZING = {
    'volatile': 0.5,
    'ranging': 1.0,
    'trending': 1.5,
    'default': 1.0
}

# Matriz custom por estrategia ⭐
REGIME_FAMILY_MATRIX = {
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
```

### 6.2 Modificar Configuración

**Para ajustar thresholds:**
1. Editar `REGIME_FAMILIES` en settings.py
2. Reiniciar bot

**Para ajustar multiplicadores:**
1. Editar `REGIME_FAMILY_MATRIX` en settings.py
2. Reiniciar bot

**Ejemplos:**

```python
# Más restrictivo en volatile
'volatile': {'atr_min': 4.0, 'pe_min': 0.75}

# Más agresivo en trending
'trending': {
    'trending': 2.0,  # Antes 1.8x, ahora 2.0x
    ...
}

# Bloquear combinación específica
'trending': {
    'ranging': 0.0,  # Antes 1.0x, ahora BLOQUEAR
    ...
}
```

### 6.3 Validación

El sistema valida automáticamente en startup:
- Todas las familias en MATRIX existen
- Todos los regímenes en MATRIX existen
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
4. **Cache de régimen por timeframe**
5. **Lookup de multiplicador custom por estrategia**
6. Ajuste de position sizing
7. Sincronización con broker

### 7.2 Variables de Estado

```python
class BotOrchestrator:
    def __init__(self, ...):
        self.OPEN_POSITIONS = {}      # {strategy_id: [positions]}
        self.STRATEGY_CANDLES = {}    # {strategy_id: counter}
        self.strategies = []          # Cargadas desde YAML
        self.regime_cache = {}        # {timeframe: multiplier}
```

**regime_cache:** Dict en memoria (NO persiste)
- Clave: timeframe ('4H', '1H', etc.)
- Valor: multiplier actual
- Se recalcula cada vela cerrada
- Compartido entre estrategias del mismo timeframe

### 7.3 Método `_update_regime_for_timeframes()`

Calcula y cachea régimen tras cerrar vela:

```python
def _update_regime_for_timeframes(self, timeframes):
    """
    Calcula regime para lista de timeframes y cachea
    """
    for tf in timeframes:
        try:
            # Detectar régimen del mercado
            market_regime, metrics = get_current_regime(tf)
            
            # Cachear para este timeframe
            self.regime_cache[tf] = market_regime
            
            self.logger.info(
                f"[REGIME] {tf}: {market_regime.upper()} "
                f"(ATR={metrics['atr']:.2f}%, "
                f"ER={metrics['er']:.2f}, "
                f"PE={metrics['pe']:.2f})"
            )
        except Exception as e:
            # Fallback: sin clasificación
            self.regime_cache[tf] = 'default'
            self.logger.error(f"[REGIME] Error for {tf}: {e}")
```

### 7.4 Método `_search_signals()`

Lookup de matriz y ajuste de sizing:

```python
def _search_signals(self, strategies_to_process):
    """
    Busca señales con custom multipliers
    """
    for strat in strategies_to_process:
        strat_id = strat['id']
        
        # Skip si ya tiene posiciones
        if len(self.open_positions.get(strat_id, [])) > 0:
            continue
        
        # Obtener régimen del mercado (del cache)
        timeframe = strat['timeframe']
        market_regime = self.regime_cache.get(timeframe, 'default')
        
        # Obtener familia de la estrategia (del YAML)
        strategy_family = strat.get('regime_family')
        
        # Lookup multiplier
        if strategy_family:
            # Custom: usar matriz
            multiplier = REGIME_FAMILY_MATRIX[strategy_family][market_regime]
            source = f"MATRIX[{strategy_family}][{market_regime}]"
        else:
            # Fallback: usar sizing global
            multiplier = REGIME_FAMILY_SIZING[market_regime]
            source = f"SIZING[{market_regime}]"
        
        # Si multiplier == 0 → Bloquear estrategia
        if multiplier == 0:
            self.logger.info(
                f"[REGIME] Skipping {strat_id}: "
                f"multiplier=0 ({source})"
            )
            continue
        
        # Calcular adjusted amount
        base_amount = strat['order_amount']
        adjusted_amount = base_amount * multiplier
        
        self.logger.debug(
            f"[REGIME] {strat_id}: Base=${base_amount:.2f}, "
            f"Multiplier={multiplier}x ({source}), "
            f"Adjusted=${adjusted_amount:.2f}"
        )
        
        # Buscar señales con adjusted amount
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
16:00:02 - Calcular regime → 'trending'
16:00:03 - Cachear en regime_cache['1H']
16:00:04 - Para cada estrategia 1H:
           ├─ Lookup multiplier
           ├─ Calcular adjusted_amount
           └─ Buscar señales
```

---

## 8. Sistema de Estrategias

### 8.1 Definición en YAML

**strategies/strategies.yaml**

Cada estrategia ahora puede declarar `regime_family`:

```yaml
strategies:
  - id: "06_reversal_long_1H"
    function_name: "add_signals_reversal_long"
    timeframe: "1H"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "trending"  # ⭐ NUEVO: declara familia
    
  - id: "10_parity_reversal_short_1H"
    function_name: "add_signals_parity_reversal_short"
    timeframe: "1H"
    order_amount: 45
    tp_pct: 3.5
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "ranging"   # ⭐ NUEVO: otra familia
    
  - id: "12_legacy_strategy_4H"
    function_name: "add_signals_legacy"
    timeframe: "4H"
    order_amount: 50
    # regime_family: null       # ⭐ Sin definir → usa fallback global
```

### 8.2 Registro en strategy_registry.py

**strategies/strategy_registry.py**

Sin cambios. Sigue siendo un elif explícito:

```python
# strategies/strategy_registry.py
from signals.add_signals_reversal import (
    add_signals_reversal_long,
    add_signals_reversal_short
)
from signals.add_signals_parity import (
    add_signals_parity_reversal_long,
    add_signals_parity_reversal_short
)

IMPLEMENTED_STRATEGIES = {
    'add_signals_reversal_long': add_signals_reversal_long,
    'add_signals_reversal_short': add_signals_reversal_short,
    'add_signals_parity_reversal_long': add_signals_parity_reversal_long,
    'add_signals_parity_reversal_short': add_signals_parity_reversal_short,
    # ...
}

def get_signal_function(function_name):
    """
    Retorna función de señal o None
    """
    if function_name in IMPLEMENTED_STRATEGIES:
        return IMPLEMENTED_STRATEGIES[function_name]
    else:
        return None
```

### 8.3 Validación

El sistema valida en startup:
- `regime_family` existe en REGIME_FAMILY_MATRIX (si está definido)
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
    """
    Retorna REGIME_FAMILY_MATRIX completa
    """
    return jsonify({
        'matrix': REGIME_FAMILY_MATRIX,
        'sizing': REGIME_FAMILY_SIZING,
        'families': REGIME_FAMILIES
    })

@app.route('/api/regime/current', methods=['GET'])
def get_current_regime():
    """
    Retorna régimen actual por timeframe
    Query params: timeframe (ej: '1H')
    """
    tf = request.args.get('timeframe', '1H')
    
    regime, metrics = get_regime_info(tf)
    
    return jsonify({
        'timeframe': tf,
        'regime': regime,
        'metrics': metrics,
        'matrix': REGIME_FAMILY_MATRIX,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/strategies/regime', methods=['GET'])
def get_strategies_regime():
    """
    Retorna estrategias con su regime_family
    """
    strategies_info = []
    for strat in strategies:
        strategies_info.append({
            'id': strat['id'],
            'regime_family': strat.get('regime_family'),
            'timeframe': strat['timeframe'],
            'order_amount': strat['order_amount']
        })
    
    return jsonify({
        'strategies': strategies_info,
        'total': len(strategies_info)
    })
```

### 9.2 Visualización en Dashboard

**api/templates/dashboard.html**

Nuevo tab "Regime Matrix":

```html
<!-- Tab Regime Matrix -->
<div class="tab-content" id="tab-regime-matrix">
    <h3>Regime Family Matrix</h3>
    
    <!-- Selector timeframe -->
    <select id="regime-timeframe-selector">
        <option value="4H">4H</option>
        <option value="1H" selected>1H</option>
        <option value="6Hutc">6Hutc</option>
    </select>
    
    <!-- Régimen actual del mercado -->
    <div class="regime-current">
        <h4>Current Market Regime: <span id="current-regime"></span></h4>
        <div class="metrics">
            <div>ATR: <span id="metric-atr"></span>%</div>
            <div>ER: <span id="metric-er"></span></div>
            <div>PE: <span id="metric-pe"></span></div>
        </div>
    </div>
    
    <!-- Tabla matriz -->
    <table class="matrix-table">
        <thead>
            <tr>
                <th>Strategy Family</th>
                <th>TRENDING Market</th>
                <th>RANGING Market</th>
                <th>VOLATILE Market</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Trending Strategies</strong></td>
                <td class="mult-high">1.8x</td>
                <td class="mult-normal">1.0x</td>
                <td class="mult-block">0.0x</td>
            </tr>
            <tr>
                <td><strong>Ranging Strategies</strong></td>
                <td class="mult-normal">1.0x</td>
                <td class="mult-high">1.8x</td>
                <td class="mult-block">0.0x</td>
            </tr>
            <tr>
                <td><strong>Volatile Strategies</strong></td>
                <td class="mult-low">0.5x</td>
                <td class="mult-low">0.5x</td>
                <td class="mult-high">1.5x</td>
            </tr>
        </tbody>
    </table>
    
    <!-- Lista estrategias con familia -->
    <div class="strategies-by-family">
        <h4>Strategies by Family</h4>
        <div id="strategies-list"></div>
    </div>
</div>

<script>
// Actualizar régimen cada 5s
setInterval(updateRegimeMatrix, 5000);

function updateRegimeMatrix() {
    const tf = document.getElementById('regime-timeframe-selector').value;
    
    fetch(`/api/regime/current?timeframe=${tf}`)
        .then(r => r.json())
        .then(data => {
            document.getElementById('current-regime').textContent = 
                data.regime.toUpperCase();
            document.getElementById('metric-atr').textContent = 
                data.metrics.atr.toFixed(2);
            document.getElementById('metric-er').textContent = 
                data.metrics.er.toFixed(2);
            document.getElementById('metric-pe').textContent = 
                data.metrics.pe.toFixed(2);
            
            // Highlight columna activa
            highlightActiveRegime(data.regime);
        });
    
    // Cargar estrategias
    fetch('/api/strategies/regime')
        .then(r => r.json())
        .then(data => {
            renderStrategiesByFamily(data.strategies);
        });
}

function highlightActiveRegime(regime) {
    // Quitar highlight anterior
    document.querySelectorAll('.matrix-table td').forEach(td => {
        td.classList.remove('active');
    });
    
    // Añadir highlight a columna activa
    const colIndex = {
        'trending': 2,
        'ranging': 3,
        'volatile': 4
    }[regime];
    
    document.querySelectorAll(`.matrix-table tr td:nth-child(${colIndex})`)
        .forEach(td => td.classList.add('active'));
}

function renderStrategiesByFamily(strategies) {
    const grouped = {
        'trending': [],
        'ranging': [],
        'volatile': [],
        'global': []
    };
    
    strategies.forEach(s => {
        const family = s.regime_family || 'global';
        grouped[family].push(s);
    });
    
    let html = '';
    for (const [family, strats] of Object.entries(grouped)) {
        if (strats.length === 0) continue;
        
        html += `
            <div class="family-group">
                <h5>${family.toUpperCase()} (${strats.length})</h5>
                <ul>
                    ${strats.map(s => `
                        <li>
                            ${s.id} 
                            <span class="timeframe">${s.timeframe}</span>
                            <span class="amount">$${s.order_amount}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    document.getElementById('strategies-list').innerHTML = html;
}
</script>

<style>
.matrix-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

.matrix-table th,
.matrix-table td {
    padding: 12px;
    text-align: center;
    border: 1px solid #ddd;
}

.matrix-table th {
    background: #f5f5f5;
    font-weight: bold;
}

.mult-high {
    background: #d4edda;
    color: #155724;
    font-weight: bold;
}

.mult-normal {
    background: #fff3cd;
    color: #856404;
}

.mult-low {
    background: #f8d7da;
    color: #721c24;
}

.mult-block {
    background: #343a40;
    color: #fff;
    font-weight: bold;
}

.matrix-table td.active {
    box-shadow: 0 0 0 3px #007bff;
}

.family-group {
    margin: 15px 0;
    padding: 10px;
    border-left: 3px solid #007bff;
    background: #f8f9fa;
}

.family-group h5 {
    margin: 0 0 10px 0;
}

.family-group ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.family-group li {
    padding: 5px 0;
}

.timeframe {
    display: inline-block;
    padding: 2px 6px;
    background: #6c757d;
    color: white;
    border-radius: 3px;
    font-size: 0.85em;
    margin-left: 8px;
}

.amount {
    display: inline-block;
    margin-left: 8px;
    color: #28a745;
    font-weight: bold;
}
</style>
```

---

# PARTE 4: CONFIGURACIÓN

## 10. Settings.py

**Ubicación:** `config/settings.py`

### 10.1 Configuración de Cuentas

```python
ACCOUNTS = {
    '00': {
        'initial_capital': 3671,
        'port': 5000,
        'description': 'Main Account'
    },
    'E1': {
        'initial_capital': 1761,
        'port': 5001,
        'description': 'Elite Account'
    },
    '01': {
        'initial_capital': 117,
        'port': 5099,
        'description': 'Testing Account'
    }
}

ACCOUNT_STRATEGIES = {
    '00': ['01', '02', '03', '04', '06', '07', '08', '09', '10', '11', '13', '15', '16', '17'],
    'E1': ['01', '02', '03', '04', '06', '07', '08', '09', '10', '11', '13', '15', '16'],
    '01': ['06', '07']
}
```

### 10.2 Configuración de Régimen

```python
# ===== MARKET REGIME CONFIGURATION =====

# Símbolo de referencia para cálculo
REGIME_REFERENCE_SYMBOL = 'BTCUSDT'

# Thresholds para clasificación de mercado
REGIME_FAMILIES = {
    'volatile': {
        'atr_min': 3.5,  # ATR > 3.5% (alta volatilidad)
        'pe_min': 0.70   # PE > 0.70 (alta aleatoriedad)
    },
    'ranging': {
        'er_max': 0.45   # ER < 0.45 (baja eficiencia direccional)
    },
    'trending': {
        'er_min': 0.45   # ER >= 0.45 (alta eficiencia direccional)
    },
    'default': {}
}

# Multiplicadores globales (fallback para estrategias sin family)
REGIME_FAMILY_SIZING = {
    'volatile': 0.5,   # Reducir 50%
    'ranging': 1.0,    # Sin ajuste
    'trending': 1.5,   # Aumentar 50%
    'default': 1.0
}

# ⭐ MATRIZ CUSTOM POR ESTRATEGIA
REGIME_FAMILY_MATRIX = {
    'trending': {
        'trending': 1.8,   # Trending strategy + Trending market = Muy agresivo
        'ranging': 1.0,    # Trending strategy + Ranging market = Normal
        'volatile': 0.0    # Trending strategy + Volatile market = BLOQUEAR
    },
    'ranging': {
        'trending': 1.0,   # Ranging strategy + Trending market = Normal
        'ranging': 1.8,    # Ranging strategy + Ranging market = Muy agresivo
        'volatile': 0.0    # Ranging strategy + Volatile market = BLOQUEAR
    },
    'volatile': {
        'trending': 0.5,   # Volatile strategy + Trending market = Reducir
        'ranging': 0.5,    # Volatile strategy + Ranging market = Reducir
        'volatile': 1.5    # Volatile strategy + Volatile market = Aprovechar
    }
}
```

### 10.3 Validación Settings

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

**Ubicación:** `strategies/strategies.yaml`

### 11.1 Estructura Completa

```yaml
strategies:
  # ===== TRENDING STRATEGIES =====
  - id: "06_reversal_long_1H"
    function_name: "add_signals_reversal_long"
    timeframe: "1H"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "trending"  # ⭐ Declara familia
    description: "Reversal alcista 1H - opera mejor en trending"
    
  - id: "07_reversal_short_1H"
    function_name: "add_signals_reversal_short"
    timeframe: "1H"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "trending"
    description: "Reversal bajista 1H - opera mejor en trending"
    
  # ===== RANGING STRATEGIES =====
  - id: "10_parity_reversal_short_1H"
    function_name: "add_signals_parity_reversal_short"
    timeframe: "1H"
    order_amount: 45
    tp_pct: 3.5
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "ranging"  # ⭐ Familia ranging
    description: "Parity reversal short - mejor en ranging"
    
  - id: "11_parity_reversal_long_1H"
    function_name: "add_signals_parity_reversal_long"
    timeframe: "1H"
    order_amount: 45
    tp_pct: 3.5
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "ranging"
    description: "Parity reversal long - mejor en ranging"
    
  # ===== VOLATILE STRATEGIES =====
  # (Actualmente no implementadas)
    
  # ===== LEGACY STRATEGIES (sin family - usan fallback) =====
  - id: "01_double_top_long_4H"
    function_name: "add_signals_double_top_long"
    timeframe: "4H"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    # regime_family: null  # ⭐ Sin definir → usa REGIME_FAMILY_SIZING
    description: "Double top largo 4H - usa multiplicadores globales"
```

### 11.2 Campos Disponibles

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `id` | String | Sí | Identificador único |
| `function_name` | String | Sí | Nombre en strategy_registry |
| `timeframe` | String | Sí | 4H, 1H, 6Hutc, etc. |
| `order_amount` | Float | Sí | Monto base en USDT |
| `tp_pct` | Float | Sí | Take Profit % |
| `sl_pct` | Float | Sí | Stop Loss % |
| `candles_timeout` | Int | Sí | Velas hasta timeout |
| `symbols` | String | Sí | "multi" o lista |
| `regime_family` | String | No | trending/ranging/volatile ⭐ |
| `description` | String | No | Descripción humana |

---

## 12. Alta de Estrategias

### 12.1 Proceso Completo

**Paso 1: Crear función de señal**

```python
# bitget/signals/add_signals_mi_estrategia.py
import numpy as np
from .indicators import calculate_rsi, calculate_ema

def add_signals_mi_estrategia_long(data):
    """
    Señal long para mi estrategia
    
    Args:
        data: Dict con arrays numpy {open, high, low, close, volume}
    
    Returns:
        signals: Array numpy (1=long, -1=short, 0=nada)
    """
    close = data['close']
    high = data['high']
    low = data['low']
    
    # Calcular indicadores
    rsi = calculate_rsi(close, period=14)
    ema_fast = calculate_ema(close, period=12)
    ema_slow = calculate_ema(close, period=26)
    
    # Inicializar señales
    signals = np.zeros(len(close))
    
    # Lógica de señal
    for i in range(50, len(close)):
        # Condición long
        if (rsi[i] > 50 and 
            ema_fast[i] > ema_slow[i] and
            close[i] > high[i-1]):
            signals[i] = 1  # LONG
    
    return signals

def add_signals_mi_estrategia_short(data):
    """
    Señal short para mi estrategia
    """
    close = data['close']
    low = data['low']
    
    rsi = calculate_rsi(close, period=14)
    ema_fast = calculate_ema(close, period=12)
    ema_slow = calculate_ema(close, period=26)
    
    signals = np.zeros(len(close))
    
    for i in range(50, len(close)):
        # Condición short
        if (rsi[i] < 50 and 
            ema_fast[i] < ema_slow[i] and
            close[i] < low[i-1]):
            signals[i] = -1  # SHORT
    
    return signals
```

**Paso 2: Registrar en strategy_registry.py**

```python
# strategies/strategy_registry.py
from signals.add_signals_mi_estrategia import (
    add_signals_mi_estrategia_long,
    add_signals_mi_estrategia_short
)

IMPLEMENTED_STRATEGIES = {
    # ... estrategias existentes ...
    
    # Nueva estrategia
    'add_signals_mi_estrategia_long': add_signals_mi_estrategia_long,
    'add_signals_mi_estrategia_short': add_signals_mi_estrategia_short,
}
```

**Paso 3: Añadir a strategies.yaml**

```yaml
# strategies/strategies.yaml
strategies:
  # ... estrategias existentes ...
  
  # Nueva estrategia
  - id: "18_mi_estrategia_long_1H"
    function_name: "add_signals_mi_estrategia_long"
    timeframe: "1H"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "trending"  # ⭐ Declarar familia
    description: "Mi estrategia - mejor en trending"
    
  - id: "19_mi_estrategia_short_1H"
    function_name: "add_signals_mi_estrategia_short"
    timeframe: "1H"
    order_amount: 40
    tp_pct: 4.0
    sl_pct: 10.0
    candles_timeout: 50
    symbols: "multi"
    regime_family: "trending"
    description: "Mi estrategia short - mejor en trending"
```

**Paso 4: Validar**

```bash
# Reiniciar bot
python3 main.py --account 01

# Verificar en logs:
# ✓ Strategy loaded: 18_mi_estrategia_long_1H
# ✓ Regime family: trending
# ✓ Function: add_signals_mi_estrategia_long FOUND
```

### 12.2 Checklist

- [ ] Función creada en `signals/`
- [ ] Retorna array numpy con señales (1/-1/0)
- [ ] Registrada en `strategy_registry.py`
- [ ] Añadida a `strategies.yaml`
- [ ] `regime_family` definido (opcional)
- [ ] Validación pasa en startup
- [ ] Probada en cuenta 01 (testing)

---

# PARTE 5: FLUJOS Y OPERACIÓN

## 13. Ciclo de Vida

### 13.1 Inicialización

```
STARTUP
├─ Parse args: --account XX --set-active "06,07,10"
├─ Cargar settings.py
├─ Cargar strategies.yaml
├─ Filtrar estrategias activas
├─ Validar configuración
│  ├─ regime_family existe en MATRIX
│  ├─ function_name existe en registry
│  └─ Todos los campos válidos
├─ Cargar símbolos por estrategia
├─ Conectar a Bitget API
├─ Inicializar dashboard (Flask)
├─ Cargar estado previo (JSON)
├─ Sync con broker
└─ Calcular próximas velas

RUNNING
└─ Main loop infinito

SHUTDOWN (Ctrl+C)
├─ Guardar estado
├─ Cerrar posiciones (opcional)
└─ Exit graceful
```

### 13.2 Main Loop

```python
while True:
    now = datetime.now(tz=UTC)
    
    # Check velas cerradas
    closed_timeframes = []
    for tf, next_time in next_candle_times.items():
        if now >= next_time:
            closed_timeframes.append(tf)
    
    if closed_timeframes:
        # Procesar timeframes cerrados
        for tf in closed_timeframes:
            process_timeframe(tf)
            # ├─ Sync broker
            # ├─ Update regime cache
            # ├─ Increment candles
            # ├─ Check timeout
            # └─ Search signals (con custom multipliers)
            
            # Recalcular próxima vela
            next_candle_times[tf] = calculate_next(tf, now)
    
    # Check TP/SL cada 10s
    if (now - last_check).total_seconds() >= 10:
        check_tp_sl_all_positions()
        last_check = now
    
    time.sleep(0.05)
```

---

## 14. Position Sizing Adaptativo

### 14.1 Flujo Completo

```
VELA CIERRA (ej: 1H)
    ↓
1. SYNC BROKER
└─ Reconciliar posiciones
    ↓
2. DETECTAR RÉGIMEN MERCADO
├─ Fetch OHLCV de BTCUSDT
├─ Calcular ATR, ER, PE
├─ Clasificar: 'trending' (ER=0.67)
└─ Cachear: regime_cache['1H'] = 'trending'
    ↓
3. PARA CADA ESTRATEGIA 1H:
    ↓
    ├─ Estrategia: 06_reversal_long_1H
    │  ├─ regime_family: 'trending' (del YAML)
    │  └─ order_amount: 40 USDT
    │
    ├─ Lookup Multiplier:
    │  └─ MATRIX['trending']['trending'] = 1.8x
    │
    ├─ Calcular Adjusted:
    │  └─ adjusted = 40 * 1.8 = 72 USDT
    │
    └─ Si multiplier != 0:
        └─ Buscar señales con adjusted=72 USDT
```

### 14.2 Ejemplos Detallados

**Caso 1: Trending + Trending = Muy Agresivo**
```
Estrategia: 06_reversal_long_1H
- regime_family: 'trending'
- order_amount: 40 USDT
- timeframe: 1H

Mercado: TRENDING
- ATR: 2.1%
- ER: 0.67
- PE: 0.42

Lookup: MATRIX['trending']['trending'] = 1.8x
Adjusted: 40 * 1.8 = 72 USDT

→ Señal detectada
→ Posición abierta: 72 USDT
→ TP: +4% (2.88 USDT profit potencial)
```

**Caso 2: Trending + Volatile = Bloqueado**
```
Estrategia: 06_reversal_long_1H
- regime_family: 'trending'
- order_amount: 40 USDT

Mercado: VOLATILE
- ATR: 4.5%
- ER: 0.28
- PE: 0.81

Lookup: MATRIX['trending']['volatile'] = 0.0x

→ Estrategia bloqueada (skip)
→ No se buscan señales
→ Log: "[REGIME] Skipping 06_reversal_long_1H: multiplier=0"
```

**Caso 3: Sin Familia = Fallback Global**
```
Estrategia: 01_double_top_long_4H
- regime_family: null
- order_amount: 40 USDT

Mercado: TRENDING

Lookup: REGIME_FAMILY_SIZING['trending'] = 1.5x
Adjusted: 40 * 1.5 = 60 USDT

→ Usa multiplicador global
→ Posición abierta: 60 USDT
```

### 14.3 Logs de Régimen

**Logs INFO:**
```
[REGIME] Updating regime for timeframes: ['1H']
[REGIME] 1H: TRENDING (ATR=2.34%, ER=0.67, PE=0.45)
[REGIME] 06_reversal_long_1H: MATRIX[trending][trending]=1.8x → Adjusted=$72.00
[REGIME] 10_parity_reversal_short_1H: MATRIX[ranging][trending]=1.0x → Adjusted=$45.00
[REGIME] Skipping 07_reversal_short_1H: multiplier=0 (blocked by regime)
```

**Logs DEBUG:**
```
[REGIME] Strategy: 06_reversal_long_1H
[REGIME]   Family: trending
[REGIME]   Market: trending
[REGIME]   Base: $40.00
[REGIME]   Multiplier: 1.8x (MATRIX[trending][trending])
[REGIME]   Adjusted: $72.00
```

---

## 15. Troubleshooting

### 15.1 Matriz No Funciona

**Síntoma:** Multipliers siempre 1.0x

**Diagnóstico:**
```bash
# Verificar MATRIX en settings
grep "REGIME_FAMILY_MATRIX" config/settings.py

# Verificar regime_family en YAML
grep "regime_family" strategies/strategies.yaml

# Ver logs
grep "MATRIX" persistence/bot_files_XX/BOT_orchestator_XX.log
```

**Solución:**
1. Verificar REGIME_FAMILY_MATRIX definido
2. Verificar estrategia tiene `regime_family` en YAML
3. Reiniciar bot

### 15.2 Estrategia Bloqueada

**Síntoma:** Logs "Skipping ... multiplier=0"

**Diagnóstico:**
```bash
# Ver configuración
python3 -c "
from config.settings import REGIME_FAMILY_MATRIX
print(REGIME_FAMILY_MATRIX['trending']['volatile'])
"
# → 0.0 (bloqueado intencionalmente)
```

**Solución:**
- Esto es CORRECTO si la estrategia/mercado no alinean
- Para cambiar: editar MATRIX en settings.py

### 15.3 Posiciones con Tamaño Incorrecto

**Síntoma:** Posición abierta con monto inesperado

**Diagnóstico:**
```bash
# Ver logs de cálculo
grep "adjusted_order_amount" persistence/bot_files_XX/BOT_orchestator_XX.log

# Ejemplo esperado:
# [REGIME] Base=$40.00, Multiplier=1.8x, Adjusted=$72.00
# [INFO] Opening position: BTCUSDT LONG | Amount: $72.00
```

**Verificar:**
1. regime_family correcto en YAML
2. Mercado clasificado correctamente
3. Multiplier en MATRIX correcto
4. adjusted_order_amount pasado a processor

### 15.4 Dashboard No Muestra Matriz

**Síntoma:** Tab "Regime Matrix" vacío

**Diagnóstico:**
```bash
# Probar endpoint
curl http://localhost:5000/api/regime/matrix

# Verificar JavaScript
# Abrir browser → F12 → Console → Buscar errores
```

**Solución:**
1. Verificar endpoint `/api/regime/matrix` en backend.py
2. Verificar tab HTML con id `tab-regime-matrix`
3. Hard refresh: Ctrl+Shift+R

### 15.5 Validación Falla en Startup

**Síntoma:** Bot no arranca, error de validación

**Ejemplo error:**
```
ERROR - Validation failed: regime_family 'trending2' not in REGIME_FAMILY_MATRIX
ERROR - Strategy: 06_reversal_long_1H
```

**Solución:**
1. Verificar `regime_family` en YAML es válido
2. Valores permitidos: 'trending', 'ranging', 'volatile', null
3. Corregir typo en YAML
4. Reiniciar

---

# PARTE 6: REFERENCIA RÁPIDA

## 16. Estructuras de Datos

### 16.1 Strategy Dict

```python
{
    'id': '06_reversal_long_1H',
    'function_name': 'add_signals_reversal_long',
    'timeframe': '1H',
    'order_amount': 40.0,
    'tp_pct': 4.0,
    'sl_pct': 10.0,
    'candles_timeout': 50,
    'symbols': 'multi',
    'regime_family': 'trending',  # ⭐ NUEVO
    'description': '...'
}
```

### 16.2 Position Dict

```python
{
    'symbol': 'BTCUSDT',
    'size': '0.000763',
    'entry_price': '94356.8',
    'direction': 'long',
    'tp': '98131.07',
    'sl': '84921.12',
    'order_id': '1391784175051902977',
    'opened_at': '2026-01-15T16:00:10+00:00',
    'usdt_amount': 72.0  # ⭐ Refleja adjusted amount
}
```

### 16.3 Regime Cache

```python
# En memoria (orchestrator)
self.regime_cache = {
    '4H': 'ranging',   # String - régimen actual
    '1H': 'trending',
    '6Hutc': 'volatile'
}

# NO persiste en JSON
# Se recalcula cada vela cerrada
```

### 16.4 Regime Info Dict

```python
{
    'regime': 'trending',
    'metrics': {
        'atr': 2.34,
        'er': 0.67,
        'pe': 0.45
    },
    'timeframe': '1H',
    'matrix': {
        'trending': {'trending': 1.8, 'ranging': 1.0, 'volatile': 0.0},
        'ranging': {'trending': 1.0, 'ranging': 1.8, 'volatile': 0.0},
        'volatile': {'trending': 0.5, 'ranging': 0.5, 'volatile': 1.5}
    },
    'timestamp': '2026-01-15T16:00:05+00:00'
}
```

---

## 17. Comandos y Endpoints

### 17.1 Comandos CLI

```bash
# Iniciar bot (todas las estrategias activas de cuenta)
python3 main.py --account 00

# Iniciar con estrategias específicas
python3 main.py --account 00 --set-active "06,07,10,11"

# Ver ayuda
python3 main.py --help

# Ver logs en vivo
tail -f persistence/bot_files_00/BOT_orchestator_00.log

# Filtrar logs de régimen
grep REGIME persistence/bot_files_00/BOT_orchestator_00.log

# Ver posiciones en JSON
cat persistence/bot_files_00/bot_state_00.json | jq '.positions'
```

### 17.2 API Endpoints

**Dashboard:**
```
http://localhost:5000/                # Dashboard principal
http://localhost:5000/api/health      # Health check
http://localhost:5000/api/status      # Estado completo
```

**Regime:**
```
GET /api/regime/current?timeframe=1H  # Régimen actual
GET /api/regime/matrix                # Matriz completa
GET /api/strategies/regime            # Estrategias con familia
```

**Trading:**
```
GET /api/positions                    # Posiciones activas
GET /api/trades                       # Histórico trades
POST /api/correlation-matrix          # Correlación
```

**Logs:**
```
GET /api/logs/stream                  # Logs nuevos
```

### 17.3 Testing Rápido

```python
# Test clasificador
from market_regime.regime_classifier import get_current_regime

regime, metrics = get_current_regime('1H')
print(f"Regime: {regime}")
print(f"Metrics: {metrics}")

# Test lookup matriz
from config.settings import REGIME_FAMILY_MATRIX

family = 'trending'
market = 'trending'
mult = REGIME_FAMILY_MATRIX[family][market]
print(f"Multiplier: {mult}x")

# Test estrategia tiene familia
from strategies.strategy_loader import load_strategies

strategies = load_strategies()
for s in strategies:
    if s['id'] == '06_reversal_long_1H':
        print(f"Family: {s.get('regime_family')}")
```

---

# 🎉 FIN DE DOCUMENTACIÓN

**BOT_trading v2.3 - Sistema de Trading Automatizado con Custom Regime Multipliers**

---

**Última actualización:** 2026-01-15  
**Autor:** Trading Bot Team  
**Nueva Feature:** Custom Regime Multipliers por Estrategia

---

## 📝 RESUMEN EJECUTIVO

### ¿Qué es Custom Regime Multipliers?

Sistema que permite a cada estrategia definir su "personalidad" (trending/ranging/volatile) y ajustar automáticamente el tamaño de posición según cómo alinee con el régimen actual del mercado.

### Beneficios Clave

1. **Precisión:** Estrategias trending operan más agresivas en mercados trending
2. **Protección:** Bloqueo automático cuando estrategia/mercado no alinean
3. **Flexibilidad:** Cada estrategia puede tener comportamiento único
4. **Backward Compatible:** Estrategias sin familia usan multiplicadores globales

### Configuración Mínima

```yaml
# strategies.yaml
- id: "06_reversal_long_1H"
  regime_family: "trending"  # ← Añadir esta línea
```

```python
# settings.py
REGIME_FAMILY_MATRIX = {
    'trending': {
        'trending': 1.8,   # ← Ajustar multiplicadores
        'ranging': 1.0,
        'volatile': 0.0
    },
    # ...
}
```

### Quick Start

1. Editar `strategies.yaml` → Añadir `regime_family`
2. (Opcional) Ajustar `REGIME_FAMILY_MATRIX` en settings.py
3. Reiniciar bot: `python3 main.py --account 00`
4. Ver dashboard: `http://localhost:5000` → Tab "Regime Matrix"
5. Monitorear logs: `grep MATRIX persistence/bot_files_00/BOT_orchestator_00.log`

---

**¿Preguntas? Consulta la sección [Troubleshooting](#15-troubleshooting) o los ejemplos en [Position Sizing Adaptativo](#14-position-sizing-adaptativo).**
