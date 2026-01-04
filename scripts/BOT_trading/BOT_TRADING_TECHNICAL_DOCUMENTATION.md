# 🤖 BOT TRADING - DOCUMENTACIÓN TÉCNICA COMPLETA

**Sistema de Trading Automatizado Multi-Estrategia**

---

**Versión:** 2.0  
**Fecha:** 2026-01-05  
**Python:** 3.12  
**Framework:** Flask + ccxt + Bitget API  

---

## 📖 PROPÓSITO

Este documento es la referencia técnica completa del sistema BOT_trading. Contiene toda la arquitectura, lógica de negocio, componentes, flujos y configuración del bot de trading.

**Público objetivo:** Desarrolladores que necesiten entender, mantener o extender el sistema.

---

## 📋 TABLA DE CONTENIDOS

### PARTE 1: VISIÓN GENERAL
1. [Introducción al Sistema](#1-introducción-al-sistema)
2. [Arquitectura General](#2-arquitectura-general)
3. [Tecnologías y Stack](#3-tecnologías-y-stack)

### PARTE 2: ARQUITECTURA DE COMPONENTES
4. [Estructura de Directorios](#4-estructura-de-directorios)
5. [Core: BotOrchestrator](#5-core-botaniquestrator)
6. [Strategies: Sistema de Estrategias](#6-strategies-sistema-de-estrategias)
7. [Execution: Cliente Bitget](#7-execution-cliente-bitget)
8. [API: Dashboard Web](#8-api-dashboard-web)
9. [Signals: Funciones de Detección](#9-signals-funciones-de-detección)
10. [Persistence: Sistema de Estado](#10-persistence-sistema-de-estado)

### PARTE 3: CONFIGURACIÓN
11. [settings.py - Configuración Central](#11-settingspy---configuración-central)
12. [strategies.yaml - Definiciones](#12-strategiesyaml---definiciones)
13. [registry.py - Implementaciones](#13-registrypy---implementaciones)
14. [Cuentas de Trading](#14-cuentas-de-trading)

### PARTE 4: LÓGICA DE NEGOCIO
15. [Detección de Señales](#15-detección-de-señales)
16. [Gestión de Posiciones](#16-gestión-de-posiciones)
17. [Take Profit y Stop Loss](#17-take-profit-y-stop-loss)
18. [Timeout por Velas](#18-timeout-por-velas)
19. [Sincronización con Broker](#19-sincronización-con-broker)

### PARTE 5: FLUJOS DE EJECUCIÓN
20. [Ciclo de Vida del Bot](#20-ciclo-de-vida-del-bot)
21. [Flujo de Detección de Velas](#21-flujo-de-detección-de-velas)
22. [Flujo de Apertura de Posición](#22-flujo-de-apertura-de-posición)
23. [Flujo de Cierre de Posición](#23-flujo-de-cierre-de-posición)
24. [Main Loop Explicado](#24-main-loop-explicado)

### PARTE 6: INTEGRACIONES
25. [Bitget API en Detalle](#25-bitget-api-en-detalle)
26. [WebSocket vs REST](#26-websocket-vs-rest)
27. [Manejo de Errores](#27-manejo-de-errores)
28. [Rate Limits y Retry](#28-rate-limits-y-retry)

### PARTE 7: DESARROLLO
29. [Añadir Nueva Estrategia](#29-añadir-nueva-estrategia)
30. [Sistema de Logging](#30-sistema-de-logging)
31. [Testing y Validación](#31-testing-y-validación)
32. [Troubleshooting](#32-troubleshooting)

### PARTE 8: REFERENCIA
33. [Catálogo de Estrategias](#33-catálogo-de-estrategias)
34. [Estructuras de Datos](#34-estructuras-de-datos)
35. [Variables de Estado](#35-variables-de-estado)
36. [Glosario Técnico](#36-glosario-técnico)

---

# PARTE 1: VISIÓN GENERAL

## 1. Introducción al Sistema

### 1.1 ¿Qué es BOT_trading?

**BOT_trading** es un sistema automatizado de trading en futuros de criptomonedas que:

- Opera 24/7 sin intervención humana
- Gestiona múltiples estrategias simultáneamente
- Soporta diferentes timeframes (4H, 1H, 6Hutc, 2m, 5m)
- Ejecuta en múltiples cuentas independientes
- Proporciona monitoreo en tiempo real vía dashboard web

### 1.2 Características Principales

**Trading:**
- ✅ 14 estrategias diferentes implementadas
- ✅ Gestión automática de TP (Take Profit) y SL (Stop Loss)
- ✅ Timeout automático por número de velas
- ✅ Sincronización continua con el broker
- ✅ Soporte para LONG y SHORT

**Arquitectura:**
- ✅ Configuración declarativa (YAML)
- ✅ Separación de responsabilidades (módulos independientes)
- ✅ Sistema de logging dual (consola + archivo)
- ✅ Estado persistente (recuperación tras crashes)
- ✅ Validación exhaustiva de configuración

**Monitoreo:**
- ✅ Dashboard web en tiempo real (Flask)
- ✅ API REST para integración
- ✅ Logs streaming en vivo
- ✅ Métricas de rendimiento
- ✅ Histórico de trades (Excel)

### 1.3 Flujo de Alto Nivel

```
┌─────────────────────────────────────────────────────────┐
│  1. INICIALIZACIÓN                                      │
│     • Cargar configuración (settings.py)                │
│     • Cargar estrategias (strategies.yaml)              │
│     • Conectar a Bitget (API + WebSocket)              │
│     • Recuperar estado anterior (JSON)                  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  2. MAIN LOOP (infinito)                                │
│     ┌───────────────────────────────────────┐          │
│     │ ¿Nueva vela cerrada?                  │          │
│     ├─ SÍ → Procesar estrategias            │          │
│     │        ├─ Sync con broker              │          │
│     │        ├─ Incrementar candles          │          │
│     │        ├─ Check timeout                │          │
│     │        └─ Detectar señales             │          │
│     │                                         │          │
│     ├─ NO → Continuar                        │          │
│     │                                         │          │
│     │ Cada 10s: Verificar TP/SL              │          │
│     └───────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  3. SEÑAL DETECTADA                                     │
│     • Verificar balance                                 │
│     • Calcular size                                     │
│     • Calcular TP/SL                                    │
│     • Enviar orden market                               │
│     • Registrar posición                                │
│     • Guardar estado                                    │
└─────────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  4. TRACKING POSICIÓN                                   │
│     • Verificar TP (cada 10s)                           │
│     • Verificar SL (cada 10s)                           │
│     • Verificar timeout (cada nueva vela)               │
│     • Cerrar si se alcanza cualquier condición          │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Arquitectura General

### 2.1 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                       BOT_trading SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐                                         │
│  │    main.py         │  ← Entry Point                          │
│  │  (Entry Point)     │                                         │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│  ┌──────────────────────────────────────────────────┐          │
│  │          BotOrchestrator (core/)                 │          │
│  │  ┌────────────────────────────────────────────┐ │          │
│  │  │ • Inicialización del sistema               │ │          │
│  │  │ • Gestión del ciclo de vida                │ │          │
│  │  │ • Coordinación de estrategias              │ │          │
│  │  │ • Control de timeframes                    │ │          │
│  │  │ • Main loop                                │ │          │
│  │  └────────────────────────────────────────────┘ │          │
│  └───────┬──────────────┬───────────────┬──────────┘          │
│          │              │               │                      │
│    ┌─────▼─────┐  ┌────▼────┐  ┌──────▼──────┐               │
│    │Strategies │  │Execution│  │  Dashboard  │               │
│    │ System    │  │ Manager │  │   (Flask)   │               │
│    │(YAML+Reg) │  │(Bitget) │  │             │               │
│    └─────┬─────┘  └────┬────┘  └──────┬──────┘               │
│          │              │               │                      │
│    ┌─────▼─────┐  ┌────▼────┐  ┌──────▼──────┐               │
│    │  Signal   │  │  Order  │  │  Templates  │               │
│    │ Detection │  │ Handling│  │    (HTML)   │               │
│    └───────────┘  └─────────┘  └─────────────┘               │
│                         │                                      │
│            ┌────────────▼──────────────┐                      │
│            │  Persistence (JSON/Excel) │                      │
│            └───────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   Bitget Exchange           │
            │   • REST API                │
            │   • WebSocket (public)      │
            │   • WebSocket (private)     │
            └─────────────────────────────┘
```

### 2.2 Flujo de Datos

```
┌──────────────┐
│ Market Data  │
│  (Bitget)    │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐      ┌───────────────────┐
│  WebSocket/REST      │─────▶│  OHLCV Data       │
│  Data Fetcher        │      │  (DataFrame)      │
└──────────────────────┘      └─────────┬─────────┘
                                        │
                                        ▼
┌──────────────────────┐      ┌───────────────────┐
│  Strategy Processor  │◀─────│  Signal Detection │
│  (orchestrator)      │      │  (functions)      │
└──────┬───────────────┘      └───────────────────┘
       │
       ├─ Signal detected? → Open position
       │
       ├─ Position tracking → Check TP/SL/Timeout
       │
       └─ Update state → Save JSON
```

### 2.3 Separación de Responsabilidades

| Módulo | Responsabilidad | Archivos Clave |
|--------|-----------------|----------------|
| **core/** | Orquestación y ciclo de vida | orchestrator.py |
| **strategies/** | Carga y validación de estrategias | strategy_loader.py, registry.py |
| **signals/** | Funciones de detección | Z_add_signals_*.py |
| **execution/** | Comunicación con broker | bitget_client.py |
| **api/** | Dashboard web | backend.py, templates/ |
| **config/** | Configuración central | settings.py |
| **persistence/** | Estado y logs | bot_state_XX.json, logs/ |
| **validation/** | Validación de configuración | strategy_validator.py |
| **bot_utils/** | Utilidades (logger, etc.) | logger.py |

---

## 3. Tecnologías y Stack

### 3.1 Lenguajes y Frameworks

**Python 3.12**
- Lenguaje principal del sistema
- Tipado dinámico con hints opcionales
- Async/await para operaciones concurrentes

**Flask 3.x**
- Framework web para dashboard
- API REST para frontend
- Template engine (Jinja2)

**ccxt**
- Librería unificada para exchanges
- Usado para obtener OHLCV
- Abstracción sobre APIs de exchanges

### 3.2 Librerías Clave

```python
# Trading & Data
import ccxt                    # Exchange abstraction
import requests                # HTTP requests
import hmac, hashlib, base64  # API authentication
import websocket              # WebSocket connections

# Data Processing
import pandas as pd           # DataFrames
import numpy as np            # Arrays numéricos
import yaml                   # YAML parsing

# Sistema
import logging                # Logging system
import json                   # JSON serialization
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo # Timezone handling

# Dashboard
from flask import Flask, render_template, jsonify
```

### 3.3 Formato de Datos

**YAML (strategies.yaml)**
- Definición declarativa de estrategias
- Fácil de leer y modificar
- Validación con pydantic o dict checks

**JSON (bot_state_XX.json)**
- Estado persistente del bot
- Posiciones abiertas
- Contadores de velas

**Excel (bot_trades_XX.xlsx)**
- Histórico de trades
- Análisis de rendimiento
- Auditoría

**Logs (.log files)**
- Rotación automática
- Formato dual (clean + detailed)
- Niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL

### 3.4 APIs Externas

**Bitget API**
- **Base URL:** `https://api.bitget.com`
- **Autenticación:** HMAC SHA256
- **Endpoints:** REST + WebSocket
- **Product Type:** USDT-FUTURES (futuros perpetuos)

**Endpoints principales:**
```
POST /api/v2/mix/order/place           # Colocar orden
GET  /api/v2/mix/position/all-position # Obtener posiciones
GET  /api/v2/mix/account/account       # Obtener balance
GET  /api/v2/mix/market/candles        # OHLCV histórico
```

---

# PARTE 2: ARQUITECTURA DE COMPONENTES

## 4. Estructura de Directorios

### 4.1 Árbol Completo

```
BOT_trading/
│
├── main.py                              # ← Entry point del sistema
│
├── config/                              # Configuración
│   ├── __init__.py
│   └── settings.py                      # Settings centralizados
│
├── core/                                # Componentes core
│   ├── __init__.py
│   └── orchestrator.py                  # BotOrchestrator (cerebro)
│
├── strategies/                          # Sistema de estrategias
│   ├── __init__.py
│   ├── strategies.yaml                  # Definiciones YAML
│   ├── registry.py                      # Mapeo nombre → función
│   ├── strategy_loader.py               # Cargador de YAML
│   └── strategy_processor.py            # Procesador de señales
│
├── signals/                             # Funciones de señales
│   ├── __init__.py
│   ├── Z_add_signals_double_top.py      # Double top pattern
│   ├── Z_add_signals_reversal.py        # Reversal pattern
│   ├── Z_add_signals_parity.py          # Parity pattern
│   └── Z_add_signals_orderblocks.py     # Order blocks pattern
│
├── execution/                           # Ejecución de órdenes
│   ├── __init__.py
│   └── bitget_client.py                 # Cliente API Bitget
│
├── api/                                 # Dashboard web
│   ├── __init__.py
│   ├── backend.py                       # Flask server + API
│   └── templates/
│       └── dashboard.html               # Template HTML
│
├── analytics/                           # Métricas
│   ├── __init__.py
│   └── metrics.py                       # Cálculo de métricas
│
├── persistence/                         # Datos persistentes
│   ├── bot_files_00/                    # Cuenta 00
│   │   ├── BOT_orchestator_00.log       # Logs limpios
│   │   ├── BOT_orchestator_00_detailed.log
│   │   ├── bot_trades_00.xlsx           # Histórico trades
│   │   └── bot_state_00.json            # Estado del bot
│   │
│   ├── bot_files_E1/                    # Cuenta E1
│   └── bot_files_01/                    # Cuenta 01
│
├── bot_utils/                           # Utilidades
│   ├── __init__.py
│   └── logger.py                        # Sistema de logging
│
├── validation/                          # Validaciones
│   ├── __init__.py
│   └── strategy_validator.py            # Validador de configs
│
└── utils/                               # Utils generales
    └── ZZ_connect.py                    # Credenciales (privado)
```

### 4.2 Propósito de Cada Directorio

| Directorio | Propósito | Archivos Clave |
|------------|-----------|----------------|
| **config/** | Configuración central del sistema | settings.py |
| **core/** | Lógica central de orquestación | orchestrator.py |
| **strategies/** | Sistema de estrategias (YAML + registry) | strategies.yaml, registry.py |
| **signals/** | Implementación de funciones de señales | Z_add_signals_*.py |
| **execution/** | Comunicación con broker | bitget_client.py |
| **api/** | Dashboard web y API REST | backend.py, dashboard.html |
| **analytics/** | Métricas y análisis de rendimiento | metrics.py |
| **persistence/** | Estado, logs, trades (datos) | JSON, Excel, logs |
| **bot_utils/** | Herramientas compartidas | logger.py |
| **validation/** | Validación de configuración | strategy_validator.py |
| **utils/** | Utilidades generales | ZZ_connect.py |

### 4.3 Archivos Principales

**main.py**
- Entry point del sistema
- Parse de argumentos (--account, --set-active)
- Inicialización del BotOrchestrator
- Manejo de Ctrl+C

**orchestrator.py**
- Cerebro del sistema
- ~500 líneas de código
- Coordina todos los componentes
- Main loop infinito

**strategies.yaml**
- Definición declarativa de todas las estrategias
- ~400 líneas YAML
- Parámetros de cada estrategia

**registry.py**
- Mapeo nombre → función
- ~50 líneas
- STRATEGY_FUNCTIONS dict

**backend.py**
- Servidor Flask
- ~300 líneas
- API REST + templates

---

## 5. Core: BotOrchestrator

### 5.1 Responsabilidades

El **BotOrchestrator** es el componente central que:

1. **Inicializa** todo el sistema
2. **Coordina** las estrategias
3. **Gestiona** el ciclo de vida
4. **Controla** el main loop
5. **Sincroniza** con el broker

### 5.2 Clase BotOrchestrator

```python
class BotOrchestrator:
    def __init__(self, account_number, bitget_client, connect_bitget_func,
                 active_strategy_ids=None):
        """
        Inicializa el orquestador.
        
        Args:
            account_number: "00", "E1", "01"
            bitget_client: Cliente de Bitget API
            connect_bitget_func: Función para ccxt
            active_strategy_ids: Lista de IDs a activar
        """
        self.account_number = account_number
        self.bitget_client = bitget_client
        self.connect_bitget_func = connect_bitget_func
        self.active_strategy_ids = active_strategy_ids
        
        # Configuración
        self.config = get_account_config(account_number)
        self.base_dir = self.config['base_dir']
        self.log_file = self.config['log_file']
        self.state_file = self.config['state_file']
        self.trades_file = self.config['trades_file']
        self.dashboard_port = self.config['dashboard_port']
        self.initial_capital = self.config['initial_capital']
        
        # Estado global
        self.OPEN_POSITIONS = {}
        self.STRATEGY_CANDLES = {}
        
        # Estrategias
        self.strategies = []
        self.strategies_by_timeframe = {}
        self.symbols_by_strategy = {}
        
        # Dashboard
        self.dashboard_server = None
```

### 5.3 Método run()

El método `run()` es el corazón del sistema:

```python
def run(self):
    """
    Método principal que ejecuta el bot.
    
    Workflow:
    1. Inicialización
       - Cargar estrategias
       - Validar configuración
       - Conectar a Bitget
       - Inicializar dashboard
       - Cargar estado previo
    
    2. Main Loop (infinito)
       - Detectar nuevas velas
       - Procesar estrategias
       - Verificar TP/SL
       - Guardar estado
    """
    
    # ═══════════════════════════════════════════════════════════
    # FASE 1: INICIALIZACIÓN
    # ═══════════════════════════════════════════════════════════
    
    logger.info("=" * 60)
    logger.info("BOT INITIALIZATION STARTED")
    logger.info("=" * 60)
    
    # 1.1 Cargar estrategias desde YAML
    strategy_ids = get_account_strategies(self.account_number)
    self.strategies = load_strategies(strategy_ids)
    
    # 1.2 Aplicar filtro de --set-active
    if self.active_strategy_ids:
        self.strategies = filter_strategies_by_ids(
            self.strategies, 
            self.active_strategy_ids
        )
    
    # 1.3 Validar configuración
    errors, warnings = validate_strategy_configuration(
        self.strategies,
        IMPLEMENTED_STRATEGIES
    )
    
    if errors:
        logger.critical("CONFIGURATION ERRORS FOUND")
        for error in errors:
            logger.error(f"  • {error}")
        sys.exit(1)
    
    # 1.4 Cargar símbolos para cada estrategia
    all_symbols = self.bitget_client.get_all_symbols()
    
    for strat in self.strategies:
        symbols = load_final_symbols(all_symbols, strat, strat['timeframe'])
        self.symbols_by_strategy[strat['id']] = symbols
    
    # 1.5 Agrupar estrategias por timeframe
    self.strategies_by_timeframe = group_strategies_by_timeframe(
        self.strategies
    )
    
    # 1.6 Inicializar dashboard
    self.dashboard_server = DashboardServer(
        account_number=self.account_number,
        base_dir=self.base_dir,
        # ... más parámetros
    )
    self.dashboard_server.start(port=self.dashboard_port)
    
    # 1.7 Cargar estado previo
    self.OPEN_POSITIONS, self.STRATEGY_CANDLES = load_state(self.state_file)
    
    # 1.8 Sincronizar con broker
    sync_broker(self.OPEN_POSITIONS, self.STRATEGY_CANDLES, self.state_file)
    
    # 1.9 Calcular próximas velas
    next_candle_times = {}
    for tf in self.strategies_by_timeframe.keys():
        next_candle_times[tf] = calculate_next_candle_time(tf, HOUR_ZONE)
    
    logger.info("BOT INITIALIZATION COMPLETED")
    logger.info("=" * 60)
    
    # ═══════════════════════════════════════════════════════════
    # FASE 2: MAIN LOOP
    # ═══════════════════════════════════════════════════════════
    
    last_tpsl_check = time.time()
    
    try:
        while True:
            now = datetime.now(HOUR_ZONE)
            
            # ───────────────────────────────────────────────────
            # CHECK 1: ¿Nueva vela cerrada?
            # ───────────────────────────────────────────────────
            closed_timeframes = []
            for tf in self.strategies_by_timeframe.keys():
                if now >= next_candle_times[tf]:
                    closed_timeframes.append(tf)
            
            if closed_timeframes:
                for tf in closed_timeframes:
                    # Procesar estrategias de este timeframe
                    self._process_timeframe(tf)
                    
                    # Recalcular próxima vela
                    next_candle_times[tf] = calculate_next_candle_time(
                        tf, 
                        HOUR_ZONE
                    )
            
            # ───────────────────────────────────────────────────
            # CHECK 2: Verificación periódica TP/SL
            # ───────────────────────────────────────────────────
            current_time = time.time()
            if current_time - last_tpsl_check >= CHECK_INTERVAL:
                check_all_tp_sl(
                    self.strategies,
                    self.OPEN_POSITIONS,
                    self.STRATEGY_CANDLES,
                    self.state_file,
                    self.bitget_client
                )
                last_tpsl_check = current_time
            
            # Pequeño sleep para no saturar CPU
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected - Shutting down gracefully")
        save_state(self.OPEN_POSITIONS, self.STRATEGY_CANDLES, self.state_file)
        logger.info("State saved - Goodbye!")
```

### 5.4 Método _process_timeframe()

```python
def _process_timeframe(self, timeframe):
    """
    Procesa todas las estrategias de un timeframe tras cerrar vela.
    
    Workflow:
    1. Sync con broker
    2. Para cada estrategia:
       a. Si tiene posiciones → incrementar candles + check timeout
       b. Si NO tiene posiciones → buscar señales
    """
    
    logger.info(f"New {timeframe} candle closed")
    
    # 1. Sync con broker
    sync_broker(self.OPEN_POSITIONS, self.STRATEGY_CANDLES, self.state_file)
    
    # 2. Procesar cada estrategia
    strategies = self.strategies_by_timeframe[timeframe]
    
    for strat in strategies:
        strat_id = strat['id']
        
        # ¿Tiene posiciones?
        num_positions = len(self.OPEN_POSITIONS.get(strat_id, []))
        
        if num_positions > 0:
            # Ya tiene posiciones → skip nuevas señales
            candles = self.STRATEGY_CANDLES.get(strat_id, 0)
            logger.info(
                f"Skip {strat_id} - "
                f"{candles}/{strat['sell_after_ncandles']} candles | "
                f"{num_positions} positions"
            )
            
            # Incrementar contador
            increment_strategy_candles(
                strat_id,
                self.STRATEGY_CANDLES,
                self.OPEN_POSITIONS,
                self.state_file
            )
            
            # Check timeout
            check_candles_timeout_for_strategy(
                strat_id,
                strat['sell_after_ncandles'],
                self.OPEN_POSITIONS,
                self.STRATEGY_CANDLES,
                self.state_file,
                self.bitget_client
            )
        
        else:
            # Sin posiciones → buscar señales
            try:
                process_strategy(
                    strat=strat,
                    final_symbols=self.symbols_by_strategy.get(strat_id, []),
                    exchange=self.exchange,
                    open_positions=self.OPEN_POSITIONS,
                    strategy_candles=self.STRATEGY_CANDLES,
                    state_file=self.state_file,
                    send_request_func=self.bitget_client.place_order,
                    get_balance_func=self.bitget_client.get_balance
                )
            except Exception as e:
                logger.error(f"Error processing {strat_id}: {e}")
```

### 5.5 Inicialización desde main.py

```python
# main.py

import argparse
from core.orchestrator import BotOrchestrator
from execution.bitget_client import BitgetClient
from utils.ZZ_connect import connect_bitget_00, connect_bitget_E1, connect_bitget_01

def main():
    # Parse argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', required=True, choices=['00', 'E1', '01'])
    parser.add_argument('--set-active', type=str, help='IDs de estrategias (CSV)')
    args = parser.parse_args()
    
    # Mapeo de clientes
    BITGET_CLIENTS = {
        "00": BitgetClient(API_KEY_00, API_SECRET_00, API_PASS_00),
        "E1": BitgetClient(API_KEY_E1, API_SECRET_E1, API_PASS_E1),
        "01": BitgetClient(API_KEY_01, API_SECRET_01, API_PASS_01)
    }
    
    CCXT_CONNECTIONS = {
        "00": connect_bitget_00,
        "E1": connect_bitget_E1,
        "01": connect_bitget_01
    }
    
    # Obtener cliente y función de conexión
    bitget_client = BITGET_CLIENTS[args.account]
    connect_func = CCXT_CONNECTIONS[args.account]
    
    # Parsear estrategias activas
    active_ids = None
    if args.set_active:
        active_ids = args.set_active.split(',')
    
    # Crear orchestrator
    orchestrator = BotOrchestrator(
        account_number=args.account,
        bitget_client=bitget_client,
        connect_bitget_func=connect_func,
        active_strategy_ids=active_ids
    )
    
    # Run!
    orchestrator.run()

if __name__ == '__main__':
    main()
```

---

## 6. Strategies: Sistema de Estrategias

### 6.1 Arquitectura del Sistema

El sistema de estrategias separa **configuración** (YAML) de **implementación** (Python):

```
┌────────────────────────────────────────────────────────┐
│  CONFIGURATION (strategies.yaml)                       │
│  • Parámetros de estrategias                           │
│  • TP/SL, order_amount, timeframe                      │
│  • Lookback, tolerance, etc.                           │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  LOADER (strategy_loader.py)                           │
│  • Lee YAML                                            │
│  • Parsea configuración                                │
│  • Filtra por IDs                                      │
│  • Retorna lista de dicts                              │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  VALIDATOR (strategy_validator.py)                     │
│  • Verifica parámetros obligatorios                    │
│  • Valida rangos (TP, SL, order_amount)                │
│  • Verifica implementaciones                           │
│  • Retorna errors/warnings                             │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  REGISTRY (registry.py)                                │
│  • Mapeo: nombre → función                             │
│  • STRATEGY_FUNCTIONS dict                             │
│  • get_strategy_function(name)                         │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  PROCESSOR (strategy_processor.py)                     │
│  • Ejecuta detección de señales                        │
│  • Obtiene función desde registry                      │
│  • Descarga OHLCV                                      │
│  • Ejecuta función con parámetros                      │
│  • Retorna símbolos con señal                          │
└────────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  IMPLEMENTATION (Z_add_signals_*.py)                   │
│  • double_top_long()                                   │
│  • reversal_long()                                     │
│  • parity_short()                                      │
│  • orderblocks_long()                                  │
│  • etc.                                                │
└────────────────────────────────────────────────────────┘
```

### 6.2 strategies.yaml - Estructura

```yaml
strategies:
  # ────────────────────────────────────────────────────────
  # STRATEGY 01: Double Top Long (4H timeframe)
  # ────────────────────────────────────────────────────────
  - id: '01_double_top_long_4H'
    name: 'double_top_long_4H'
    timeframe: '4H'
    active: true
    direction: 'long'
    sell_after_ncandles: 50
    order_amount: 40
    
    # Parámetros específicos de double_top
    lookback: 2
    tolerance: 15
    trend_th: 5
    
    # TP/SL
    tp_pct: 4
    sl_pct: 10
  
  # ────────────────────────────────────────────────────────
  # STRATEGY 02: Reversal Long (4H timeframe)
  # ────────────────────────────────────────────────────────
  - id: '02_reversal_long_4H'
    name: 'reversal_long_4H'
    timeframe: '4H'
    active: true
    direction: 'long'
    sell_after_ncandles: 50
    order_amount: 40
    
    # Parámetros específicos de reversal
    lookback: 4
    tolerance: 20
    ma_period: 50
    
    # TP/SL
    tp_pct: 3
    sl_pct: 10
```

**Parámetros obligatorios:**
- `id`: Identificador único
- `name`: Nombre de función (debe estar en registry)
- `timeframe`: Timeframe de operación
- `active`: true/false
- `direction`: 'long' o 'short'
- `sell_after_ncandles`: Timeout en velas
- `order_amount`: USDT por posición
- `tp_pct`: Take profit (%)
- `sl_pct`: Stop loss (%)

**Parámetros específicos por tipo:**
- `lookback`, `tolerance`, `trend_th` (double_top)
- `lookback`, `tolerance`, `ma_period` (reversal)
- `lookback`, `tolerance` (parity)
- `lookback`, `tolerance`, `impulse` (orderblocks)

### 6.3 registry.py - Mapeo

```python
# strategies/registry.py

from signals.Z_add_signals_double_top import double_top_long
from signals.Z_add_signals_reversal import reversal_long, reversal_short
from signals.Z_add_signals_parity import parity_long, parity_short
from signals.Z_add_signals_orderblocks import orderblocks_long, orderblocks_short

# Mapeo: nombre de estrategia → función de señal
STRATEGY_FUNCTIONS = {
    # Double Top
    'double_top_long_4H': double_top_long,
    
    # Reversal (4H)
    'reversal_long_4H': reversal_long,
    'reversal_short_4H': reversal_short,
    
    # Parity (4H)
    'parity_long_4H': parity_long,
    'parity_short_4H': parity_short,
    
    # Reversal (1H) - MISMA FUNCIÓN, diferente timeframe
    'reversal_long_1H': reversal_long,
    'reversal_short_1H': reversal_short,
    
    # Reversal (6Hutc)
    'reversal_long_6Hutc': reversal_long,
    'reversal_short_6Hutc': reversal_short,
    
    # Parity (1H)
    'parity_long_1H': parity_long,
    'parity_short_1H': parity_short,
    
    # Parity (6Hutc)
    'parity_long_6Hutc': parity_long,
    
    # Order Blocks
    'orderblocks_short_4H': orderblocks_short,
    'orderblocks_long_4H': orderblocks_long,
}

# Set de estrategias implementadas (para validación)
IMPLEMENTED_STRATEGIES = set(STRATEGY_FUNCTIONS.keys())

def get_strategy_function(strategy_name: str):
    """
    Obtiene la función de detección de señal para una estrategia.
    
    Args:
        strategy_name: Nombre de la estrategia (ej: 'reversal_long_4H')
    
    Returns:
        Función de detección
    
    Raises:
        KeyError: Si la estrategia no está implementada
    """
    if strategy_name not in STRATEGY_FUNCTIONS:
        raise KeyError(f"Strategy '{strategy_name}' not implemented in registry")
    
    return STRATEGY_FUNCTIONS[strategy_name]
```

**Nota importante:** Una misma función puede usarse con diferentes nombres de estrategia. Por ejemplo, `reversal_long()` se usa para:
- `reversal_long_4H`
- `reversal_long_1H`
- `reversal_long_6Hutc`

La diferencia está en los **parámetros** (definidos en YAML) y el **timeframe**.

### 6.4 strategy_loader.py - Carga de YAML

```python
# strategies/strategy_loader.py

import yaml
import os

def load_strategies_from_yaml(yaml_path: str) -> list:
    """
    Carga todas las estrategias desde YAML.
    
    Returns:
        Lista de diccionarios con configuración
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data.get('strategies', [])

def filter_strategies_by_ids(strategies: list, strategy_ids: list) -> list:
    """
    Filtra estrategias por IDs específicos.
    
    Args:
        strategies: Lista completa de estrategias
        strategy_ids: Lista de IDs a mantener
    
    Returns:
        Lista filtrada
    """
    return [s for s in strategies if s['id'] in strategy_ids]

def load_strategies(strategy_ids: list) -> list:
    """
    Función principal: carga YAML y filtra por IDs.
    
    Workflow:
    1. Localizar strategies.yaml
    2. Cargar todas las estrategias
    3. Filtrar por los IDs proporcionados
    4. Validar que todos los IDs existen
    5. Retornar lista filtrada
    
    Args:
        strategy_ids: Lista de IDs a cargar
    
    Returns:
        Lista de estrategias (dicts)
    
    Raises:
        ValueError: Si algún ID no existe en YAML
    """
    # Localizar YAML
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        'strategies.yaml'
    )
    
    # Cargar todas
    all_strategies = load_strategies_from_yaml(yaml_path)
    
    # Filtrar por IDs
    filtered = filter_strategies_by_ids(all_strategies, strategy_ids)
    
    # Verificar que todos los IDs solicitados existen
    loaded_ids = {s['id'] for s in filtered}
    missing = set(strategy_ids) - loaded_ids
    
    if missing:
        raise ValueError(
            f"Strategies not found in YAML: {missing}\n"
            f"Available IDs: {[s['id'] for s in all_strategies]}"
        )
    
    return filtered

def group_strategies_by_timeframe(strategies: list) -> dict:
    """
    Agrupa estrategias por timeframe.
    
    Returns:
        Dict: {timeframe: [strategy1, strategy2, ...]}
    """
    grouped = {}
    
    for strat in strategies:
        tf = strat['timeframe']
        if tf not in grouped:
            grouped[tf] = []
        grouped[tf].append(strat)
    
    return grouped
```

### 6.5 strategy_processor.py - Procesamiento

```python
# strategies/strategy_processor.py

from strategies.registry import get_strategy_function
import logging

logger = logging.getLogger('BOT_trading.strategies')

class StrategyProcessor:
    def __init__(self, use_hardcoded: bool = False):
        """
        Procesador de estrategias.
        
        Args:
            use_hardcoded: Si True, retorna señales hardcoded
                          Si False, ejecuta funciones reales
        """
        self.use_hardcoded = use_hardcoded
    
    def detect_signals(self, strategy, symbols, exchange):
        """
        Detecta señales para una estrategia en una lista de símbolos.
        
        Workflow:
        1. Obtener función de señal desde registry
        2. Para cada símbolo:
           a. Descargar OHLCV
           b. Ejecutar función de señal
           c. Si señal positiva → añadir a lista
        3. Retornar símbolos con señal
        
        Args:
            strategy: Dict con configuración de estrategia
            symbols: Lista de símbolos a analizar
            exchange: Objeto ccxt exchange
        
        Returns:
            Lista de símbolos con señal detectada
        """
        if self.use_hardcoded:
            return self._hardcoded_signals()
        
        # Obtener función de señal
        strategy_func = get_strategy_function(strategy['name'])
        
        detected = []
        
        for symbol in symbols:
            try:
                # Descargar OHLCV
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=strategy['timeframe'],
                    limit=200
                )
                
                # Convertir a formato esperado por función
                ohlcv_array = self._convert_to_array(ohlcv)
                
                # Ejecutar detección
                signals = strategy_func(
                    ohlcv_array,
                    lookback=strategy.get('lookback', 5),
                    tolerance=strategy.get('tolerance', 20),
                    # Parámetros opcionales según tipo de estrategia
                    trend_th=strategy.get('trend_th'),
                    ma_period=strategy.get('ma_period'),
                    impulse=strategy.get('impulse'),
                    live_trading=True  # Solo última señal
                )
                
                # Si señal positiva (última vela)
                if signals[-1] != 0:
                    detected.append(symbol)
                    logger.info(f"Signal detected: {symbol}")
            
            except Exception as e:
                logger.error(f"Error detecting signal for {symbol}: {e}")
                continue
        
        return detected
    
    def _convert_to_array(self, ohlcv):
        """
        Convierte OHLCV de ccxt a formato array.
        
        ccxt format:
        [[timestamp, open, high, low, close, volume], ...]
        
        Expected format:
        {
            'open': np.array([...]),
            'high': np.array([...]),
            'low': np.array([...]),
            'close': np.array([...]),
            'volume': np.array([...])
        }
        """
        import numpy as np
        
        return {
            'open': np.array([x[1] for x in ohlcv]),
            'high': np.array([x[2] for x in ohlcv]),
            'low': np.array([x[3] for x in ohlcv]),
            'close': np.array([x[4] for x in ohlcv]),
            'volume': np.array([x[5] for x in ohlcv]),
        }
    
    def _hardcoded_signals(self):
        """Señales hardcoded para testing."""
        return ['BTCUSDT', 'BNBUSDT']
```

---

## 7. Execution: Cliente Bitget

### 7.1 Clase BitgetClient

```python
# execution/bitget_client.py

import requests
import hmac
import hashlib
import base64
import time
import json
from datetime import datetime

class BitgetClient:
    def __init__(self, api_key, api_secret, api_passphrase):
        """
        Cliente para API de Bitget.
        
        Args:
            api_key: API key
            api_secret: API secret
            api_passphrase: API passphrase
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = "https://api.bitget.com"
    
    def _sign_request(self, timestamp, method, request_path, body=''):
        """
        Firma la petición con HMAC SHA256.
        
        Bitget requiere:
        message = timestamp + method + request_path + body
        signature = HMAC-SHA256(message, secret)
        
        Args:
            timestamp: Timestamp en milisegundos (string)
            method: GET, POST, etc.
            request_path: /api/v2/mix/order/place
            body: JSON string (si POST)
        
        Returns:
            Base64 encoded signature
        """
        message = timestamp + method + request_path + body
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode()
    
    def _request(self, method, endpoint, params=None, data=None):
        """
        Ejecuta petición HTTP a Bitget API.
        
        Args:
            method: GET, POST
            endpoint: /api/v2/mix/order/place
            params: Query params (dict)
            data: Body data (dict)
        
        Returns:
            Response JSON
        """
        timestamp = str(int(time.time() * 1000))
        request_path = endpoint
        
        # Añadir query params al path
        if params:
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            request_path += f"?{query_string}"
        
        # Body
        body = ''
        if data:
            body = json.dumps(data)
        
        # Firma
        signature = self._sign_request(timestamp, method, request_path, body)
        
        # Headers
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.api_passphrase,
            'Content-Type': 'application/json',
            'locale': 'en-US'
        }
        
        # Request
        url = self.base_url + endpoint
        
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Parse response
        return response.json()
```

### 7.2 Métodos Principales

```python
    def place_order(self, symbol, side, size, product_type='USDT-FUTURES'):
        """
        Coloca una orden market.
        
        Args:
            symbol: BTCUSDT, ETHUSDT, etc.
            side: 'buy' o 'sell'
            size: Tamaño de la posición
            product_type: 'USDT-FUTURES'
        
        Returns:
            Response dict con:
            {
                'orderId': '...',
                'priceAvg': '91167.7',
                'size': '0.001',
                ...
            }
        """
        endpoint = '/api/v2/mix/order/place'
        
        data = {
            'symbol': symbol,
            'productType': product_type,
            'marginMode': 'crossed',
            'marginCoin': 'USDT',
            'size': str(size),
            'side': side,
            'tradeSide': 'open',
            'orderType': 'market'
        }
        
        response = self._request('POST', endpoint, data=data)
        
        if response.get('code') != '00000':
            raise Exception(f"Order failed: {response.get('msg')}")
        
        return response['data']
    
    def close_position(self, symbol, size, side, product_type='USDT-FUTURES'):
        """
        Cierra una posición.
        
        Args:
            symbol: BTCUSDT
            size: Tamaño a cerrar
            side: 'buy' (para cerrar short) o 'sell' (para cerrar long)
        """
        endpoint = '/api/v2/mix/order/place'
        
        data = {
            'symbol': symbol,
            'productType': product_type,
            'marginMode': 'crossed',
            'marginCoin': 'USDT',
            'size': str(size),
            'side': side,
            'tradeSide': 'close',
            'orderType': 'market'
        }
        
        response = self._request('POST', endpoint, data=data)
        
        if response.get('code') != '00000':
            raise Exception(f"Close failed: {response.get('msg')}")
        
        return response['data']
    
    def get_all_positions(self, product_type='USDT-FUTURES'):
        """
        Obtiene todas las posiciones abiertas.
        
        Returns:
            Lista de posiciones:
            [
                {
                    'symbol': 'BTCUSDT',
                    'holdSide': 'long',
                    'total': '0.001',
                    'available': '0.001',
                    'openPriceAvg': '91000',
                    ...
                }
            ]
        """
        endpoint = '/api/v2/mix/position/all-position'
        params = {'productType': product_type}
        
        response = self._request('GET', endpoint, params=params)
        
        if response.get('code') != '00000':
            raise Exception(f"Get positions failed: {response.get('msg')}")
        
        return response.get('data', [])
    
    def get_balance(self, product_type='USDT-FUTURES'):
        """
        Obtiene balance disponible en USDT.
        
        Returns:
            Float con USDT disponible
        """
        endpoint = '/api/v2/mix/account/account'
        params = {
            'productType': product_type,
            'marginCoin': 'USDT'
        }
        
        response = self._request('GET', endpoint, params=params)
        
        if response.get('code') != '00000':
            raise Exception(f"Get balance failed: {response.get('msg')}")
        
        data = response['data']
        return float(data.get('available', 0))
    
    def get_all_symbols(self):
        """
        Obtiene todos los símbolos disponibles.
        
        Returns:
            Lista de símbolos: ['BTCUSDT', 'ETHUSDT', ...]
        """
        endpoint = '/api/v2/mix/market/contracts'
        params = {'productType': 'USDT-FUTURES'}
        
        response = self._request('GET', endpoint, params=params)
        
        if response.get('code') != '00000':
            return []
        
        symbols = []
        for contract in response.get('data', []):
            symbols.append(contract['symbol'])
        
        return symbols
```

### 7.3 Manejo de Errores

```python
def place_order_with_retry(self, symbol, side, size, max_retries=3):
    """
    Coloca orden con retry automático.
    
    Args:
        symbol, side, size: Parámetros de orden
        max_retries: Número máximo de intentos
    
    Returns:
        Response data
    
    Raises:
        Exception si falla tras todos los retries
    """
    for attempt in range(max_retries):
        try:
            return self.place_order(symbol, side, size)
        
        except Exception as e:
            error_msg = str(e)
            
            # Errores no recuperables
            if 'Invalid API key' in error_msg:
                raise
            
            if 'Insufficient balance' in error_msg:
                raise
            
            # Errores recuperables
            if attempt < max_retries - 1:
                logger.warning(
                    f"Order attempt {attempt + 1}/{max_retries} failed: {error_msg}"
                )
                time.sleep(2 ** attempt)  # Backoff exponencial
                continue
            else:
                raise Exception(f"Order failed after {max_retries} attempts: {error_msg}")
```

---

Continúo con las siguientes secciones...

¿Quieres que continúe generando el README técnico completo? 📚

## 8. API: Dashboard Web

### 8.1 DashboardServer

```python
# api/backend.py

from flask import Flask, render_template, jsonify, request
import threading
import os
import json
import re

class DashboardServer:
    def __init__(self, account_number, base_dir, get_current_price_func,
                 get_balance_func, strategies_config, color_code=None,
                 initial_capital=0, implemented_strategies=None,
                 symbols_by_strategy=None):
        """
        Servidor Flask para dashboard web.
        
        Args:
            account_number: "00", "E1", "01"
            base_dir: Directorio base de archivos
            get_current_price_func: Función para obtener precio actual
            get_balance_func: Función para obtener balance
            strategies_config: Lista de estrategias cargadas
            initial_capital: Capital inicial de la cuenta
            implemented_strategies: Set de estrategias implementadas
            symbols_by_strategy: Dict de símbolos por estrategia
        """
        self.app = Flask(__name__)
        self.account_number = account_number
        self.base_dir = base_dir
        self.get_current_price = get_current_price_func
        self.get_balance = get_balance_func
        self.strategies = strategies_config
        self.initial_capital = initial_capital
        self.implemented_strategies = implemented_strategies or set()
        self.symbols_by_strategy = symbols_by_strategy or {}
        
        # Log file
        self.log_file = os.path.join(base_dir, f'BOT_orchestator_{account_number}.log')
        self.app.last_log_position = 0
        
        # State file
        self.state_file = os.path.join(base_dir, f'bot_state_{account_number}.json')
        
        # Trades file
        self.trades_file = os.path.join(base_dir, f'bot_trades_{account_number}.xlsx')
        
        # Registrar routes
        self._register_routes()
    
    def _register_routes(self):
        """Registra todos los endpoints del API."""
        
        @self.app.route('/')
        def index():
            """Dashboard principal (HTML)."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """Estado completo del bot."""
            try:
                # Cargar estado
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                positions = state.get('positions', {})
                
                # Obtener balance
                balance = self.get_balance()
                
                # Calcular profit
                profit = balance - self.initial_capital
                profit_pct = (profit / self.initial_capital * 100) if self.initial_capital > 0 else 0
                
                # Agrupar posiciones por símbolo
                grouped_positions = self._group_positions(positions)
                
                # Info de estrategias
                strategies_info = []
                for strat in self.strategies:
                    strat_id = strat['id']
                    
                    # Estado
                    if not strat['active']:
                        status = 'DEPRECATING'
                    elif strat['name'] not in self.implemented_strategies:
                        status = 'NOT IMPLEMENTED'
                    else:
                        status = 'ACTIVE'
                    
                    # Posiciones
                    num_positions = len(positions.get(strat_id, []))
                    
                    # Símbolos
                    num_symbols = len(self.symbols_by_strategy.get(strat_id, []))
                    
                    strategies_info.append({
                        'id': strat_id,
                        'name': strat['name'],
                        'timeframe': strat['timeframe'],
                        'direction': strat['direction'],
                        'status': status,
                        'positions': num_positions,
                        'symbols': num_symbols
                    })
                
                return jsonify({
                    'account': self.account_number,
                    'status': 'running',
                    'balance': balance,
                    'initial_capital': self.initial_capital,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'positions': grouped_positions,
                    'strategies': strategies_info,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/logs/stream')
        def stream_logs():
            """Streaming de logs (incremental)."""
            try:
                if not os.path.exists(self.log_file):
                    return jsonify({'logs': [], 'timestamp': None})
                
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    # Ir a última posición leída
                    f.seek(self.app.last_log_position)
                    
                    # Leer nuevas líneas
                    new_lines = f.readlines()
                    
                    # Guardar nueva posición
                    self.app.last_log_position = f.tell()
                
                # Filtrar códigos ANSI y extraer mensajes
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                clean_lines = []
                for line in new_lines:
                    # Remover códigos ANSI
                    line = ansi_escape.sub('', line)
                    line = line.strip()
                    
                    if line:
                        # Extraer solo el mensaje (después del último ' - ')
                        if ' - ' in line:
                            message = line.split(' - ')[-1]
                            clean_lines.append(message)
                        else:
                            clean_lines.append(line)
                
                return jsonify({
                    'logs': clean_lines,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                return jsonify({'error': str(e), 'logs': []}), 500
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'ready',
                'account': self.account_number,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/trades')
        def api_trades():
            """Histórico de trades desde Excel."""
            try:
                if not os.path.exists(self.trades_file):
                    return jsonify({'trades': []})
                
                import pandas as pd
                df = pd.read_excel(self.trades_file)
                
                trades = df.to_dict('records')
                
                return jsonify({'trades': trades})
            
            except Exception as e:
                return jsonify({'error': str(e), 'trades': []}), 500
    
    def _group_positions(self, positions):
        """Agrupa posiciones por símbolo."""
        grouped = {}
        
        for strat_id, strat_positions in positions.items():
            for pos in strat_positions:
                symbol = pos['symbol']
                
                if symbol not in grouped:
                    grouped[symbol] = {
                        'symbol': symbol,
                        'direction': pos['direction'],
                        'total_size': 0,
                        'avg_entry': 0,
                        'strategies': []
                    }
                
                grouped[symbol]['total_size'] += float(pos['size'])
                grouped[symbol]['strategies'].append(strat_id)
        
        return list(grouped.values())
    
    def start(self, port=5000):
        """Inicia el servidor en un thread separado."""
        def run_flask():
            self.app.run(host='0.0.0.0', port=port, debug=False)
        
        thread = threading.Thread(target=run_flask, daemon=True)
        thread.start()
        
        logger.info(f"Dashboard started at http://localhost:{port}")
```

### 8.2 Template HTML

```html
<!-- api/templates/dashboard.html -->

<!DOCTYPE html>
<html>
<head>
    <title>Bot Trading Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        .stat-box {
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }
        .profit-positive {
            color: #00ff00;
        }
        .profit-negative {
            color: #ff0000;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: #2d2d2d;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        th {
            background-color: #3d3d3d;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-top: 1px solid #3d3d3d;
        }
        .badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }
        .badge-active {
            background-color: #00ff00;
            color: #000;
        }
        .badge-deprecating {
            background-color: #888;
            color: #fff;
        }
        .badge-not-implemented {
            background-color: #ff0000;
            color: #fff;
        }
        .logs-container {
            background-color: #1a1a1a;
            border: 1px solid #3d3d3d;
            border-radius: 8px;
            padding: 15px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .log-line {
            padding: 2px 0;
            color: #00ff00;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Bot Trading Dashboard - Account <span id="account-number">--</span></h1>
        </div>
        
        <!-- Stats -->
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Balance</div>
                <div class="stat-value" id="balance">--</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Initial Capital</div>
                <div class="stat-value" id="initial-capital">--</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Profit ($)</div>
                <div class="stat-value" id="profit">--</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Profit (%)</div>
                <div class="stat-value" id="profit-pct">--</div>
            </div>
        </div>
        
        <!-- Positions -->
        <h2>Positions</h2>
        <table id="positions-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Size</th>
                    <th>Strategies</th>
                </tr>
            </thead>
            <tbody id="positions-body">
                <tr><td colspan="4" style="text-align: center;">No positions</td></tr>
            </tbody>
        </table>
        
        <!-- Strategies -->
        <h2>Strategies</h2>
        <table id="strategies-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Timeframe</th>
                    <th>Direction</th>
                    <th>Status</th>
                    <th>Positions</th>
                    <th>Symbols</th>
                </tr>
            </thead>
            <tbody id="strategies-body">
                <tr><td colspan="7" style="text-align: center;">Loading...</td></tr>
            </tbody>
        </table>
        
        <!-- Logs -->
        <h2>Logs</h2>
        <div class="logs-container" id="logs-container"></div>
    </div>
    
    <script>
        // Update dashboard every 2 seconds
        setInterval(updateDashboard, 2000);
        
        // Update logs every 1 second
        setInterval(updateLogs, 1000);
        
        // Initial update
        updateDashboard();
        updateLogs();
        
        async function updateDashboard() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update stats
                document.getElementById('account-number').textContent = data.account;
                document.getElementById('balance').textContent = data.balance.toFixed(2);
                document.getElementById('initial-capital').textContent = data.initial_capital.toFixed(2);
                
                const profit = data.profit;
                const profitElem = document.getElementById('profit');
                profitElem.textContent = profit.toFixed(2);
                profitElem.className = 'stat-value ' + (profit >= 0 ? 'profit-positive' : 'profit-negative');
                
                const profitPctElem = document.getElementById('profit-pct');
                profitPctElem.textContent = data.profit_pct.toFixed(2) + '%';
                profitPctElem.className = 'stat-value ' + (profit >= 0 ? 'profit-positive' : 'profit-negative');
                
                // Update positions
                updatePositionsTable(data.positions);
                
                // Update strategies
                updateStrategiesTable(data.strategies);
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions-body');
            
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(pos => `
                <tr>
                    <td>${pos.symbol}</td>
                    <td>${pos.direction.toUpperCase()}</td>
                    <td>${pos.total_size}</td>
                    <td>${pos.strategies.join(', ')}</td>
                </tr>
            `).join('');
        }
        
        function updateStrategiesTable(strategies) {
            const tbody = document.getElementById('strategies-body');
            
            tbody.innerHTML = strategies.map(strat => {
                let badgeClass = '';
                if (strat.status === 'ACTIVE') badgeClass = 'badge-active';
                else if (strat.status === 'DEPRECATING') badgeClass = 'badge-deprecating';
                else badgeClass = 'badge-not-implemented';
                
                return `
                    <tr>
                        <td>${strat.id}</td>
                        <td>${strat.name}</td>
                        <td>${strat.timeframe}</td>
                        <td>${strat.direction.toUpperCase()}</td>
                        <td><span class="badge ${badgeClass}">${strat.status}</span></td>
                        <td>${strat.positions}</td>
                        <td>${strat.symbols}</td>
                    </tr>
                `;
            }).join('');
        }
        
        async function updateLogs() {
            try {
                const response = await fetch('/api/logs/stream');
                const data = await response.json();
                
                if (data.logs && data.logs.length > 0) {
                    const container = document.getElementById('logs-container');
                    
                    data.logs.forEach(log => {
                        const div = document.createElement('div');
                        div.className = 'log-line';
                        div.textContent = log;
                        container.appendChild(div);
                    });
                    
                    // Auto-scroll to bottom
                    container.scrollTop = container.scrollHeight;
                }
            } catch (error) {
                console.error('Error updating logs:', error);
            }
        }
    </script>
</body>
</html>
```

---

## 9. Signals: Funciones de Detección

### 9.1 Estructura General

Todas las funciones de señales siguen esta estructura:

```python
def signal_function(ohlcv_array, lookback=5, tolerance=20, 
                    specific_param=None, live_trading=False):
    """
    Detecta patrón específico.
    
    Args:
        ohlcv_array: Dict con arrays numpy
            {
                'open': np.array([...]),
                'high': np.array([...]),
                'low': np.array([...]),
                'close': np.array([...]),
                'volume': np.array([...])
            }
        lookback: Velas hacia atrás
        tolerance: Tolerancia de precio (%)
        specific_param: Parámetro específico de esta estrategia
        live_trading: Si True, retorna solo última señal
    
    Returns:
        Array de señales (0 = no señal, 1 = señal)
        Si live_trading=True, retorna array con solo última posición
    """
    # Extraer arrays
    close = ohlcv_array['close']
    high = ohlcv_array['high']
    low = ohlcv_array['low']
    
    # Inicializar señales
    signals = np.zeros(len(close))
    
    # Loop sobre velas
    for i in range(lookback, len(close)):
        # LÓGICA DE DETECCIÓN DEL PATRÓN
        # ...
        
        if pattern_detected:
            signals[i] = 1
    
    # Si live trading, retornar solo última señal
    if live_trading:
        return signals[-1:]
    
    return signals
```

### 9.2 Ejemplo: Double Top Long

```python
# signals/Z_add_signals_double_top.py

import numpy as np

def double_top_long(ohlcv_array, lookback=2, tolerance=15, trend_th=5, 
                    live_trading=False):
    """
    Detecta patrón de doble techo para long.
    
    Lógica:
    1. Buscar dos techos consecutivos a similar altura
    2. Precio actual debe estar por debajo de los techos
    3. Confirmar con trend threshold
    
    Args:
        lookback: Velas hacia atrás para buscar techos
        tolerance: Tolerancia de precio (en %)
        trend_th: Threshold de tendencia
        live_trading: Si True, solo última señal
    
    Returns:
        Array de señales
    """
    close = ohlcv_array['close']
    high = ohlcv_array['high']
    low = ohlcv_array['low']
    
    signals = np.zeros(len(close))
    
    for i in range(lookback + 1, len(close)):
        # Buscar techos en ventana
        window_highs = high[i-lookback:i+1]
        
        # Encontrar índices de los 2 máximos más altos
        sorted_indices = np.argsort(window_highs)[-2:]
        
        peak1_idx = i - lookback + sorted_indices[0]
        peak2_idx = i - lookback + sorted_indices[1]
        
        peak1 = high[peak1_idx]
        peak2 = high[peak2_idx]
        
        # Verificar que están a similar altura
        diff_pct = abs(peak1 - peak2) / peak1 * 100
        
        if diff_pct <= tolerance:
            # Verificar que precio actual está por debajo
            current_price = close[i]
            avg_peak = (peak1 + peak2) / 2
            
            if current_price < avg_peak * (1 - trend_th / 100):
                # Señal de long
                signals[i] = 1
    
    if live_trading:
        return signals[-1:]
    
    return signals
```

### 9.3 Ejemplo: Reversal Long

```python
# signals/Z_add_signals_reversal.py

import numpy as np

def reversal_long(ohlcv_array, lookback=4, tolerance=20, ma_period=50,
                  live_trading=False):
    """
    Detecta reversión de tendencia para long.
    
    Lógica:
    1. Precio debe estar en zona de soporte
    2. Confirmación con MA (media móvil)
    3. Mínimo local + rebote
    
    Args:
        lookback: Velas hacia atrás
        tolerance: Tolerancia de precio
        ma_period: Período de media móvil
        live_trading: Solo última señal
    
    Returns:
        Array de señales
    """
    close = ohlcv_array['close']
    low = ohlcv_array['low']
    
    signals = np.zeros(len(close))
    
    # Calcular MA
    ma = np.zeros(len(close))
    for i in range(ma_period, len(close)):
        ma[i] = np.mean(close[i-ma_period:i])
    
    for i in range(max(lookback, ma_period) + 1, len(close)):
        # Verificar mínimo local
        window_lows = low[i-lookback:i+1]
        is_local_min = low[i-1] == np.min(window_lows)
        
        if is_local_min:
            # Verificar rebote
            price_increase = (close[i] - low[i-1]) / low[i-1] * 100
            
            # Verificar que está cerca de MA
            distance_to_ma = abs(close[i] - ma[i]) / ma[i] * 100
            
            if price_increase > tolerance / 10 and distance_to_ma < tolerance:
                signals[i] = 1
    
    if live_trading:
        return signals[-1:]
    
    return signals

def reversal_short(ohlcv_array, lookback=4, tolerance=20, ma_period=50,
                   live_trading=False):
    """
    Detecta reversión de tendencia para short.
    
    Lógica invertida de reversal_long.
    """
    close = ohlcv_array['close']
    high = ohlcv_array['high']
    
    signals = np.zeros(len(close))
    
    # Calcular MA
    ma = np.zeros(len(close))
    for i in range(ma_period, len(close)):
        ma[i] = np.mean(close[i-ma_period:i])
    
    for i in range(max(lookback, ma_period) + 1, len(close)):
        # Verificar máximo local
        window_highs = high[i-lookback:i+1]
        is_local_max = high[i-1] == np.max(window_highs)
        
        if is_local_max:
            # Verificar caída
            price_decrease = (high[i-1] - close[i]) / high[i-1] * 100
            
            # Verificar que está cerca de MA
            distance_to_ma = abs(close[i] - ma[i]) / ma[i] * 100
            
            if price_decrease > tolerance / 10 and distance_to_ma < tolerance:
                signals[i] = 1
    
    if live_trading:
        return signals[-1:]
    
    return signals
```

---

## 10. Persistence: Sistema de Estado

### 10.1 Estructura de bot_state_XX.json

```json
{
  "positions": {
    "01_double_top_long_4H": [
      {
        "symbol": "BTCUSDT",
        "size": "0.0004",
        "entry_price": "91167.7",
        "direction": "long",
        "tp": "94734.408",
        "sl": "82051.03",
        "order_id": "1391784175051902977",
        "opened_at": "2026-01-04T18:58:46.394725+00:00",
        "usdt_amount": 40.0
      }
    ],
    "02_reversal_long_4H": []
  },
  "strategy_candles": {
    "01_double_top_long_4H": 5,
    "02_reversal_long_4H": 0
  }
}
```

### 10.2 Funciones de Persistencia

```python
# persistence/state_manager.py

import json
import os

def save_state(open_positions, strategy_candles, state_file):
    """
    Guarda estado completo a JSON.
    
    Args:
        open_positions: Dict de posiciones por estrategia
        strategy_candles: Dict de contadores de velas
        state_file: Path al archivo JSON
    """
    state = {
        'positions': open_positions,
        'strategy_candles': strategy_candles
    }
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    # Guardar
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def load_state(state_file):
    """
    Carga estado desde JSON.
    
    Args:
        state_file: Path al archivo JSON
    
    Returns:
        (open_positions, strategy_candles)
        Si no existe el archivo, retorna ({}, {})
    """
    if not os.path.exists(state_file):
        return {}, {}
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    return state.get('positions', {}), state.get('strategy_candles', {})
```

### 10.3 Logging de Trades

```python
# persistence/trade_logger.py

import pandas as pd
import os

def log_trade_to_excel(symbol, profit, profit_pct, close_reason, strategy_id,
                       entry_price, exit_price, size, duration, direction,
                       trades_file):
    """
    Registra un trade cerrado en Excel.
    
    Args:
        symbol: BTCUSDT, ETHUSDT, etc.
        profit: Ganancia/pérdida en USD
        profit_pct: Ganancia/pérdida en %
        close_reason: 'TP', 'SL', 'TIMEOUT'
        strategy_id: ID de la estrategia
        entry_price: Precio de entrada
        exit_price: Precio de salida
        size: Tamaño de posición
        duration: Duración en velas
        direction: 'long' o 'short'
        trades_file: Path al archivo Excel
    """
    trade_data = {
        'DATE': [datetime.now()],
        'SYMBOL': [symbol],
        'STRATEGY': [strategy_id],
        'DIRECTION': [direction],
        'ENTRY_PRICE': [entry_price],
        'EXIT_PRICE': [exit_price],
        'SIZE': [size],
        'PROFIT': [profit],
        'PROFIT_PCT': [profit_pct],
        'CLOSE_REASON': [close_reason],
        'DURATION': [duration]
    }
    
    # Si archivo existe, append
    if os.path.exists(trades_file):
        existing_df = pd.read_excel(trades_file)
        new_df = pd.DataFrame(trade_data)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(trade_data)
    
    # Guardar
    df.to_excel(trades_file, index=False)
```

---

Continúo en el siguiente mensaje con las Partes 3, 4, 5, 6, 7 y 8...


# PARTE 3: CONFIGURACIÓN

## 11. settings.py - Configuración Central

### 11.1 Estructura Completa

```python
# config/settings.py

from datetime import datetime
from zoneinfo import ZoneInfo

# ════════════════════════════════════════════════════════════════
# EXCHANGE SETTINGS
# ════════════════════════════════════════════════════════════════

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_MODE = "crossed"
MARGIN_COIN = "USDT"

# ════════════════════════════════════════════════════════════════
# GENERAL BOT SETTINGS
# ════════════════════════════════════════════════════════════════

HOUR_ZONE = ZoneInfo('UTC')
CHECK_INTERVAL = 10  # Segundos entre checks de TP/SL
USE_HARDCODED_SIGNALS = False
DISPLAY_MODE = "summary"

# Colores para logs
COLOR_BLUE = '\033[1;94m'
COLOR_CYAN = '\033[1;96m'
COLOR_WHITE = '\033[1;97m'
COLOR_RESET = '\033[0m'

# ════════════════════════════════════════════════════════════════
# ACCOUNT-SPECIFIC SETTINGS
# ════════════════════════════════════════════════════════════════

ACCOUNTS = {
    "00": {
        "initial_capital": 3671,
        "dashboard_port": 5000,
        "color": COLOR_BLUE,
        "description": "Main Account"
    },
    "E1": {
        "initial_capital": 1761,
        "dashboard_port": 5001,
        "color": COLOR_CYAN,
        "description": "Elite Account"
    },
    "01": {
        "initial_capital": 117,
        "dashboard_port": 5099,
        "color": COLOR_WHITE,
        "description": "Testing Account"
    }
}

# ════════════════════════════════════════════════════════════════
# STRATEGY ASSIGNMENT PER ACCOUNT
# ════════════════════════════════════════════════════════════════

ACCOUNT_STRATEGIES = {
    "00": [
        '01_double_top_long_4H',
        '02_reversal_long_4H',
        '03_parity_long_4H',
        '04_reversal_short_4H',
        '05_parity_short_4H',
        '06_reversal_long_1H',
        '07_reversal_short_1H',
        '08_reversal_long_6Hutc',
        '09_reversal_short_6Hutc',
        '10_parity_long_1H',
        '11_parity_short_1H',
        '12_parity_long_6Hutc',
        '13_orderblocks_short_4H',
        '14_orderblocks_long_4H'
    ],
    "E1": [
        '01_double_top_long_4H',
        '02_reversal_long_4H',
        '03_parity_long_4H',
        '04_reversal_short_4H',
        '06_reversal_long_1H',
        '07_reversal_short_1H',
        '08_reversal_long_6Hutc',
        '09_reversal_short_6Hutc',
        '10_parity_long_1H',
        '11_parity_short_1H',
        '13_orderblocks_short_4H'
    ],
    "01": [
        '01_double_top_long_4H',
        '02_reversal_long_5m'
    ]
}

# ════════════════════════════════════════════════════════════════
# VALIDATION SETTINGS
# ════════════════════════════════════════════════════════════════

MIN_ORDER_AMOUNT = 40
MAX_ORDER_AMOUNT = 100
MIN_TP_PCT = 1.5
MAX_TP_PCT = 10
MIN_SL_PCT = 1.5
MAX_SL_PCT = 10
MIN_CANDLES = 49
MAX_CANDLES = 51
VALID_TIMEFRAMES = ['1H', '4H', '6Hutc', '2m', '5m', '15m', '30m']

# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def get_account_config(account_number):
    """Obtiene configuración de una cuenta específica."""
    if account_number not in ACCOUNTS:
        raise ValueError(f"Unknown account: {account_number}")
    
    base_config = ACCOUNTS[account_number]
    
    # Añadir paths
    base_dir = f'persistence/bot_files_{account_number}'
    
    return {
        **base_config,
        'base_dir': base_dir,
        'log_file': f'{base_dir}/BOT_orchestator_{account_number}.log',
        'state_file': f'{base_dir}/bot_state_{account_number}.json',
        'trades_file': f'{base_dir}/bot_trades_{account_number}.xlsx'
    }

def get_account_strategies(account_number):
    """Obtiene estrategias asignadas a una cuenta."""
    if account_number not in ACCOUNT_STRATEGIES:
        raise ValueError(f"No strategies configured for account: {account_number}")
    
    return ACCOUNT_STRATEGIES[account_number]
```

---

## 12. strategies.yaml - Definiciones

El archivo completo tiene ~400 líneas con 14 estrategias. Aquí un ejemplo completo de 3 estrategias:

```yaml
strategies:
  # ══════════════════════════════════════════════════════════════
  # 4H TIMEFRAME STRATEGIES
  # ══════════════════════════════════════════════════════════════
  
  # ────────────────────────────────────────────────────────────
  # STRATEGY 01: Double Top Long (4H)
  # ────────────────────────────────────────────────────────────
  - id: '01_double_top_long_4H'
    name: 'double_top_long_4H'
    timeframe: '4H'
    active: true
    direction: 'long'
    sell_after_ncandles: 50
    order_amount: 40
    
    # Parámetros específicos de double_top
    lookback: 2
    tolerance: 15
    trend_th: 5
    
    # TP/SL
    tp_pct: 4
    sl_pct: 10
  
  # ────────────────────────────────────────────────────────────
  # STRATEGY 02: Reversal Long (4H)
  # ────────────────────────────────────────────────────────────
  - id: '02_reversal_long_4H'
    name: 'reversal_long_4H'
    timeframe: '4H'
    active: true
    direction: 'long'
    sell_after_ncandles: 50
    order_amount: 40
    
    # Parámetros específicos de reversal
    lookback: 4
    tolerance: 20
    ma_period: 50
    
    # TP/SL
    tp_pct: 3
    sl_pct: 10
  
  # ────────────────────────────────────────────────────────────
  # STRATEGY 03: Parity Long (4H)
  # ────────────────────────────────────────────────────────────
  - id: '03_parity_long_4H'
    name: 'parity_long_4H'
    timeframe: '4H'
    active: true
    direction: 'long'
    sell_after_ncandles: 50
    order_amount: 40
    
    # Parámetros específicos de parity
    lookback: 150
    tolerance: 40
    ma_period: 50
    
    # TP/SL
    tp_pct: 3
    sl_pct: 10
  
  # ══════════════════════════════════════════════════════════════
  # 1H TIMEFRAME STRATEGIES
  # ══════════════════════════════════════════════════════════════
  
  # ... más estrategias ...
```

---

## 13. registry.py - Implementaciones

Ya cubierto en sección 6.3.

---

## 14. Cuentas de Trading

### 14.1 Cuenta 00 (Principal)

**Características:**
- Capital: 3671 USDT
- Puerto: 5000
- CPU Core: 0
- Estrategias: 14 (todas las validadas)
- Uso: Producción principal

**Estrategias asignadas:**
```
01-05: Estrategias 4H (5 estrategias)
06-07: Reversal 1H (2 estrategias)
08-09: Reversal 6Hutc (2 estrategias)
10-12: Parity 1H/6Hutc (3 estrategias)
13-14: Order Blocks 4H (2 estrategias)
```

### 14.2 Cuenta E1 (Elite)

**Características:**
- Capital: 1761 USDT
- Puerto: 5001
- CPU Core: 1
- Estrategias: 11 (subset optimizado)
- Uso: Estrategias de mejor rendimiento

**Excluidas:**
- 05_parity_short_4H (bajo rendimiento)
- 12_parity_long_6Hutc (bajo rendimiento)
- 14_orderblocks_long_4H (en evaluación)

### 14.3 Cuenta 01 (Testing)

**Características:**
- Capital: 117 USDT
- Puerto: 5099
- CPU Core: Ninguno (usa todos)
- Estrategias: 2 (experimentales)
- Uso: Testing de nuevas estrategias

---

# PARTE 4: LÓGICA DE NEGOCIO

## 15. Detección de Señales

### 15.1 Workflow Completo

```
Nueva vela cerrada (ej: 4H)
  │
  ├─→ Obtener estrategias de ese timeframe
  │
  └─→ Para cada estrategia:
         │
         ├─→ ¿Tiene posiciones abiertas?
         │   ├─ SÍ: Skip (no buscar nuevas)
         │   └─ NO: ↓
         │
         ├─→ Obtener símbolos para esta estrategia
         │      (ej: BTCUSDT, ETHUSDT, BNBUSDT...)
         │
         └─→ Para cada símbolo:
                │
                ├─→ Descargar OHLCV (ccxt)
                │      └─ fetch_ohlcv(symbol, '4H', limit=200)
                │
                ├─→ Convertir a arrays numpy
                │      └─ {open: [...], high: [...], ...}
                │
                ├─→ Ejecutar función de señal
                │      └─ double_top_long(ohlcv, lookback=2, ...)
                │         └─ Returns: [0, 0, 0, 1] (señal en última)
                │
                └─→ ¿Señal positiva? (signals[-1] != 0)
                      ├─ SÍ: Abrir posición
                      └─ NO: Continuar
```

### 15.2 Código de Detección

```python
def detect_signal_for_strategy(strategy, symbols, exchange):
    """
    Detecta señales para una estrategia.
    
    Returns:
        Lista de símbolos con señal detectada
    """
    processor = StrategyProcessor(use_hardcoded=USE_HARDCODED_SIGNALS)
    
    detected_symbols = processor.detect_signals(
        strategy=strategy,
        symbols=symbols,
        exchange=exchange
    )
    
    return detected_symbols

def process_strategy(strat, final_symbols, exchange, open_positions,
                     strategy_candles, state_file, send_request_func,
                     get_balance_func):
    """
    Procesa una estrategia: detecta señales y abre posiciones.
    """
    strat_id = strat['id']
    
    logger.info(f"Processing strategy: {strat_id}")
    
    # Detectar señales
    detected = detect_signal_for_strategy(strat, final_symbols, exchange)
    
    logger.info(f"Signals detected: {len(detected)}")
    
    if not detected:
        return
    
    # Para cada señal
    for symbol in detected:
        # Verificar balance
        balance = get_balance_func()
        
        if balance < strat['order_amount']:
            logger.warning(
                f"Insufficient balance: {balance} < {strat['order_amount']}"
            )
            continue
        
        # Abrir posición
        try:
            success = open_position_for_signal(
                strategy=strat,
                symbol=symbol,
                exchange=exchange,
                open_positions=open_positions,
                strategy_candles=strategy_candles,
                state_file=state_file,
                send_request_func=send_request_func,
                get_balance_func=get_balance_func
            )
            
            if success:
                logger.info(f"Position opened: {symbol}")
                
                # Solo abrir una posición por estrategia
                break
        
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            continue
```

---

## 16. Gestión de Posiciones

### 16.1 Apertura de Posición

```python
def open_position_for_signal(strategy, symbol, exchange, open_positions,
                              strategy_candles, state_file, send_request_func,
                              get_balance_func):
    """
    Abre una posición para una señal detectada.
    
    Workflow:
    1. Obtener precio actual
    2. Calcular size
    3. Calcular TP/SL
    4. Enviar orden
    5. Registrar posición
    6. Guardar estado
    """
    # 1. Precio actual
    current_price = get_current_price(symbol)
    
    # 2. Calcular size
    size = calculate_position_size(
        symbol,
        strategy['order_amount'],
        current_price
    )
    
    # 3. Calcular TP/SL
    if strategy['direction'] == 'long':
        tp = current_price * (1 + strategy['tp_pct'] / 100)
        sl = current_price * (1 - strategy['sl_pct'] / 100)
    else:  # short
        tp = current_price * (1 - strategy['tp_pct'] / 100)
        sl = current_price * (1 + strategy['sl_pct'] / 100)
    
    # 4. Enviar orden
    logger.info(
        f"Placing order: {symbol} {strategy['direction']} "
        f"@ {current_price} - Size: {size}"
    )
    
    order_response = send_request_func(
        symbol=symbol,
        side='buy' if strategy['direction'] == 'long' else 'sell',
        size=size
    )
    
    # 5. Parsear respuesta
    order_id = order_response['orderId']
    fill_price = float(order_response['priceAvg'])
    fill_size = float(order_response['size'])
    
    logger.info(f"Order filled: {symbol} @ {fill_price}")
    
    # 6. Registrar posición
    position = {
        'symbol': symbol,
        'size': str(fill_size),
        'entry_price': str(fill_price),
        'direction': strategy['direction'],
        'tp': str(tp),
        'sl': str(sl),
        'order_id': order_id,
        'opened_at': datetime.now(timezone.utc).isoformat(),
        'usdt_amount': strategy['order_amount']
    }
    
    strat_id = strategy['id']
    
    if strat_id not in open_positions:
        open_positions[strat_id] = []
    
    open_positions[strat_id].append(position)
    
    # 7. Inicializar contador de velas
    strategy_candles[strat_id] = 0
    
    # 8. Guardar estado
    save_state(open_positions, strategy_candles, state_file)
    
    logger.info(f"Position tracked: {symbol}")
    
    return True

def calculate_position_size(symbol, order_amount, current_price):
    """
    Calcula el size de la posición.
    
    size = order_amount / current_price
    
    Ajustado a la precisión del símbolo.
    """
    size = order_amount / current_price
    
    # Redondear a 4 decimales (típico en futuros)
    size = round(size, 4)
    
    return size
```

---

## 17. Take Profit y Stop Loss

### 17.1 Verificación TP/SL

```python
def check_tp_sl_for_strategy(strategy_id, positions, send_request_func):
    """
    Verifica TP/SL para todas las posiciones de una estrategia.
    
    Returns:
        True si se cerró alguna posición
    """
    if not positions:
        return False
    
    closed_any = False
    
    for pos in positions[:]:  # Iterar sobre copia
        symbol = pos['symbol']
        current_price = get_current_price(symbol)
        
        should_close = False
        close_reason = None
        
        entry = float(pos['entry_price'])
        tp = float(pos['tp'])
        sl = float(pos['sl'])
        
        if pos['direction'] == 'long':
            # TP hit
            if current_price >= tp:
                should_close = True
                close_reason = 'TP'
            
            # SL hit
            elif current_price <= sl:
                should_close = True
                close_reason = 'SL'
        
        else:  # short
            # TP hit
            if current_price <= tp:
                should_close = True
                close_reason = 'TP'
            
            # SL hit
            elif current_price >= sl:
                should_close = True
                close_reason = 'SL'
        
        if should_close:
            # Calcular profit
            if pos['direction'] == 'long':
                profit_pct = ((current_price - entry) / entry) * 100
            else:
                profit_pct = ((entry - current_price) / entry) * 100
            
            profit_usd = (profit_pct / 100) * pos['usdt_amount']
            
            # Cerrar posición
            try:
                close_position(
                    pos,
                    send_request_func,
                    close_reason
                )
                
                # Log trade
                log_trade_to_excel(
                    symbol=symbol,
                    profit=profit_usd,
                    profit_pct=profit_pct,
                    close_reason=close_reason,
                    strategy_id=strategy_id,
                    entry_price=entry,
                    exit_price=current_price,
                    size=float(pos['size']),
                    duration=0,  # calculado desde opened_at
                    direction=pos['direction']
                )
                
                # Remover de tracking
                positions.remove(pos)
                
                closed_any = True
                
                logger.info(
                    f"Position closed: {symbol} - "
                    f"Reason: {close_reason} - "
                    f"Profit: {profit_usd:.2f}$ ({profit_pct:.2f}%)"
                )
            
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
    
    return closed_any

def close_position(position, send_request_func, close_reason):
    """
    Cierra una posición.
    """
    symbol = position['symbol']
    size = float(position['size'])
    direction = position['direction']
    
    # Orden de cierre
    # Si es LONG → sell para cerrar
    # Si es SHORT → buy para cerrar
    side = 'sell' if direction == 'long' else 'buy'
    
    response = send_request_func(
        symbol=symbol,
        side=side,
        size=size
    )
    
    logger.info(f"Position closed: {symbol} ({close_reason})")
```

---

## 18. Timeout por Velas

### 18.1 Incremento de Contador

```python
def increment_strategy_candles(strategy_id, strategy_candles, open_positions,
                                state_file):
    """
    Incrementa el contador de velas para una estrategia.
    
    Solo incrementa si hay posiciones abiertas.
    """
    # Verificar si hay posiciones
    if strategy_id not in open_positions or not open_positions[strategy_id]:
        return
    
    # Incrementar
    if strategy_id not in strategy_candles:
        strategy_candles[strategy_id] = 0
    
    strategy_candles[strategy_id] += 1
    
    # Guardar
    save_state(open_positions, strategy_candles, state_file)
    
    logger.debug(
        f"Strategy {strategy_id} candles: {strategy_candles[strategy_id]}"
    )
```

### 18.2 Verificación de Timeout

```python
def check_candles_timeout_for_strategy(strategy_id, max_candles, open_positions,
                                       strategy_candles, state_file,
                                       send_request_func):
    """
    Verifica timeout por velas para una estrategia.
    
    Si candles >= max_candles:
    - Cerrar todas las posiciones
    - Reset contador a 0
    """
    current_candles = strategy_candles.get(strategy_id, 0)
    
    if current_candles < max_candles:
        return
    
    positions = open_positions.get(strategy_id, [])
    
    if not positions:
        # Sin posiciones pero contador alto → reset
        strategy_candles[strategy_id] = 0
        save_state(open_positions, strategy_candles, state_file)
        return
    
    logger.warning(
        f"TIMEOUT for {strategy_id}: {current_candles}/{max_candles} candles"
    )
    
    # Cerrar todas las posiciones
    for pos in positions[:]:
        try:
            current_price = get_current_price(pos['symbol'])
            
            # Calcular profit
            entry = float(pos['entry_price'])
            if pos['direction'] == 'long':
                profit_pct = ((current_price - entry) / entry) * 100
            else:
                profit_pct = ((entry - current_price) / entry) * 100
            
            profit_usd = (profit_pct / 100) * pos['usdt_amount']
            
            # Cerrar
            close_position(pos, send_request_func, 'TIMEOUT')
            
            # Log
            log_trade_to_excel(
                symbol=pos['symbol'],
                profit=profit_usd,
                profit_pct=profit_pct,
                close_reason='TIMEOUT',
                strategy_id=strategy_id,
                entry_price=entry,
                exit_price=current_price,
                size=float(pos['size']),
                duration=current_candles,
                direction=pos['direction']
            )
            
            # Remover
            positions.remove(pos)
            
            logger.info(
                f"Position closed (TIMEOUT): {pos['symbol']} - "
                f"Profit: {profit_usd:.2f}$ ({profit_pct:.2f}%)"
            )
        
        except Exception as e:
            logger.error(f"Error closing position (TIMEOUT): {e}")
    
    # Reset contador
    strategy_candles[strategy_id] = 0
    
    # Guardar
    save_state(open_positions, strategy_candles, state_file)
    
    logger.info(f"All positions closed for {strategy_id} (TIMEOUT)")
```

---

## 19. Sincronización con Broker

### 19.1 Propósito

Reconciliar estado local (JSON) con estado real (Bitget):

- Detectar posiciones cerradas externamente (usuario las cerró en Bitget)
- Detectar posiciones cerradas por liquidación
- Actualizar tracking con estado real

### 19.2 Código

```python
def sync_broker(open_positions, strategy_candles, state_file, bitget_client):
    """
    Sincroniza posiciones locales con Bitget.
    
    Workflow:
    1. Obtener posiciones reales de Bitget
    2. Para cada posición local:
       - Si NO existe en Bitget → cerrada externamente
       - Remover de tracking local
    3. Guardar estado si hubo cambios
    """
    logger.debug("Syncing with broker...")
    
    # 1. Obtener posiciones reales
    real_positions = bitget_client.get_all_positions('USDT-FUTURES')
    
    # Crear set de (symbol, direction) de posiciones reales
    real_pos_set = set()
    for pos in real_positions:
        if float(pos['total']) > 0:
            symbol = pos['symbol']
            side = pos['holdSide']  # 'long' o 'short'
            real_pos_set.add((symbol, side))
    
    # 2. Verificar posiciones locales
    changes_made = False
    
    for strategy_id, positions in open_positions.items():
        for pos in positions[:]:  # Iterar sobre copia
            key = (pos['symbol'], pos['direction'])
            
            if key not in real_pos_set:
                # Posición cerrada externamente
                logger.warning(
                    f"Position {pos['symbol']} {pos['direction']} "
                    f"closed externally (not in Bitget)"
                )
                
                # Remover de tracking
                positions.remove(pos)
                
                # Reset contador de velas
                strategy_candles[strategy_id] = 0
                
                changes_made = True
    
    # 3. Guardar si hubo cambios
    if changes_made:
        save_state(open_positions, strategy_candles, state_file)
        logger.info("State synchronized with broker")
```

---

# PARTE 5: FLUJOS DE EJECUCIÓN

## 20. Ciclo de Vida del Bot

```
┌──────────────────────────────────────────────────────────┐
│  1. INICIO                                               │
│     • python3 main.py --account 00                       │
│     • Parse argumentos                                   │
│     • Seleccionar cliente Bitget                         │
│     • Crear BotOrchestrator                              │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  2. INICIALIZACIÓN (orchestrator.run())                  │
│     ├─ Cargar estrategias desde YAML                     │
│     ├─ Aplicar filtro --set-active                       │
│     ├─ Validar configuración                             │
│     ├─ Cargar símbolos por estrategia                    │
│     ├─ Agrupar por timeframe                             │
│     ├─ Inicializar dashboard Flask                       │
│     ├─ Cargar estado previo (JSON)                       │
│     ├─ Sync con broker                                   │
│     └─ Calcular próximas velas                           │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  3. MAIN LOOP (infinito)                                 │
│     while True:                                          │
│        ├─ Detectar velas cerradas                        │
│        │  └─ Procesar estrategias de esos timeframes     │
│        │                                                  │
│        ├─ Cada 10s: Verificar TP/SL                      │
│        │  └─ Para todas las posiciones abiertas          │
│        │                                                  │
│        └─ Sleep 0.05s (evitar CPU spin)                  │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│  4. SHUTDOWN (Ctrl+C)                                    │
│     • Capturar señal de interrupción                     │
│     • Guardar estado final                               │
│     • Cerrar conexiones                                  │
│     • Exit                                               │
└──────────────────────────────────────────────────────────┘
```

---

## 21. Flujo de Detección de Velas

```
Timeframe: 4H
Hora actual: 2026-01-04 20:00:00 UTC
Próxima vela: 2026-01-04 20:00:00 UTC

┌──────────────────────────────────────────────────────────┐
│  DETECCIÓN                                               │
└───────────────────┬──────────────────────────────────────┘
                    │
          now >= next_candle_time?
                    │
        ┌───────────┴───────────┐
        │                       │
       SÍ                      NO
        │                       │
        ▼                       └─→ Continuar loop
  ┌──────────────┐
  │ 4H CERRADA!  │
  └──────┬───────┘
         │
         ├─→ Log: "New 4H candle closed"
         │
         ├─→ Sync con broker
         │
         ├─→ Para cada estrategia 4H:
         │      ├─ 01_double_top_long_4H
         │      ├─ 02_reversal_long_4H
         │      ├─ 03_parity_long_4H
         │      └─ ...
         │
         └─→ Recalcular next_candle_time
               └─ next_candle_time = 2026-01-05 00:00:00
```

---

## 22. Flujo de Apertura de Posición

```
Señal detectada: BTCUSDT, estrategia 01_double_top_long_4H

┌──────────────────────────────────────────────────────────┐
│  1. VERIFICACIONES                                       │
└───────────────────┬──────────────────────────────────────┘
                    │
             ┌──────▼──────┐
             │ Balance OK? │
             └──────┬──────┘
                    │
        ┌───────────┴───────────┐
        │                       │
       SÍ                      NO
        │                       │
        ▼                       └─→ Log warning + Skip
┌───────────────────┐
│ 2. CÁLCULOS       │
└────────┬──────────┘
         │
         ├─→ Precio actual: 91167.7
         ├─→ Size: 40 / 91167.7 = 0.0004
         ├─→ TP: 91167.7 * 1.04 = 94814.4
         └─→ SL: 91167.7 * 0.90 = 82051.0
         
┌────────▼──────────┐
│ 3. ORDEN MARKET   │
└────────┬──────────┘
         │
         ├─→ Bitget API: place_order()
         ├─→ Response: orderId, priceAvg, size
         │
┌────────▼──────────┐
│ 4. TRACKING       │
└────────┬──────────┘
         │
         ├─→ Crear position dict
         ├─→ Añadir a OPEN_POSITIONS[strat_id]
         ├─→ Init STRATEGY_CANDLES[strat_id] = 0
         ├─→ Guardar estado (JSON)
         └─→ Log: "Position opened"
```

---

## 23. Flujo de Cierre de Posición

```
TP/SL Check (cada 10s)

Posición: BTCUSDT LONG
Entry: 91167.7
TP: 94814.4
SL: 82051.0
Current: 94850.0

┌──────────────────────────────────────────────────────────┐
│  1. VERIFICACIÓN                                         │
└───────────────────┬──────────────────────────────────────┘
                    │
            ┌───────▼───────┐
            │ current >= TP?│
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        │                       │
       SÍ                      NO
        │                       │
        │                  ┌────▼────┐
        │                  │SL hit?  │
        │                  └────┬────┘
        │                       │
        │           ┌───────────┴───────────┐
        │           │                       │
        │          SÍ                      NO
        │           │                       │
        │           │                       └─→ Continuar
        │           │
        ▼           ▼
┌────────────────────────────┐
│ 2. CALCULAR PROFIT         │
└──────────┬─────────────────┘
           │
           ├─→ profit_pct = (94850 - 91167.7) / 91167.7 * 100 = 4.04%
           └─→ profit_usd = 0.0404 * 40 = 1.62$
           
┌──────────▼─────────────────┐
│ 3. CERRAR ORDEN            │
└──────────┬─────────────────┘
           │
           ├─→ Bitget API: close_position()
           ├─→ Side: sell (para cerrar long)
           │
┌──────────▼─────────────────┐
│ 4. LOG Y ACTUALIZAR        │
└──────────┬─────────────────┘
           │
           ├─→ log_trade_to_excel()
           ├─→ Remover de OPEN_POSITIONS
           ├─→ Guardar estado
           └─→ Log: "Position closed (TP) - Profit: 1.62$ (4.04%)"
```

---

## 24. Main Loop Explicado

```python
# Pseudocódigo del main loop

last_tpsl_check = time.time()
CHECK_INTERVAL = 10  # segundos

while True:
    now = datetime.now(HOUR_ZONE)
    
    # ═══════════════════════════════════════════════════════════
    # CHECK 1: ¿Vela cerrada?
    # ═══════════════════════════════════════════════════════════
    
    closed_timeframes = []
    
    for tf in unique_timeframes:  # ['4H', '1H', '6Hutc']
        if now >= next_candle_times[tf]:
            closed_timeframes.append(tf)
    
    if closed_timeframes:
        for tf in closed_timeframes:
            # Procesar estrategias de este TF
            strategies = strategies_by_timeframe[tf]
            
            # Sync con broker
            sync_broker(...)
            
            for strat in strategies:
                strat_id = strat['id']
                
                # ¿Tiene posiciones?
                num_pos = len(OPEN_POSITIONS.get(strat_id, []))
                
                if num_pos > 0:
                    # Skip nuevas señales
                    candles = STRATEGY_CANDLES[strat_id]
                    logger.info(f"Skip {strat_id} - {candles}/50 candles")
                    
                    # Incrementar candles
                    increment_strategy_candles(...)
                    
                    # Check timeout
                    check_candles_timeout_for_strategy(...)
                
                else:
                    # Buscar señales
                    process_strategy(
                        strat=strat,
                        symbols=symbols_by_strategy[strat_id],
                        exchange=exchange,
                        ...
                    )
            
            # Recalcular próxima vela
            next_candle_times[tf] = calculate_next_candle_time(tf, HOUR_ZONE)
    
    # ═══════════════════════════════════════════════════════════
    # CHECK 2: TP/SL periódico
    # ═══════════════════════════════════════════════════════════
    
    current_time = time.time()
    
    if current_time - last_tpsl_check >= CHECK_INTERVAL:
        # Verificar TP/SL para todas las estrategias con posiciones
        for strat in strategies:
            strat_id = strat['id']
            positions = OPEN_POSITIONS.get(strat_id, [])
            
            if positions:
                closed = check_tp_sl_for_strategy(
                    strat_id,
                    positions,
                    bitget_client
                )
                
                if closed:
                    save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        
        last_tpsl_check = current_time
    
    # ═══════════════════════════════════════════════════════════
    # Sleep para evitar spin de CPU
    # ═══════════════════════════════════════════════════════════
    
    time.sleep(0.05)  # 50ms
```

---

Continuaré con las Partes 6, 7 y 8 en el siguiente bloque...


# PARTE 6: INTEGRACIONES

## 25. Bitget API en Detalle

### 25.1 Autenticación HMAC

```python
def _sign_request(timestamp, method, request_path, body=''):
    """
    Firma petición con HMAC SHA256.
    
    Algoritmo:
    1. Construir mensaje: timestamp + method + path + body
    2. Firmar con HMAC-SHA256 usando API secret
    3. Codificar en Base64
    
    Ejemplo:
    timestamp = "1704398400000"
    method = "POST"
    request_path = "/api/v2/mix/order/place"
    body = '{"symbol":"BTCUSDT","size":"0.001",...}'
    
    message = "1704398400000POST/api/v2/mix/order/place{...}"
    signature = HMAC-SHA256(message, secret)
    b64_signature = Base64(signature)
    """
    message = timestamp + method + request_path + body
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    return base64.b64encode(signature).decode()
```

### 25.2 Headers Requeridos

```python
headers = {
    'ACCESS-KEY': api_key,                  # API key
    'ACCESS-SIGN': signature,               # Firma HMAC
    'ACCESS-TIMESTAMP': timestamp,          # Milisegundos
    'ACCESS-PASSPHRASE': api_passphrase,    # Passphrase
    'Content-Type': 'application/json',
    'locale': 'en-US'
}
```

### 25.3 Endpoints Principales

**Colocar Orden:**
```
POST /api/v2/mix/order/place

Body:
{
    "symbol": "BTCUSDT",
    "productType": "USDT-FUTURES",
    "marginMode": "crossed",
    "marginCoin": "USDT",
    "size": "0.001",
    "side": "buy",           # buy o sell
    "tradeSide": "open",     # open o close
    "orderType": "market"    # market o limit
}

Response:
{
    "code": "00000",
    "msg": "success",
    "data": {
        "orderId": "1391784175051902977",
        "clientOid": "...",
        "priceAvg": "91167.7",
        "size": "0.001"
    }
}
```

**Obtener Posiciones:**
```
GET /api/v2/mix/position/all-position?productType=USDT-FUTURES

Response:
{
    "code": "00000",
    "data": [
        {
            "symbol": "BTCUSDT",
            "marginCoin": "USDT",
            "holdSide": "long",
            "total": "0.001",
            "available": "0.001",
            "openPriceAvg": "91167.7",
            "unrealizedPL": "1.25"
        }
    ]
}
```

**Obtener Balance:**
```
GET /api/v2/mix/account/account?productType=USDT-FUTURES&marginCoin=USDT

Response:
{
    "code": "00000",
    "data": {
        "marginCoin": "USDT",
        "available": "1750.50",
        "locked": "240.00",
        "maxOpenPosAvailable": "1750.50"
    }
}
```

---

## 26. WebSocket vs REST

### 26.1 Cuándo Usar Cada Uno

| Operación | REST | WebSocket | Razón |
|-----------|------|-----------|-------|
| Colocar orden | ✅ | ❌ | Operación única, REST es suficiente |
| Cerrar posición | ✅ | ❌ | Operación única |
| Obtener balance | ✅ | ✅ | WebSocket más eficiente |
| Obtener posiciones | ✅ | ✅ | WebSocket para updates en tiempo real |
| OHLCV histórico | ✅ | ❌ | REST más simple |
| Precio actual | ❌ | ✅ | WebSocket evita rate limits |

### 26.2 WebSocket para Datos en Tiempo Real

```python
# WebSocket manager (simplificado)

class WebSocketManager:
    def __init__(self, api_key, api_secret, api_passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.ws = None
    
    def connect(self):
        """Conecta a WebSocket público y privado."""
        # Público: market data
        self.ws_public = websocket.create_connection(
            "wss://ws.bitget.com/v2/ws/public"
        )
        
        # Privado: account data (requiere auth)
        self.ws_private = websocket.create_connection(
            "wss://ws.bitget.com/v2/ws/private"
        )
        
        # Autenticar WS privado
        self._authenticate_private()
    
    def _authenticate_private(self):
        """Autentica WebSocket privado."""
        timestamp = str(int(time.time()))
        sign = self._sign_request(timestamp, 'GET', '/user/verify', '')
        
        auth_msg = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.api_passphrase,
                "timestamp": timestamp,
                "sign": sign
            }]
        }
        
        self.ws_private.send(json.dumps(auth_msg))
    
    def subscribe_positions(self):
        """Suscribe a updates de posiciones."""
        sub_msg = {
            "op": "subscribe",
            "args": [{
                "instType": "USDT-FUTURES",
                "channel": "positions",
                "instId": "default"
            }]
        }
        
        self.ws_private.send(json.dumps(sub_msg))
```

---

## 27. Manejo de Errores

### 27.1 Códigos de Error Comunes

| Código | Significado | Acción |
|--------|-------------|--------|
| `00000` | Success | ✅ OK |
| `40005` | Invalid API key | Verificar credenciales |
| `40006` | Invalid timestamp | Sincronizar reloj del sistema |
| `40007` | Invalid signature | Verificar algoritmo de firma |
| `40014` | Insufficient balance | Skip orden |
| `40015` | Invalid symbol | Validar símbolo |
| `40400` | Rate limit exceeded | Esperar y reintentar |
| `43025` | Position not exist | Posición ya cerrada |

### 27.2 Retry Logic

```python
def execute_with_retry(func, max_retries=3, delay=2):
    """
    Ejecuta función con retry automático.
    
    Args:
        func: Función a ejecutar
        max_retries: Máximo de intentos
        delay: Segundos entre intentos
    
    Returns:
        Resultado de la función
    
    Raises:
        Exception si falla tras todos los retries
    """
    for attempt in range(max_retries):
        try:
            result = func()
            
            # Verificar respuesta de Bitget
            if isinstance(result, dict) and result.get('code') == '00000':
                return result
            
            # Error en respuesta
            error_msg = result.get('msg', 'Unknown error')
            
            # Errores no recuperables
            if result.get('code') in ['40005', '40007', '40014', '40015']:
                raise Exception(f"Non-recoverable error: {error_msg}")
            
            # Retry para otros errores
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}"
                )
                time.sleep(delay * (2 ** attempt))  # Backoff exponencial
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts: {error_msg}")
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"Network error, retrying: {e}")
                time.sleep(delay)
                continue
            else:
                raise
    
    raise Exception("Retry logic failed")
```

---

## 28. Rate Limits y Retry

### 28.1 Rate Limits de Bitget

| Endpoint | Límite | Ventana |
|----------|--------|---------|
| `/order/place` | 10 req/s | 1 segundo |
| `/position/all-position` | 20 req/s | 1 segundo |
| `/account/account` | 5 req/s | 1 segundo |
| `/market/candles` | 20 req/s | 1 segundo |

### 28.2 Estrategias para Evitar Rate Limits

**1. Batch requests cuando sea posible**
```python
# En lugar de:
for symbol in symbols:
    get_position(symbol)  # N requests

# Hacer:
all_positions = get_all_positions()  # 1 request
filter_positions(all_positions, symbols)
```

**2. Caché de datos que no cambian frecuentemente**
```python
# Caché de símbolos disponibles (no cambian frecuentemente)
@lru_cache(maxsize=1)
def get_all_symbols():
    return bitget_client.get_all_symbols()

# Expirar caché cada hora
cache_expiry = 3600
```

**3. Espaciar requests en loops**
```python
for symbol in symbols:
    detect_signal(symbol)
    time.sleep(0.1)  # 100ms entre símbolos
```

---

# PARTE 7: DESARROLLO

## 29. Añadir Nueva Estrategia

### 29.1 Paso a Paso

**PASO 1: Crear función de señal**

```bash
nano signals/Z_add_signals_mi_estrategia.py
```

```python
import numpy as np

def mi_estrategia_long(ohlcv_array, lookback=10, threshold=5, live_trading=False):
    """
    Mi nueva estrategia para long.
    
    Args:
        ohlcv_array: Dict con arrays numpy
        lookback: Velas hacia atrás
        threshold: Threshold de señal
        live_trading: Solo última señal
    
    Returns:
        Array de señales
    """
    close = ohlcv_array['close']
    
    signals = np.zeros(len(close))
    
    for i in range(lookback, len(close)):
        # IMPLEMENTAR LÓGICA AQUÍ
        # Ejemplo: señal si precio sube más del threshold%
        price_change = (close[i] - close[i-lookback]) / close[i-lookback] * 100
        
        if price_change > threshold:
            signals[i] = 1
    
    if live_trading:
        return signals[-1:]
    
    return signals
```

**PASO 2: Registrar en registry.py**

```bash
nano strategies/registry.py
```

```python
# Añadir import
from signals.Z_add_signals_mi_estrategia import mi_estrategia_long

# Añadir al mapeo
STRATEGY_FUNCTIONS = {
    # ... estrategias existentes ...
    
    # Nueva estrategia
    'mi_estrategia_long_4H': mi_estrategia_long,
}

# Actualizar set
IMPLEMENTED_STRATEGIES = set(STRATEGY_FUNCTIONS.keys())
```

**PASO 3: Definir en YAML**

```bash
nano strategies/strategies.yaml
```

```yaml
strategies:
  # ... estrategias existentes ...
  
  # ────────────────────────────────────────────────────────────
  # STRATEGY 15: Mi Estrategia Long (4H)
  # ────────────────────────────────────────────────────────────
  - id: '15_mi_estrategia_long_4H'
    name: 'mi_estrategia_long_4H'
    timeframe: '4H'
    active: true
    direction: 'long'
    sell_after_ncandles: 50
    order_amount: 40
    
    # Parámetros específicos
    lookback: 10
    threshold: 5
    
    # TP/SL
    tp_pct: 3
    sl_pct: 10
```

**PASO 4: Asignar a cuenta de testing**

```bash
nano config/settings.py
```

```python
ACCOUNT_STRATEGIES = {
    "00": [
        # ... estrategias existentes ...
    ],
    "E1": [
        # ... estrategias existentes ...
    ],
    "01": [
        '01_double_top_long_4H',
        '15_mi_estrategia_long_4H'  # ← Nueva
    ]
}
```

**PASO 5: Probar**

```bash
cd ~/projects/quant/quant_g/scripts/BOT_trading
source ~/projects/quant/env_quant/bin/activate

# Lanzar en cuenta de testing
python3 main.py --account 01 --set-active 15_mi_estrategia_long_4H

# Verificar logs
tail -f persistence/bot_files_01/BOT_orchestator_01.log
```

### 29.2 Checklist de Validación

```
✅ Función creada en signals/
✅ Función retorna array de señales correcto
✅ live_trading=True retorna solo última señal
✅ Registrada en registry.py
✅ Añadida a IMPLEMENTED_STRATEGIES
✅ Definida en strategies.yaml
✅ Todos los parámetros obligatorios presentes
✅ TP/SL en rangos válidos
✅ Asignada a cuenta de testing
✅ Bot arranca sin errores
✅ Validación pasa sin errors
✅ Señales se detectan correctamente
✅ Posiciones se abren
✅ TP/SL funcionan
✅ Timeout funciona
```

---

## 30. Sistema de Logging

### 30.1 Configuración

```python
# bot_utils/logger.py

def setup_logger(log_dir, logfile_name='bot.log', 
                 console_level=logging.INFO, file_level=logging.DEBUG,
                 max_bytes=10*1024*1024, backup_count=5):
    """
    Configura logging profesional con dual output.
    
    Características:
    - Consola: Solo mensajes (para humanos)
    - Archivo: Timestamps + módulo + línea (para debugging)
    - Rotación automática (10MB, 5 backups)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, logfile_name)
    
    # Logger raíz
    root_logger = logging.getLogger('BOT_trading')
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.propagate = False
    
    # Console handler (limpio)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler (detallado + rotación)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Añadir handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger
```

### 30.2 Uso de Logger

```python
import logging

# En cada módulo
logger = logging.getLogger('BOT_trading.core')

# Diferentes niveles
logger.debug("Detalle muy específico")
logger.info("Evento importante")
logger.warning("Advertencia no crítica")
logger.error("Error que afecta funcionalidad")
logger.critical("Error grave que puede detener bot")
```

---

## 31. Testing y Validación

### 31.1 Testing de Funciones de Señal

```python
# test_signals.py

import numpy as np
from signals.Z_add_signals_double_top import double_top_long

def test_double_top_basic():
    """Test básico de double_top_long."""
    
    # Datos de prueba
    test_data = {
        'open': np.array([100, 101, 102, 98, 97, 99, 103]),
        'high': np.array([102, 103, 104, 100, 99, 101, 105]),
        'low': np.array([99, 100, 101, 97, 96, 98, 102]),
        'close': np.array([101, 102, 103, 99, 98, 100, 104]),
        'volume': np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600])
    }
    
    # Ejecutar función
    signals = double_top_long(
        test_data,
        lookback=2,
        tolerance=15,
        trend_th=5,
        live_trading=False
    )
    
    # Verificaciones
    assert len(signals) == len(test_data['close'])
    assert signals.dtype == np.float64
    assert all(s in [0, 1] for s in signals)
    
    print("✅ test_double_top_basic PASSED")

def test_live_trading_mode():
    """Test de modo live_trading."""
    
    test_data = {
        'open': np.array([100] * 10),
        'high': np.array([102] * 10),
        'low': np.array([98] * 10),
        'close': np.array([101] * 10),
        'volume': np.array([1000] * 10)
    }
    
    # Con live_trading=True
    signals = double_top_long(
        test_data,
        lookback=2,
        tolerance=15,
        trend_th=5,
        live_trading=True
    )
    
    # Debe retornar solo 1 valor
    assert len(signals) == 1
    
    print("✅ test_live_trading_mode PASSED")

if __name__ == '__main__':
    test_double_top_basic()
    test_live_trading_mode()
    print("\n✅ All tests passed!")
```

### 31.2 Validación de Configuración

```bash
# Validar YAML
python3 -c "import yaml; yaml.safe_load(open('strategies/strategies.yaml'))"

# Validar carga de estrategias
python3 -c "from strategies import load_strategies; print(load_strategies(['01_double_top_long_4H']))"

# Validar configuración de cuenta
python3 -c "from config.settings import get_account_config; print(get_account_config('00'))"
```

---

## 32. Troubleshooting

### 32.1 Bot No Arranca

**Síntoma:** Error al ejecutar `python3 main.py --account 00`

**Diagnóstico:**
```bash
# Verificar Python
python3 --version

# Verificar virtualenv
which python3
ls -la ~/projects/quant/env_quant/bin/python3

# Probar imports
python3 -c "import flask, ccxt, yaml"
```

**Soluciones:**
- Activar virtualenv: `source ~/projects/quant/env_quant/bin/activate`
- Reinstalar dependencias: `pip install -r requirements.txt`

### 32.2 Estrategia No Carga

**Síntoma:** Error "Strategy not found in YAML"

**Diagnóstico:**
```bash
# Ver IDs en YAML
grep "id:" strategies/strategies.yaml

# Verificar asignación
python3 -c "from config.settings import get_account_strategies; print(get_account_strategies('00'))"
```

**Solución:** Verificar que el ID está en YAML y en settings.py

### 32.3 Dashboard No Carga

**Síntoma:** http://localhost:5000 no responde

**Diagnóstico:**
```bash
# Verificar bot corriendo
pgrep -f "main.py --account 00"

# Verificar puerto
netstat -tlnp | grep 5000

# Probar health
curl http://localhost:5000/api/health
```

**Solución:** Verificar que el bot esté corriendo y el puerto esté libre

### 32.4 Posiciones No Cierran

**Síntoma:** Posición no cierra en TP/SL

**Diagnóstico:**
```bash
# Ver logs de TP/SL
grep "check_tp_sl" persistence/bot_files_00/BOT_orchestator_00.log | tail -20

# Ver posiciones actuales
cat persistence/bot_files_00/bot_state_00.json | jq '.positions'
```

**Solución:** Verificar que el precio actual alcanza TP/SL

### 32.5 Error de Conexión API

**Síntoma:** Error "Invalid API key"

**Diagnóstico:**
```python
# Verificar credenciales
from utils.ZZ_connect import BITGET_API_KEY_00
print(BITGET_API_KEY_00[:10])  # Primeros 10 chars
```

**Solución:** Verificar que las credenciales son correctas en `utils/ZZ_connect.py`

---

# PARTE 8: REFERENCIA

## 33. Catálogo de Estrategias

### 33.1 Resumen por Timeframe

| Timeframe | Total | Long | Short |
|-----------|-------|------|-------|
| 4H | 7 | 4 | 3 |
| 1H | 4 | 2 | 2 |
| 6Hutc | 3 | 2 | 1 |
| 2m/5m | 2 | 2 | 0 |
| **TOTAL** | **16** | **10** | **6** |

### 33.2 Listado Completo

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

## 34. Estructuras de Datos

### 34.1 Posición

```python
position = {
    'symbol': 'BTCUSDT',              # Par operado
    'size': '0.0004',                 # Tamaño de posición
    'entry_price': '91167.7',         # Precio de entrada
    'direction': 'long',              # 'long' o 'short'
    'tp': '94734.408',                # Take profit (absoluto)
    'sl': '82051.03',                 # Stop loss (absoluto)
    'order_id': '1391784175051902977', # ID en Bitget
    'opened_at': '2026-01-04T18:58:46.394725+00:00',  # Timestamp ISO
    'usdt_amount': 40.0               # USDT invertidos
}
```

### 34.2 Estrategia (dict desde YAML)

```python
strategy = {
    'id': '01_double_top_long_4H',
    'name': 'double_top_long_4H',
    'timeframe': '4H',
    'active': True,
    'direction': 'long',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 2,
    'tolerance': 15,
    'trend_th': 5,
    'tp_pct': 4.0,
    'sl_pct': 10.0
}
```

### 34.3 OHLCV Array

```python
ohlcv_array = {
    'open': np.array([91000, 91100, 91200, ...]),
    'high': np.array([91500, 91600, 91700, ...]),
    'low': np.array([90800, 90900, 91000, ...]),
    'close': np.array([91167, 91250, 91300, ...]),
    'volume': np.array([1000000, 1100000, 1200000, ...])
}
```

---

## 35. Variables de Estado

### 35.1 OPEN_POSITIONS

```python
# En memoria
OPEN_POSITIONS = {
    '01_double_top_long_4H': [
        {
            'symbol': 'BTCUSDT',
            'size': '0.0004',
            'entry_price': '91167.7',
            # ... más campos
        }
    ],
    '02_reversal_long_4H': []
}

# En JSON (bot_state_XX.json)
{
  "positions": {
    "01_double_top_long_4H": [ {...} ],
    "02_reversal_long_4H": []
  }
}
```

### 35.2 STRATEGY_CANDLES

```python
# En memoria
STRATEGY_CANDLES = {
    '01_double_top_long_4H': 5,
    '02_reversal_long_4H': 0
}

# En JSON
{
  "strategy_candles": {
    "01_double_top_long_4H": 5,
    "02_reversal_long_4H": 0
  }
}
```

---

## 36. Glosario Técnico

| Término | Definición |
|---------|------------|
| **Account** | Cuenta de trading (00, E1, 01) |
| **Active** | Estrategia habilitada (`active: true`) |
| **ANSI Codes** | Códigos de escape para colores (`[1;96m`) |
| **BotOrchestrator** | Componente central del bot |
| **ccxt** | Librería para exchanges de crypto |
| **Deprecating** | Estrategia deshabilitada (`active: false`) |
| **Direction** | 'long' (compra) o 'short' (venta) |
| **Entry Price** | Precio de apertura de posición |
| **Flask** | Framework web para dashboard |
| **HMAC** | Hash-based Message Authentication Code |
| **Lookback** | Velas hacia atrás para análisis |
| **Main Loop** | Bucle infinito del bot |
| **OHLCV** | Open, High, Low, Close, Volume |
| **Order Amount** | USDT a invertir por posición |
| **Position** | Orden abierta en el exchange |
| **Registry** | Mapeo nombre → función |
| **sell_after_ncandles** | Timeout en velas |
| **Signal** | Indicación de abrir posición |
| **SL (Stop Loss)** | Cierre automático por pérdidas |
| **State File** | JSON con posiciones y candles |
| **Strategy** | Configuración de trading |
| **Sync** | Sincronización con broker |
| **Timeframe** | Intervalo de velas (4H, 1H, etc.) |
| **Tolerance** | Tolerancia de precio (%) |
| **TP (Take Profit)** | Cierre automático por ganancias |
| **Tracking** | Seguimiento de posiciones |
| **USDT** | Tether (stablecoin) |
| **WebSocket** | Conexión bidireccional persistente |
| **YAML** | Formato de configuración |

---

# 🎉 FIN DEL DOCUMENTO

Este documento técnico completo cubre toda la arquitectura, lógica, componentes, flujos y configuración del sistema BOT_trading.

**Para consultas o actualizaciones, referirse a este documento.**

---

**Última actualización:** 2026-01-05  
**Versión:** 2.0  
**Autor:** Trading Bot Team  
**Sistema:** BOT_trading

