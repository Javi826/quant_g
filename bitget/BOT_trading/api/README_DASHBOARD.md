# 📊 README_DASHBOARD

## 🎯 RESUMEN
Este documento explica las modificaciones realizadas en el dashboard del BOT_trading y cómo implementarlas.

---

## 📁 ARCHIVOS MODIFICADOS

### **1. dashboard.js** (`api/static/js/dashboard.js`)
- **Tamaño:** ~101KB (2407 líneas)
- **Función:** Lógica del dashboard (frontend JavaScript)

### **2. dashboard.html** (`api/templates/dashboard.html`)
- **Tamaño:** ~38KB (650 líneas)
- **Función:** Estructura HTML del dashboard

### **3. base.html** (`api/templates/base.html`)
- **Tamaño:** ~1KB (30 líneas)
- **Función:** Template base (favicon, CSS, JS)

---

## 🔧 CAMBIOS IMPLEMENTADOS

### **A. POSITIONS TAB - Sort Feature**
- ✅ Botones **TP** y **SL** para ordenar posiciones en vista Detailed
- ✅ Botones integrados inline a la derecha de los headers de columna
- ✅ Variable `positionSortBy` controla el orden ('tp' o 'sl')

**Funciones nuevas:**
```javascript
sortPositionsBy(type)  // Cambia el orden TP/SL
```

**Modificaciones:**
- `renderDetailedView()` - Usa positionSortBy para ordenar
- `setPositionsView()` - Muestra/oculta según vista

---

### **B. CONFIGURATION TAB - Reordenado**

**Orden NUEVO:**
1. **Strategies List** (tabla principal de estrategias)
2. **Market Regime Strategy Matrix** (NEW - con columna #)
3. **Regime Family Matrix** (NEW - multiplicadores globales)
4. **WebSocket Connections** (movido aquí)
5. **Configuration** (movido aquí)

**Eliminado:**
- ❌ Sección "Timeframes"
- ❌ Columna `regime_family` de Strategies List (redundante)
- ❌ Emojis de títulos

---

### **C. MARKET REGIME - Nuevas Tablas**

#### **Tabla 1: Market Regime Strategy Matrix**
Muestra el regime family de cada estrategia y sus multiplicadores.

**Columnas:**
- `#` - Número correlativo
- `Strategy` - ID de la estrategia
- `Family` - trending/ranging/volatile/Global
- `Trending` - Multiplicador cuando mercado es trending
- `Ranging` - Multiplicador cuando mercado es ranging
- `Volatile` - Multiplicador cuando mercado es volatile

**Funciones nuevas:**
```javascript
renderRegimeStrategyMatrix(strategies)  // Renderiza la tabla
getFamilyColor(family)                  // Colores por familia
```

#### **Tabla 2: Regime Family Matrix**
Matriz global de multiplicadores 3x3 (familia × mercado).

**Columnas:**
- `Family` - trending/ranging/volatile
- `Trending` - Multiplicador en mercado trending
- `Ranging` - Multiplicador en mercado ranging
- `Volatile` - Multiplicador en mercado volatile

**Funciones nuevas:**
```javascript
loadRegimeGlobalMatrix()  // Carga matriz desde API
loadRegimeSizing()        // Carga sizing global
```

---

### **D. REGIME CARDS - Sin Multiplicadores**
Las 3 cards de régimen (VOLATILE, RANGING, TRENDING) ya **NO** muestran multiplicadores.

Solo muestran:
- Nombre del régimen
- Reglas de detección
- Descripción

---

## 🚀 IMPLEMENTACIÓN

### **PASO 1: Backup**
```bash
cd /home/javi/projects/quant/quant_g/bitget/BOT_trading

# Crear backups con timestamp
cp api/static/js/dashboard.js api/static/js/dashboard.js.backup_$(date +%Y%m%d_%H%M%S)
cp api/templates/dashboard.html api/templates/dashboard.html.backup_$(date +%Y%m%d_%H%M%S)
```

### **PASO 2: Reemplazar archivos**
```bash
# Mover archivos descargados desde /home/javi/Descargas
mv /home/javi/Descargas/dashboard_FINAL.js api/static/js/dashboard.js
mv /home/javi/Descargas/dashboard_FINAL.html api/templates/dashboard.html

# Verificar permisos
chmod 644 api/static/js/dashboard.js
chmod 644 api/templates/dashboard.html
```

### **PASO 3: Verificar**
```bash
ls -lh api/static/js/dashboard.js
ls -lh api/templates/dashboard.html
```

### **PASO 4: Reiniciar bot**
Los cambios se aplican automáticamente al recargar la página (Flask en modo desarrollo).

---

## 🐛 TROUBLESHOOTING

### **1. Favicon no aparece**
**Problema:** El favicon de la cuenta 01 no se muestra.

**Causa:** Caché del navegador (los favicons se cachean muy agresivamente).

**Solución:**
```bash
# A. Hard refresh
Ctrl + Shift + R (Windows/Linux)
Cmd + Shift + R (Mac)

# B. Abrir en incógnito
Ctrl + Shift + N (Chrome)
Ctrl + Shift + P (Firefox)

# C. Limpiar caché completa
Ctrl + Shift + Delete → "Imágenes y archivos en caché" → "Desde siempre"

# D. Forzar recarga del favicon
1. Abrir: http://localhost:5001/static/bots/01/favicon.jpg
2. Ctrl + Shift + R en esa página
3. Volver al dashboard
```

**Ubicación del favicon:**
```
api/static/bots/01/favicon.jpg
```

---

### **2. Sort buttons no aparecen**
**Causa:** Solo visibles en vista "Detailed".

**Solución:**
1. Click en botón **Detailed** (arriba a la derecha)
2. Los botones **TP** y **SL** aparecen inline en los headers

---

### **3. Tablas de régimen vacías**
**Causa:** Backend no está devolviendo datos.

**Verificar:**
```bash
# Comprobar que existe el endpoint
curl http://localhost:5001/api/regime/matrix

# Debe devolver JSON con:
# { "success": true, "matrix": {...} }
```

**Si falla:** Verificar que el backend tiene las modificaciones de régimen implementadas.

---

### **4. JavaScript no carga**
**Síntomas:** Dashboard no funciona, botones no responden.

**Solución:**
```bash
# A. Limpiar caché del navegador
Ctrl + Shift + R

# B. Verificar que el archivo existe y tiene permisos
ls -lh api/static/js/dashboard.js
chmod 644 api/static/js/dashboard.js

# C. Ver errores en consola del navegador
F12 → Console (buscar errores en rojo)
```

---

## 📊 DEPENDENCIAS DEL BACKEND

Para que las nuevas tablas funcionen, el backend debe tener:

### **1. Endpoint `/api/regime/matrix`**
Devuelve la matriz global de multiplicadores:
```json
{
  "success": true,
  "matrix": {
    "trending": { "trending": 2.0, "ranging": 0.5, "volatile": 0.3 },
    "ranging": { "trending": 0.5, "ranging": 1.5, "volatile": 0.5 },
    "volatile": { "trending": 0.3, "ranging": 0.5, "volatile": 1.0 }
  }
}
```

### **2. Endpoint `/api/regime/current?timeframe=4H`**
Devuelve régimen actual y configuración:
```json
{
  "success": true,
  "family": "trending",
  "multiplier": 2.0,
  "metrics": { "hurst": 0.65, "efficiency_ratio": 0.72, ... },
  "all_families": { "trending": 2.0, "ranging": 1.0, "volatile": 0.5 },
  "all_thresholds": { ... }
}
```

### **3. Campo `regime_family` en strategies**
Cada estrategia debe tener el campo `regime_family` en `/api/bot-config`:
```json
{
  "strategies": [
    {
      "id": "00_strat_name",
      "regime_family": "trending",  // ← Este campo
      ...
    }
  ]
}
```

---

## 📝 NOTAS IMPORTANTES

1. **Los cambios son solo frontend** - No afectan la lógica del bot
2. **Compatibilidad:** Funciona con cuentas 00, 01, E1
3. **Caché:** Siempre usar Ctrl+Shift+R al actualizar archivos
4. **Backups:** Los backups se guardan con timestamp en el mismo directorio
5. **Variables glob