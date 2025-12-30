#!/usr/bin/env python3
"""
Script de migración AUTOCONTENIDO - Crea nuevo state.json con IDs migrados
Cuenta: 00
NO modifica el archivo original - crea uno nuevo con sufijo _new
"""

import json
import os

# ==========================================================================
# CONFIGURACIÓN
# ==========================================================================
ACCOUNT = "E1"
BASE_DIR = os.path.expanduser(f'~/projects/quant/quant_g/scripts/live_trading/bot_files_{ACCOUNT}')
STATE_ORIGINAL = os.path.join(BASE_DIR, f'bot_state_{ACCOUNT}.json')
STATE_NUEVO = os.path.join(BASE_DIR, f'bot_state_{ACCOUNT}_new.json')

# ==========================================================================
# MAPEO DE IDs
# ==========================================================================
MAPEO = {
    # Double top
    'double_top_long_4H': '01_double_top_long_4H',
    
    # Reversal - Excel/state tiene "revers" pero mapeamos a "reversal" completo
    'revers_long_4H': '02_reversal_long_4H',
    'revers_short_4H': '04_reversal_short_4H',
    'revers_long_1H': '06_reversal_long_1H',
    'revers_short_1H': '07_reversal_short_1H',
    'revers_long_6Hutc': '08_reversal_long_6Hutc',
    'revers_short_6Hutc': '09_reversal_short_6Hutc',
    
    # Parity
    'parity_long_4H': '03_parity_long_4H',
    'parity_short_4H': '05_parity_short_4H',
    'parity_long_1H': '10_parity_long_1H',
    'parity_short_1H': '11_parity_short_1H',
    'parity_long_6Hutc': '12_parity_long_6Hutc',
    
    # Orderblocks
    'orderblocks_short_4H': '13_orderblocks_short_4H',
    'orderblocks_long_4H': '14_orderblocks_long_4H',
}

# ==========================================================================
# SCRIPT
# ==========================================================================
def main():
    print("=" * 80)
    print("MIGRACIÓN STATE.JSON - CUENTA 00")
    print("Crea archivo nuevo sin modificar el original")
    print("=" * 80)
    
    # 1. VERIFICAR ARCHIVO ORIGINAL
    if not os.path.exists(STATE_ORIGINAL):
        print(f"\n❌ ERROR: No se encuentra el archivo original")
        print(f"   Buscado en: {STATE_ORIGINAL}")
        return
    
    print(f"\n📂 Archivo original: {os.path.basename(STATE_ORIGINAL)}")
    print(f"📂 Archivo nuevo:    {os.path.basename(STATE_NUEVO)}")
    
    # 2. LEER JSON ORIGINAL
    print(f"\n📊 Leyendo state.json original...")
    try:
        with open(STATE_ORIGINAL, 'r') as f:
            state = json.load(f)
    except Exception as e:
        print(f"❌ ERROR al leer state.json: {e}")
        return
    
    # 3. ANALIZAR POSICIONES
    positions = state.get('positions', {})
    strategy_candles = state.get('strategy_candles', {})
    
    print(f"\n🔍 Posiciones abiertas:")
    print(f"   Total estrategias con posiciones: {len(positions)}")
    
    total_positions = sum(len(pos_list) for pos_list in positions.values())
    print(f"   Total posiciones abiertas: {total_positions}\n")
    
    sin_mapear_pos = []
    for strat_id in sorted(positions.keys()):
        pos_count = len(positions[strat_id])
        if strat_id in MAPEO:
            print(f"   ✅ {strat_id:<30} → {MAPEO[strat_id]:<35} ({pos_count:>2} pos.)")
        else:
            print(f"   ⚠️  {strat_id:<30} → SIN MAPEO                            ({pos_count:>2} pos.)")
            sin_mapear_pos.append(strat_id)
    
    # 4. ANALIZAR STRATEGY_CANDLES
    print(f"\n🔍 Contadores de velas:")
    print(f"   Total estrategias con contador: {len(strategy_candles)}\n")
    
    sin_mapear_candles = []
    for strat_id in sorted(strategy_candles.keys()):
        candles = strategy_candles[strat_id]
        if strat_id in MAPEO:
            print(f"   ✅ {strat_id:<30} → {MAPEO[strat_id]:<35} ({candles:>2} velas)")
        else:
            print(f"   ⚠️  {strat_id:<30} → SIN MAPEO                            ({candles:>2} velas)")
            sin_mapear_candles.append(strat_id)
    
    # 5. ADVERTENCIA SI HAY SIN MAPEAR
    sin_mapear = list(set(sin_mapear_pos + sin_mapear_candles))
    if sin_mapear:
        print(f"\n⚠️  ADVERTENCIA: {len(sin_mapear)} estrategia(s) sin mapear:")
        for s in sin_mapear:
            print(f"   - {s}")
        print(f"\n   Estas estrategias NO se modificarán (quedarán con ID original)")
        print(f"   Si quieres mapearlas, añádelas al script y vuelve a ejecutar")
    
    # 6. MIGRAR POSITIONS
    print(f"\n🔄 Migrando posiciones...")
    new_positions = {}
    for old_id, pos_list in positions.items():
        new_id = MAPEO.get(old_id, old_id)
        new_positions[new_id] = pos_list
        if old_id in MAPEO:
            print(f"   {old_id:<30} → {new_id}")
    
    # 7. MIGRAR STRATEGY_CANDLES
    print(f"\n🔄 Migrando contadores de velas...")
    new_candles = {}
    for old_id, candles in strategy_candles.items():
        new_id = MAPEO.get(old_id, old_id)
        new_candles[new_id] = candles
        if old_id in MAPEO:
            print(f"   {old_id:<30} → {new_id}")
    
    # 8. CREAR NUEVO STATE
    state_nuevo = state.copy()
    state_nuevo['positions'] = new_positions
    state_nuevo['strategy_candles'] = new_candles
    
    # 9. MOSTRAR RESULTADO
    print(f"\n✅ Vista previa del resultado:")
    print(f"\n   Posiciones después de migración:")
    for strat_id in sorted(new_positions.keys()):
        pos_count = len(new_positions[strat_id])
        print(f"   {strat_id:<40} ({pos_count:>2} pos.)")
    
    print(f"\n   Contadores después de migración:")
    for strat_id in sorted(new_candles.keys()):
        candles = new_candles[strat_id]
        print(f"   {strat_id:<40} ({candles:>2} velas)")
    
    # 10. GUARDAR NUEVO ARCHIVO
    print(f"\n💾 Guardando nuevo archivo...")
    try:
        with open(STATE_NUEVO, 'w') as f:
            json.dump(state_nuevo, f, indent=2)
        print(f"   ✅ Archivo creado: {STATE_NUEVO}")
    except Exception as e:
        print(f"   ❌ ERROR al guardar: {e}")
        return
    
    # 11. RESUMEN FINAL
    print(f"\n" + "=" * 80)
    print("✅ MIGRACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\nArchivos:")
    print(f"   📂 Original (sin cambios): {STATE_ORIGINAL}")
    print(f"   📂 Nuevo (migrado):        {STATE_NUEVO}")
    
    print(f"\nEstadísticas:")
    print(f"   Total posiciones:          {total_positions}")
    print(f"   Estrategias con posiciones: {len(new_positions)}")
    print(f"   Estrategias con contadores: {len(new_candles)}")
    print(f"   Estrategias mapeadas:       {len(positions) - len(sin_mapear_pos)}")
    print(f"   Sin mapear:                 {len(sin_mapear)}")
    
    print(f"\n" + "=" * 80)
    print("PRÓXIMOS PASOS:")
    print("=" * 80)
    print("1. Abre el archivo nuevo y verifica que los IDs sean correctos")
    print("2. ⚠️  IMPORTANTE: Detén el bot antes de reemplazar el state.json")
    print("3. Si todo está bien, renombra:")
    print(f"   mv {STATE_ORIGINAL} {STATE_ORIGINAL}.backup")
    print(f"   mv {STATE_NUEVO} {STATE_ORIGINAL}")
    print("4. Reinicia el bot con los nuevos IDs")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()