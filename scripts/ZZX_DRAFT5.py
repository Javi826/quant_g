#!/usr/bin/env python3
"""
Script de migración AUTOCONTENIDO - Crea nuevo Excel con IDs migrados
Cuenta: 00
NO modifica el archivo original - crea uno nuevo con sufijo _new
"""

import pandas as pd
import os

# ==========================================================================
# CONFIGURACIÓN
# ==========================================================================
ACCOUNT = "E1"
BASE_DIR = os.path.expanduser(f'~/projects/quant/quant_g/scripts/live_trading/bot_files_{ACCOUNT}')
EXCEL_ORIGINAL = os.path.join(BASE_DIR, f'bot_trades_{ACCOUNT}.xlsx')
EXCEL_NUEVO = os.path.join(BASE_DIR, f'bot_trades_{ACCOUNT}_new.xlsx')

# ==========================================================================
# MAPEO DE IDs
# ==========================================================================
MAPEO = {
    # Double top
    'double_top_long_4H': '01_double_top_long_4H',
    
    # Reversal - Excel tiene "revers" pero mapeamos a "reversal" completo
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
    print("MIGRACIÓN EXCEL - CUENTA 00")
    print("Crea archivo nuevo sin modificar el original")
    print("=" * 80)
    
    # 1. VERIFICAR ARCHIVO ORIGINAL
    if not os.path.exists(EXCEL_ORIGINAL):
        print(f"\n❌ ERROR: No se encuentra el archivo original")
        print(f"   Buscado en: {EXCEL_ORIGINAL}")
        return
    
    print(f"\n📂 Archivo original: {os.path.basename(EXCEL_ORIGINAL)}")
    print(f"📂 Archivo nuevo:    {os.path.basename(EXCEL_NUEVO)}")
    
    # 2. LEER EXCEL ORIGINAL
    print(f"\n📊 Leyendo Excel original...")
    try:
        df = pd.read_excel(EXCEL_ORIGINAL, engine='openpyxl')
    except Exception as e:
        print(f"❌ ERROR al leer Excel: {e}")
        return
    
    print(f"   Total trades: {len(df)}")
    print(f"   Columnas: {', '.join(df.columns.tolist())}")
    
    # 3. VERIFICAR COLUMNA STRATEGY
    if 'STRATEGY' not in df.columns:
        print(f"\n❌ ERROR: No existe columna STRATEGY en el Excel")
        return
    
    # 4. ANALIZAR ESTRATEGIAS
    print(f"\n🔍 Estrategias encontradas:")
    estrategias_unicas = df['STRATEGY'].unique()
    print(f"   Total únicas: {len(estrategias_unicas)}\n")
    
    sin_mapear = []
    for strat in sorted(estrategias_unicas):
        count = len(df[df['STRATEGY'] == strat])
        if strat in MAPEO:
            print(f"   ✅ {strat:<30} → {MAPEO[strat]:<35} ({count:>3} trades)")
        else:
            print(f"   ⚠️  {strat:<30} → SIN MAPEO                            ({count:>3} trades)")
            sin_mapear.append(strat)
    
    # 5. ADVERTENCIA SI HAY SIN MAPEAR
    if sin_mapear:
        print(f"\n⚠️  ADVERTENCIA: {len(sin_mapear)} estrategia(s) sin mapear:")
        for s in sin_mapear:
            print(f"   - {s}")
        print(f"\n   Estas estrategias NO se modificarán (quedarán con ID original)")
        print(f"   Si quieres mapearlas, añádelas al script y vuelve a ejecutar")
    
    # 6. APLICAR MIGRACIÓN
    print(f"\n🔄 Aplicando migración...")
    df_nuevo = df.copy()
    df_nuevo['STRATEGY'] = df_nuevo['STRATEGY'].replace(MAPEO)
    
    # 7. MOSTRAR RESULTADO
    print(f"\n✅ Vista previa del resultado:")
    estrategias_nuevas = df_nuevo['STRATEGY'].value_counts()
    print(f"\n   Estrategias después de migración:")
    for strat in sorted(estrategias_nuevas.index):
        count = estrategias_nuevas[strat]
        print(f"   {strat:<40} ({count:>3} trades)")
    
    # 8. GUARDAR NUEVO ARCHIVO
    print(f"\n💾 Guardando nuevo archivo...")
    try:
        df_nuevo.to_excel(EXCEL_NUEVO, index=False, engine='openpyxl')
        print(f"   ✅ Archivo creado: {EXCEL_NUEVO}")
    except Exception as e:
        print(f"   ❌ ERROR al guardar: {e}")
        return
    
    # 9. RESUMEN FINAL
    print(f"\n" + "=" * 80)
    print("✅ MIGRACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\nArchivos:")
    print(f"   📂 Original (sin cambios): {EXCEL_ORIGINAL}")
    print(f"   📂 Nuevo (migrado):        {EXCEL_NUEVO}")
    
    print(f"\nEstadísticas:")
    print(f"   Total trades:          {len(df_nuevo)}")
    print(f"   Estrategias únicas:    {len(estrategias_nuevas)}")
    print(f"   Estrategias mapeadas:  {len(estrategias_unicas) - len(sin_mapear)}")
    print(f"   Sin mapear:            {len(sin_mapear)}")
    
    print(f"\n" + "=" * 80)
    print("PRÓXIMOS PASOS:")
    print("=" * 80)
    print("1. Abre el archivo nuevo y verifica que los IDs sean correctos")
    print("2. Si todo está bien, renombra:")
    print(f"   mv {EXCEL_ORIGINAL} {EXCEL_ORIGINAL}.backup")
    print(f"   mv {EXCEL_NUEVO} {EXCEL_ORIGINAL}")
    print("3. Actualiza dashboard para eliminar mapeos")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()