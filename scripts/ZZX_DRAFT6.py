#!/usr/bin/env python3
"""
Script para verificar y reemplazar el archivo bot_state.json correcto
"""

import json
import os
import shutil
from datetime import datetime

# Archivo correcto
CORRECT_FILE = '/home/javi/projects/quant/quant_g/scripts/live_trading/bot_state.json'

print("=" * 80)
print("🔍 VERIFICANDO ARCHIVO ACTUAL")
print("=" * 80)

if os.path.exists(CORRECT_FILE):
    print(f"\n📁 Archivo encontrado: {CORRECT_FILE}")
    
    # Hacer backup
    backup_file = CORRECT_FILE + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(CORRECT_FILE, backup_file)
    print(f"✅ Backup creado: {backup_file}")
    
    # Leer contenido actual
    with open(CORRECT_FILE, 'r') as f:
        data = json.load(f)
    
    positions = data.get('positions', {})
    
    print(f"\n📊 CONTENIDO ACTUAL:")
    print(f"   revers_short_4H: {len(positions.get('revers_short_4H', []))} posiciones")
    print(f"   revers_long_1H: {len(positions.get('revers_long_1H', []))} posiciones")
    
    # Mostrar detalles si hay posiciones
    for strat in ['revers_short_4H', 'revers_long_1H']:
        pos_list = positions.get(strat, [])
        if pos_list:
            print(f"\n   {strat}:")
            for pos in pos_list:
                print(f"      - {pos.get('symbol')} | Entry: {pos.get('entry_price')} | Size: {pos.get('size')}")
    
    # Limpiar esas estrategias
    print(f"\n🧹 LIMPIANDO ESTRATEGIAS...")
    positions['revers_short_4H'] = []
    positions['revers_long_1H'] = []
    
    # Guardar archivo limpio
    with open(CORRECT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Archivo limpiado y guardado")
    
    # Verificar
    with open(CORRECT_FILE, 'r') as f:
        data_verify = json.load(f)
    
    positions_verify = data_verify.get('positions', {})
    print(f"\n✅ VERIFICACIÓN:")
    print(f"   revers_short_4H: {len(positions_verify.get('revers_short_4H', []))} posiciones")
    print(f"   revers_long_1H: {len(positions_verify.get('revers_long_1H', []))} posiciones")
    
else:
    print(f"❌ Archivo no encontrado: {CORRECT_FILE}")

print("\n" + "=" * 80)