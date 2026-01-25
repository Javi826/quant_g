"""
Smoke Test - Prueba rápida con datos reales de producción
Verifica que funciones críticas NO crasheen con posiciones activas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state.state_manager import load_state
from execution.position_tracker import check_all_tp_sl
from config.settings import POSTGRES_CONFIG

def test_check_tpsl_with_real_positions():
    """
    Test que check_all_tp_sl funciona con posiciones reales de PostgreSQL.
    
    Este test:
    1. Carga posiciones reales de cuenta 00
    2. Si hay posiciones, ejecuta check_all_tp_sl
    3. Verifica que NO crashea
    """
    print("="*70)
    print("SMOKE TEST: check_all_tp_sl con posiciones reales")
    print("="*70)
    
    # Cargar posiciones reales de cuenta 00
    print("\n[1/3] Cargando estado real de cuenta 00 desde PostgreSQL...")
    try:
        positions, candles = load_state("00", "persistence/bot_state_00.json")
        
        total_positions = sum(len(p) for p in positions.values())
        print(f"✓ Cargadas {total_positions} posiciones activas")
        
        if total_positions == 0:
            print("⚠️  No hay posiciones activas, test no aplicable")
            return True
            
    except Exception as e:
        print(f"❌ Error cargando estado: {e}")
        return False
    
    # Mock simple de funciones necesarias
    print("\n[2/3] Preparando mocks mínimos...")
    
    def mock_send_request(*args, **kwargs):
        return {'code': '00000', 'data': {}}
    
    def mock_check_tp_sl(*args, **kwargs):
        pass  # Solo verifica que se llama sin error
    
    strategies = [
        {
            'id': strat_id,
            'sell_after_ncandles': 50
        }
        for strat_id in positions.keys()
    ]
    
    # Ejecutar check_all_tp_sl
    print("\n[3/3] Ejecutando check_all_tp_sl con datos reales...")
    try:
        result = check_all_tp_sl(
            strategies=strategies,
            open_positions=positions,
            strategy_candles=candles,
            account_number="00",  # ← CRÍTICO
            state_file="persistence/bot_state_00.json",
            send_request_func=mock_send_request,
            check_tp_sl_for_strategy_func=mock_check_tp_sl,
            bot_state=None
        )
        
        print(f"✓ check_all_tp_sl ejecutado sin errores")
        print(f"✓ Resultado: {result}")
        print("\n" + "="*70)
        print("✅ SMOKE TEST PASSED")
        print("="*70)
        return True
        
    except TypeError as e:
        if "'BotState' object is not subscriptable" in str(e):
            print(f"\n❌ ERROR CRÍTICO: Bug de migración no corregido")
            print(f"   {e}")
            return False
        raise
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_check_tpsl_with_real_positions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test crasheó: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
