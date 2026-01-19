#!/bin/bash

echo "========================================"
echo "VERIFICACIÓN FINAL - Refactorización"
echo "========================================"
echo ""

ERRORS=0

# 1. Verificar YAMLs
echo "1. Verificando YAMLs..."
for account in "00" "E1" "01"; do
    if [ -f "config/strategies_${account}.yaml" ]; then
        echo "   ✅ strategies_${account}.yaml existe"
    else
        echo "   ❌ ERROR: strategies_${account}.yaml NO existe"
        ERRORS=$((ERRORS + 1))
    fi
done

# 2. Test de carga
echo ""
echo "2. Test de carga de estrategias..."
python3 -c "
import sys
sys.path.insert(0, '.')
from strategies.strategy_loader import load_strategies

for account in ['00', 'E1', '01']:
    try:
        strategies = load_strategies(account)
        active = sum(1 for s in strategies if s.get('active', True))
        print(f'   ✅ {account}: {len(strategies)} total, {active} activas')
    except Exception as e:
        print(f'   ❌ {account}: ERROR - {e}')
        sys.exit(1)
"

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

# 3. Verificar sintaxis
echo ""
echo "3. Verificando sintaxis Python..."
for file in config/utils.py config/settings.py strategies/strategy_loader.py core/orchestrator.py; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "   ✅ $file"
    else
        echo "   ❌ ERROR: $file"
        ERRORS=$((ERRORS + 1))
    fi
done

# Resultado final
echo ""
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ VERIFICACIÓN EXITOSA"
    echo "========================================"
    echo ""
    echo "¡Refactorización completada!"
    echo ""
    echo "SIGUIENTE: Probar el bot"
    echo "  python3 main.py --account 01"
    exit 0
else
    echo "❌ FALLÓ: $ERRORS errores"
    echo "========================================"
    exit 1
fi
