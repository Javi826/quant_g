import psutil
import time
from datetime import datetime

CORE = 2
INTERVAL = 0.1  # cada 0.5 segundos

max_usage = 0
min_usage = 100
sum_usage = 0
count = 0

print("📊 Monitor iniciado. Ctrl+C para detener y ver estadísticas...")
try:
    while True:
        usage_per_core = psutil.cpu_percent(interval=INTERVAL, percpu=True)
        core_usage = usage_per_core[CORE]

        # Mostrar en pantalla (opcional)
        print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} | Core {CORE}: {core_usage:.1f}%")

        # Actualizar estadísticas
        max_usage = max(max_usage, core_usage)
        min_usage = min(min_usage, core_usage)
        sum_usage += core_usage
        count += 1

except KeyboardInterrupt:
    promedio = sum_usage / count if count > 0 else 0
    print("\n🔚 Monitor detenido")
    print(f"📈 Estadísticas Core {CORE}:")
    print(f"Máximo: {max_usage:.1f}%")
    print(f"Mínimo:  {min_usage:.1f}%")
    print(f"Promedio: {promedio:.1f}%")
