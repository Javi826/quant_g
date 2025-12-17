import psutil
import time
import os

def monitor_cpu(interval=0.01):
    """
    Muestra el uso de todos los núcleos en tiempo real,
    cada núcleo en una línea separada.
    """
    try:
        while True:
            usage_per_core = psutil.cpu_percent(interval=interval, percpu=True)
            # Limpiar la pantalla para actualizar en “live”
            os.system('clear')
            print("🖥️ CPU usage per core:")
            for i, u in enumerate(usage_per_core):
                print(f"Core {i+1}: {u:5.1f}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🔚 Monitor detenido")

# Ejemplo de uso:
monitor_cpu(interval=1)
