import psutil
from datetime import datetime

CORES = [1, 2]
INTERVAL = 0.1  # segundos

stats = {
    core: {
        "max": 0,
        "min": 100,
        "sum": 0,
        "count": 0
    }
    for core in CORES
}

print("📊 Monitor init. Ctrl+C for stopping and stats...")

try:
    while True:
        usage_per_core = psutil.cpu_percent(interval=INTERVAL, percpu=True)
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        for core in CORES:
            usage = usage_per_core[core]
            s = stats[core]

            print(f"{timestamp} | Core {core}: {usage:.1f}%")

            s["max"] = max(s["max"], usage)
            s["min"] = min(s["min"], usage)
            s["sum"] += usage
            s["count"] += 1

except KeyboardInterrupt:
    print("\n🔚 Monitor stopped")
    for core in CORES:
        s = stats[core]
        promedio = s["sum"] / s["count"] if s["count"] > 0 else 0

        print(f"\n📈 Core stats {core}:")
        print(f"Máximo:   {s['max']:.1f}%")
        print(f"Mínimo:   {s['min']:.1f}%")
        print(f"Promedio: {promedio:.1f}%")
