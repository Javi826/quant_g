import numpy as np
import time

# Intentamos importar CuPy para GPU
try:
    import cupy as cp
    gpu_available = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    gpu_available = False

# Tamaño de los arrays grandes
N = 100_000_000

print(f"GPU available: {gpu_available}")

# --- CPU computation ---
a_cpu = np.random.rand(N)
b_cpu = np.random.rand(N)

start_cpu = time.time()
c_cpu = a_cpu + b_cpu
end_cpu = time.time()

cpu_time = end_cpu - start_cpu
print(f"CPU time: {cpu_time:.5f} seconds")

# --- GPU computation ---
if gpu_available:
    a_gpu = cp.array(a_cpu)
    b_gpu = cp.array(b_cpu)

    start_gpu = time.time()
    c_gpu = a_gpu + b_gpu
    cp.cuda.Stream.null.synchronize()  # sincroniza para medir tiempo real
    end_gpu = time.time()

    gpu_time = end_gpu - start_gpu
    print(f"GPU time: {gpu_time:.5f} seconds")

    # Comprobar que resultados coinciden
    difference = np.max(np.abs(c_cpu - cp.asnumpy(c_gpu)))
    print(f"Max difference between CPU and GPU results: {difference:.5e}")
else:
    print("GPU not available, only CPU was used.")
