import cupy as cp
print("Number of GPUs available:", cp.cuda.runtime.getDeviceCount())
