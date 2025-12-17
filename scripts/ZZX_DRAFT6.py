#!/usr/bin/env python3
import os
import threading
import multiprocessing as mp
import time
import ctypes
from datetime import datetime

TARGET_CPU = 6

# syscall para saber CPU actual
libc = ctypes.CDLL("libc.so.6")
SYS_getcpu = 309  # x86_64

def current_cpu():
    cpu = ctypes.c_uint()
    node = ctypes.c_uint()
    libc.syscall(SYS_getcpu, ctypes.byref(cpu), ctypes.byref(node), None)
    return cpu.value

def log(name):
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"{name:<10} | "
        f"PID={os.getpid()} | "
        f"TID={threading.get_native_id()} | "
        f"allowed={sorted(os.sched_getaffinity(0))} | "
        f"cpu={current_cpu()}",
        flush=True
    )

def worker_thread(i):
    for _ in range(5):
        log(f"thread-{i}")
        time.sleep(0.5)

def worker_process(i):
    for _ in range(5):
        log(f"process-{i}")
        time.sleep(0.5)

def main():
    print("=== Demo REAL de CPU affinity ===")

    # FIJAR AFINIDAD DEL PROCESO PADRE
    os.sched_setaffinity(0, {TARGET_CPU})

    log("main-start")

    # Threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_thread, args=(i,))
        t.start()
        threads.append(t)

    # Procesos hijos
    procs = []
    for i in range(2):
        p = mp.Process(target=worker_process, args=(i,))
        p.start()
        procs.append(p)

    for _ in range(5):
        log("main-loop")
        time.sleep(0.5)

    for t in threads:
        t.join()
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
