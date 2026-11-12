import math
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from harness import emit_verify_tag

STUDENT_ID = 230103374

def compute_item(i):
    iters = 1 + int((i / 40) ** 2)
    val = 0.0
    for k in range(iters):
        val += math.sin(k) * math.cos(k) + math.sqrt(k + 1.0)
    return val

def worker_block(rank, p, n, times_arr):
    t0 = time.perf_counter_ns()
    chunk_size = math.ceil(n / p)
    start = rank * chunk_size
    end = min(start + chunk_size, n)
    for i in range(start, end):
        compute_item(i)
    t1 = time.perf_counter_ns()
    times_arr[rank] = (t1 - t0) / 1e6

def worker_dynamic(rank, n, chunk_size, counter, times_arr):
    t0 = time.perf_counter_ns()
    while True:
        with counter.get_lock():
            start = counter.value
            counter.value += chunk_size
        if start >= n:
            break
        end = min(start + chunk_size, n)
        for i in range(start, end):
            compute_item(i)
    t1 = time.perf_counter_ns()
    times_arr[rank] = (t1 - t0) / 1e6

def main():
    P_phys = 8
    N = 8000
    print(f"--- Phase 4: Skewed Workloads & Dynamic Chunk Queue (N = {N}, P_phys = {P_phys}) ---")

    times_block = mp.Array('d', P_phys)
    workers = []
    t_start = time.perf_counter_ns()
    for r in range(P_phys):
        w = mp.Process(target=worker_block, args=(r, P_phys, N, times_block))
        workers.append(w)
        w.start()
    for w in workers:
        w.join()
    t_wall_block = (time.perf_counter_ns() - t_start) / 1e6

    max_t_b = max(times_block)
    avg_t_b = sum(times_block) / P_phys
    lambda_block = max_t_b / avg_t_b if avg_t_b > 0 else 1.0
    print(f"  [Block Static] T_wall = {t_wall_block:.2f} ms, Imbalance Lambda = {lambda_block:.2f}")

    chunks = [1, 8, 64, 512]
    best_c = 64
    t_wall_dynamic_optimal = 0.0

    print("  [Dynamic Queue Sweep]")
    for c_size in chunks:
        counter = mp.Value('i', 0)
        times_dyn = mp.Array('d', P_phys)
        workers = []
        t_start = time.perf_counter_ns()
        for r in range(P_phys):
            w = mp.Process(target=worker_dynamic, args=(r, N, c_size, counter, times_dyn))
            workers.append(w)
            w.start()
        for w in workers:
            w.join()
        t_wall_dyn = (time.perf_counter_ns() - t_start) / 1e6
        
        max_t_d = max(times_dyn)
        avg_t_d = sum(times_dyn) / P_phys
        lambda_dyn = max_t_d / avg_t_d if avg_t_d > 0 else 1.0
        print(f"    C = {c_size:3d}: T_wall = {t_wall_dyn:6.2f} ms, Lambda = {lambda_dyn:4.2f}")

        if c_size == 64: 
            t_wall_dynamic_optimal = t_wall_dyn

    tag = emit_verify_tag("PHASE4", STUDENT_ID, t_wall_block, t_wall_dynamic_optimal)
    print("\n" + tag)

if __name__ == "__main__":
    mp.freeze_support()
    main()