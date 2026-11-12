import math
import multiprocessing as mp
import time
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103374

def stage_a_work(data_slice):
    res = 0.0
    for x in data_slice:
        for _ in range(10):
            res += math.sin(x) + math.sqrt(abs(x) + 1.0)
    return res

def stage_b_worker_task(chunk):
    res = 0.0
    for x in chunk:
        for _ in range(30):
            res += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return res

def make_chunks(data, p):
    n = len(data)
    chunk_size = math.ceil(n / p)
    return [data[i:i + chunk_size] for i in range(0, n, chunk_size)]

def main():
    P_log = mp.cpu_count()
    P_phys = 8
    print(f"--- Phase 2: Strong & Weak Scaling (P_phys = {P_phys}, P_log = {P_log}) ---")

    N_strong = 1_200_000
    data_strong = [float(i) * 0.001 for i in range(N_strong)]
    stage_a_strong = data_strong[:int(0.05 * N_strong)]

    stats_a1 = run_benchmark(lambda: stage_a_work(stage_a_strong), warmup=1, runs=3)
    t_stage_a_1 = stats_a1.median_ms

    print("\n[Strong Scaling Sweep]")
    t1_strong = 0.0
    strong_results = []

    for p in range(1, P_log + 1):
        chunks = make_chunks(data_strong, p)
        with mp.Pool(processes=p) as pool:
            def trial_strong():
                stage_a_work(stage_a_strong)
                pool.map(stage_b_worker_task, chunks)

            stats = run_benchmark(trial_strong, warmup=1, runs=3)
            tp = stats.median_ms

        if p == 1:
            t1_strong = tp
        sp = t1_strong / tp
        ep = sp / p
        strong_results.append((p, tp, sp, ep))
        print(f"  p = {p:2d}: T_p = {tp:7.2f} ms, S_p = {sp:5.2f}, E_p = {ep:5.2f}")

    s_amdahl = t_stage_a_1 / t1_strong
    s_inf = 1.0 / s_amdahl
    s_plog_strong = strong_results[-1][2]
    print(f"  -> Amdahl Cap S_inf = {s_inf:.2f}, Observed S_{P_log} = {s_plog_strong:.2f}")

    N_0 = 200_000
    print("\n[Weak Scaling Sweep]")
    t1_weak = 0.0
    weak_results = []

    for p in range(1, P_log + 1):
        n_curr = p * N_0
        data_curr = [float(i) * 0.001 for i in range(n_curr)]
        stage_a_curr = data_curr[:int(0.05 * n_curr)]
        chunks = make_chunks(data_curr, p)

        t0_a = time.perf_counter_ns()
        stage_a_work(stage_a_curr)
        t_stage_a_p = (time.perf_counter_ns() - t0_a) / 1e6

        with mp.Pool(processes=p) as pool:
            def trial_weak():
                stage_a_work(stage_a_curr)
                pool.map(stage_b_worker_task, chunks)

            stats = run_benchmark(trial_weak, warmup=1, runs=3)
            tp_weak = stats.median_ms

        if p == 1:
            t1_weak = tp_weak

        ep_weak = t1_weak / tp_weak
        alpha = t_stage_a_p / tp_weak if tp_weak > 0 else 0
        s_gustafson = p - alpha * (p - 1)
        weak_results.append((p, n_curr, tp_weak, ep_weak, alpha, s_gustafson))
        print(f"  p = {p:2d}: T_p_weak = {tp_weak:7.2f} ms, S_Gustafson = {s_gustafson:5.2f}")

    s_plog_gustafson = weak_results[-1][5]

    tag = emit_verify_tag("PHASE2", STUDENT_ID, s_plog_strong, s_plog_gustafson)
    print("\n" + tag)

if __name__ == "__main__":
    mp.freeze_support()
    main()