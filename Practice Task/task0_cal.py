import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230103374

def main():
    p_log = mp.cpu_count()
    iterations = 1_000_000
    
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    
    avg_ns = (t1 - t0) / iterations
    print(f"Logical Cores (P_log): {p_log}")
    print(f"Average perf_counter_ns latency: {avg_ns:.2f} ns")
    
    tag = emit_verify_tag("PHASE0", STUDENT_ID, avg_ns, float(p_log))
    print(tag)

if __name__ == "__main__":
    main()