import time
import multiprocessing as mp
from harness import emit_verify_tag
from lab_config import STUDENT_ID

if __name__ == "__main__":
    N = 2_000_000
    pc = time.perf_counter_ns

    t0 = pc()
    for _ in range(N):
        pc()
    t1 = pc()
    avg_ns = (t1 - t0) / N

    t0 = pc()
    for _ in range(N):
        pass
    loop_ns = (pc() - t0) / N

    print(f"cpu_count = {mp.cpu_count()}")
    print(f"avg perf_counter_ns call (incl. loop) = {avg_ns:.2f} ns")
    print(f"empty loop overhead = {loop_ns:.2f} ns -> net = {avg_ns - loop_ns:.2f} ns")
    print(emit_verify_tag("PHASE0", STUDENT_ID, avg_ns, float(mp.cpu_count())))