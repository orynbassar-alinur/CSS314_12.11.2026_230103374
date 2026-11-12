import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from lab_config import STUDENT_ID, P_PHYS, P_LOG

M = 24_000_000
NSLOTS = 512  # 4096 B / 8 B


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * NSLOTS).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr          # освободить буфер перед close()
    shm.close()


def run_once(p: int, stride: int) -> None:
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        iters = M // p
        procs = [mp.Process(target=fs_worker, args=(shm.name, t * stride, iters))
                 for t in range(p)]
        for q in procs:
            q.start()
        for q in procs:
            q.join()
    finally:
        shm.close()
        shm.unlink()


def bench(p: int, stride: int):
    return run_benchmark(lambda: run_once(p, stride), warmup_runs=2, timed_runs=7)


if __name__ == "__main__":
    # ---------- Table 1A ----------
    print(f"## Table 1A: stride sweep at p = {P_PHYS}", flush=True)
    print("| stride | bytes | median ms | IQR ms | Mops/s |")
    print("|---|---|---|---|---|", flush=True)
    prev = None
    for stride in (1, 2, 4, 8, 16, 32):
        s = bench(P_PHYS, stride)
        mops = M / (s.median_ms * 1e3)
        line = f"| {stride} | {stride * 8} | {s.median_ms:.1f} | {s.iqr_ms:.1f} | {mops:.2f} |"
        if prev is not None:
            line += f"  <- {prev / s.median_ms:.2f}x faster than previous stride"
        print(line, flush=True)
        prev = s.median_ms

    # ---------- Table 1B ----------
    ps = sorted({1, 2, 4, 8, 16, P_PHYS, P_LOG})
    med = {}
    for stride in (1, 16):
        print(f"\n## Table 1B: stride = {stride}", flush=True)
        print("| p | T_p ms | S_p | E_p | e(p) |")
        print("|---|---|---|---|---|", flush=True)
        T1 = None
        for p in ps:
            s = bench(p, stride)
            med[(stride, p)] = s.median_ms
            if p == 1:
                T1 = s.median_ms
            Sp = T1 / s.median_ms
            Ep = Sp / p
            e = (1 / Sp - 1 / p) / (1 - 1 / p) if p >= 2 else float("nan")
            e_txt = f"{e:.3f}" if p >= 2 else "—"
            print(f"| {p} | {s.median_ms:.1f} | {Sp:.2f} | {Ep:.2f} | {e_txt} |", flush=True)

    print()
    print(emit_verify_tag("PHASE1", STUDENT_ID,
                          med[(1, P_PHYS)], med[(16, P_PHYS)]), flush=True)