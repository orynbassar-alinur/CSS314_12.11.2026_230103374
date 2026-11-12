import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103374 

def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    arr = (ctypes.c_uint64 * 512).from_buffer(existing_shm.buf)
    acc = arr[slot_idx]
    
    for _ in range(iters):
        acc = (acc * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        
    arr[slot_idx] = acc
    
    del arr
    existing_shm.close()

def run_stride_experiment(stride: int, p: int, total_iters: int = 1_600_000) -> float:
    shm = shared_memory.SharedMemory(create=True, size=4096)
    iters_per_worker = total_iters // p
    
    def job():
        workers = []
        for i in range(p):
            slot = (i * stride) % 512
            w = mp.Process(target=fs_worker, args=(shm.name, slot, iters_per_worker))
            workers.append(w)
            w.start()
        for w in workers:
            w.join()

    stats = run_benchmark(job, warmup=1, runs=3)
    shm.close()
    shm.unlink()
    return stats.median_ms

def main():
    P_phys = 8  
    strides = [1, 2, 4, 8, 16, 32]
    
    print(f"--- Phase 1: Stride Sweep (p = {P_phys} physical cores) ---")
    med_stride_1 = 0.0
    med_stride_16 = 0.0
    
    for s in strides:
        med = run_stride_experiment(s, P_phys)
        byte_offset = s * 8
        print(f"Stride {s:2d} ({byte_offset:3d} bytes): Median Time = {med:.2f} ms")
        if s == 1:
            med_stride_1 = med
        if s == 16:
            med_stride_16 = med

    tag = emit_verify_tag("PHASE1", STUDENT_ID, med_stride_1, med_stride_16)
    print(tag)

if __name__ == "__main__":
    mp.freeze_support()
    main()