import sys
import time
import statistics
import psutil
import os
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import matplotlib.pyplot as plt

# Додаємо src до шляху, щоб Python знайшов модуль findex
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from findex.index import Index, build_partial

def get_memory_rss() -> float:
    """Повертає RSS пам'ять поточного (батьківського) процесу в МБ."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_run(data_dir: Path, workers: int) -> tuple[float, float, float, float]:
    paths = list(data_dir.glob("**/*.txt"))
    if not paths:
        raise ValueError(f"Не знайдено .txt файлів у {data_dir}")

    # Розбиваємо шляхи на шматки (chunks)
    chunk_size = math.ceil(len(paths) / workers) if workers > 0 else len(paths)
    
    # Строга типізація для Pyright
    chunks: list[tuple[list[Path], int]] = []
    
    start_id = 0
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size]
        chunks.append((chunk, start_id))
        start_id += len(chunk)

    # Строга типізація для Pyright
    indices: list[Index] = []
    
    ctx = mp.get_context("spawn") # Строго spawn для стабільності
    
    # --- СТАРТ ЗАМІРІВ ---
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(build_partial, chunk, s_id) for chunk, s_id in chunks]
        for future in as_completed(futures):
            indices.append(future.result())

    # Merge фаза
    merge_start = time.perf_counter()
    final_index = Index()
    for partial_idx in indices:
        final_index.merge(partial_idx)
    merge_time = time.perf_counter() - merge_start

    # --- КІНЕЦЬ ЗАМІРІВ ---
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    
    wall_time = end_wall - start_wall
    cpu_time = end_cpu - start_cpu
    peak_rss = get_memory_rss() # Заміряємо пам'ять батьківського процесу після злиття

    return wall_time, cpu_time, peak_rss, merge_time


def main() -> None:
    # ВКАЖИ ТУТ ШЛЯХ ДО СВОГО КОРПУСУ ДОКУМЕНТІВ
    data_dir = Path("src/findex/data") #type: ignore - шлях до корпусу документів
    
    if not data_dir.exists():
        print(f"Помилка: Директорію {data_dir} не знайдено!")
        return

    workers_list = [1, 2, 4, 8, 20]
    runs_per_config = 3 # Кількість прогонів для медіани
    results: dict[int, float] = {}
    
    gil_status = "Увімкнено (Enabled)"
    if hasattr(sys, "_is_gil_enabled"):
        check_gil = getattr(sys, "_is_gil_enabled")
        gil_status = "Увімкнено" if check_gil() else "Вимкнено (Disabled - 3.13t)"

    print(f"Збірка індексу з {data_dir}")
    print(f"Статус GIL: {gil_status}")
    print(f"{'Воркери':<8} | {'Wall Time (s)':<15} | {'CPU Time (s)':<15} | {'Merge Time (s)':<15} | {'RSS (MB)':<10} | {'Прискорення':<12}")
    print("-" * 95)

    base_wall_time = None

    for w in workers_list:
        # Строга типізація для Pyright
        wall_times: list[float] = []
        cpu_times: list[float] = []
        rss_peaks: list[float] = []
        merge_times: list[float] = []
        
        for _ in range(runs_per_config):
            wt, ct, rss, mt = benchmark_run(data_dir, w)
            wall_times.append(wt)
            cpu_times.append(ct)
            rss_peaks.append(rss)
            merge_times.append(mt)
        
        # Обчислюємо медіани
        med_wt = statistics.median(wall_times)
        med_ct = statistics.median(cpu_times)
        med_rss = statistics.median(rss_peaks)
        med_mt = statistics.median(merge_times)
        
        # Фіксуємо базовий час для обчислення прискорення (Speedup)
        if w == 1:
            base_wall_time = med_wt
            
        speedup = base_wall_time / med_wt if base_wall_time else 1.0
        results[w] = speedup
        
        print(f"{w:<8} | {med_wt:<15.4f} | {med_ct:<15.4f} | {med_mt:<15.4f} | {med_rss:<10.2f} | x{speedup:<11.2f}")

    # --- ГЕНЕРАЦІЯ ГРАФІКА ---
    # --- ГЕНЕРАЦІЯ ГРАФІКА ---
    plt.figure(figsize=(10, 6)) # type: ignore
    workers_x = list(results.keys())
    speedups_y = list(results.values())
    
    plt.plot(workers_x, speedups_y, marker='o', linestyle='-', color='b', label='Фактичне прискорення (ProcessPool)') # type: ignore
    plt.plot(workers_x, workers_x, linestyle='--', color='r', label='Ідеальне лінійне прискорення') # type: ignore
    
    plt.title('Паралельна індексація: Прискорення відносно кількості воркерів') # type: ignore
    plt.xlabel('Кількість воркерів (ядер)') # type: ignore
    plt.ylabel('Прискорення (Speedup)') # type: ignore
    plt.xticks(workers_x) # type: ignore
    plt.legend() # type: ignore
    plt.grid(True, linestyle=':', alpha=0.7) # type: ignore
    
    plot_path = "speedup_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight') # type: ignore
    print(f"\nГрафік успішно збережено у файлі: {plot_path}")
if __name__ == '__main__':
    main()

    