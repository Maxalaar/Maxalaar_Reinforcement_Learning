import psutil
import GPUtil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from pathlib import Path


if __name__ == '__main__':
    # =========================
    # Configuration
    # =========================
    UPDATE_INTERVAL_MS = 100      # Update every 1 second
    MAX_POINTS = 500                # Last 60 seconds in graph

    # Desktop path (cross-platform)
    HOME_PATH = Path.home()
    OUTPUT_DIR = HOME_PATH / "system_monitor_logs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # Data buffers
    # =========================
    cpu_usage = deque(maxlen=MAX_POINTS)
    ram_usage = deque(maxlen=MAX_POINTS)
    gpu_usage = deque(maxlen=MAX_POINTS)
    gpu_memory = deque(maxlen=MAX_POINTS)

    # =========================
    # Matplotlib setup
    # =========================
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axs = axs.flatten()

    # =========================
    # GPU stats
    # =========================
    def get_gpu_stats():
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0.0, 0.0
        gpu = gpus[0]
        return gpu.load * 100.0, gpu.memoryUtil * 100.0

    # =========================
    # Update function
    # =========================
    def update(frame):
        # --- Collect stats ---
        cpu_usage.append(psutil.cpu_percent())
        ram_usage.append(psutil.virtual_memory().percent)

        gpu, gpu_mem = get_gpu_stats()
        gpu_usage.append(gpu)
        gpu_memory.append(gpu_mem)

        # --- Clear plots ---
        for ax in axs:
            ax.cla()

        # --- CPU ---
        axs[0].plot(cpu_usage)
        axs[0].set_title("CPU Usage (%)")
        axs[0].set_ylim(0, 100)

        # --- RAM ---
        axs[1].plot(ram_usage)
        axs[1].set_title("RAM Usage (%)")
        axs[1].set_ylim(0, 100)

        # --- GPU ---
        axs[2].plot(gpu_usage)
        axs[2].set_title("GPU Usage (%)")
        axs[2].set_ylim(0, 100)

        # --- GPU Memory ---
        axs[3].plot(gpu_memory)
        axs[3].set_title("GPU Memory Usage (%)")
        axs[3].set_ylim(0, 100)

        plt.tight_layout()

        # --- Save snapshot to disk ---
        filename = OUTPUT_DIR / f"system_monitor.png"
        plt.savefig(filename)

    # =========================
    # Animation loop
    # =========================
    ani = animation.FuncAnimation(
        fig,
        update,
        interval=UPDATE_INTERVAL_MS,
        cache_frame_data = False,
    )

    plt.show()
