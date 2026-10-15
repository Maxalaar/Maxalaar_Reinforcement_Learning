import os
import socket
import time
import webbrowser
import json
import subprocess
import sys
from multiprocessing import Process
from typing import List
import requests
from urllib.parse import quote, urlencode


def find_free_port() -> int:
    """Reserve an available TCP port so TensorBoard doesn't silently fall back to another one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_tensorboard(port: int):
    """Run TensorBoard as a subprocess with proper log directory."""
    path = os.getcwd()
    print(f"Current working directory: {path}")
    logdir = os.path.join(path, 'experiments')

    python_dir = os.path.dirname(sys.executable)
    tensorboard_bin = os.path.join(python_dir, 'tensorboard')

    # Start TensorBoard as a subprocess
    process = subprocess.Popen(
        [tensorboard_bin, '--bind_all', '--port', str(port), '--logdir', logdir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Stream output to console
    for line in process.stdout:
        print(line, end='')
    for line in process.stderr:
        print(line, end='', file=sys.stderr)


def list_to_url(list_monitoring: List[str], port: int) -> str:
    """Generate TensorBoard URL with pinned cards using proper encoding."""
    # Create card objects
    cards = [{"plugin": "scalars", "tag": tag} for tag in list_monitoring]

    # Build URL parameters
    params = {
        "pinnedCards": json.dumps(cards, separators=(',', ':')),
        "darkMode": "true"
    }

    # Encode and construct URL
    encoded_params = urlencode(params, quote_via=quote, safe="[]{}")
    return f"http://localhost:{port}/?{encoded_params}#timeseries"


def wait_for_tensorboard(port: int, host: str = 'http://localhost', timeout: int = 600):
    """Wait for TensorBoard to start, handling redirects properly."""
    url = f"{host}:{port}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, allow_redirects=True, timeout=2)
            if response.status_code < 400:  # Accept any successful status
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)

    raise TimeoutError(f"TensorBoard did not start within {timeout} seconds.")


if __name__ == "__main__":
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'

    # Define dashboard configurations
    list_simulation = [
        'ray/tune/perf/cpu_util_percent',
        'ray/tune/perf/gpu_util_percent0',
        'ray/tune/perf/ram_util_percent',
        'ray/tune/perf/vram_util_percent0',
        'ray/tune/env_runners/episode_duration_sec_mean',
        'ray/tune/timers/evaluation_iteration_time_sec',
        'ray/tune/timers/training_iteration_time_sec',
        'ray/tune/memory_usage_mb/driver',
        'ray/tune/memory_usage_mb/env_runners_mean',
        'ray/tune/memory_usage_mb/env_runners_max',
        'ray/tune/memory_usage_mb/env_runners_godot_mean',
        'ray/tune/memory_usage_mb/env_runners_godot_max',
        'ray/tune/memory_usage_mb/learners_mean',
        'ray/tune/memory_usage_mb/learners_max',
        'ray/tune/memory_usage_mb/learners_cuda_allocated_mean',
        'ray/tune/memory_usage_mb/learners_cuda_reserved_mean',
        'ray/tune/memory_usage_diagnostics/learners_live_tensor_count_mean',
    ]

    list_learning = [
        'ray/tune/evaluation/env_runners/episode_return_mean',
        'ray/tune/env_runners/episode_return_mean',
        'ray/tune/learners/default_policy/gradients_default_optimizer_global_norm',
        'ray/tune/env_runners/episode_len_mean',
        'ray/tune/num_agent_steps_sampled_lifetime/default_agent',
        'ray/tune/learners/default_policy/policy_loss',
        'ray/tune/learners/default_policy/vf_loss',
        'ray/tune/learners/default_policy/vf_loss_unclipped',
        'ray/tune/learners/default_policy/mean_kl_loss',
        'ray/tune/learners/default_policy/entropy',
        'ray/tune/learners/default_policy/td_error_mean',
    ]

    list_supervised = [
        'pytorch_lightning/-total_loss_train_epoch',
        'pytorch_lightning/-action_loss_train_epoch',
        'pytorch_lightning/-clustering_loss_train_epoch',
        'pytorch_lightning/-centroid_pairwise_min_epoch',
        'pytorch_lightning/-embedding_centroid_mean_epoch',
        'pytorch_lightning/-min_cluster_ratio_epoch',
        'pytorch_lightning/-max_cluster_ratio_epoch',
        'pytorch_lightning/-silhouette_score_epoch',
    ]

    # Reserve a free port up front so a stray process squatting on the default
    # TensorBoard port (e.g. another app already listening on 6006) can't cause
    # this script to silently poll the wrong port and hang.
    port = find_free_port()
    print(f"Using port {port} for TensorBoard.")

    # Start TensorBoard in a separate process
    process = Process(target=run_tensorboard, args=(port,))
    process.daemon = True  # Terminate with main process
    process.start()

    try:
        print("Waiting for TensorBoard to start...")
        wait_for_tensorboard(port)

        # Open dashboards in browser
        print("Opening monitoring dashboards...")
        webbrowser.open_new('http://127.0.0.1:8265')  # Ray dashboard
        webbrowser.open_new_tab(list_to_url(list_simulation, port))
        webbrowser.open_new_tab(list_to_url(list_supervised, port))
        webbrowser.open_new_tab(list_to_url(list_learning, port))

        print("Dashboards opened successfully. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)  # Keep main process alive

    except (TimeoutError, Exception) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nTerminating TensorBoard process...")
        sys.exit(0)