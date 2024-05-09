
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".out")]

results = {}

for result_file_name in result_files:
    method, size = result_file_name.replace(".out", "").split('_')
    size = int(size.replace("KB", "000").replace("MB", "000000").replace("GB", "000000000").replace("GFLOPS", "000000000"))
    if method not in results:
        results[method] = {}

    result_file_path = os.path.join(RESULTS_DIR, result_file_name)
    with open(result_file_path) as result_file:
        result_file_contents = result_file.readlines()
        for line in result_file_contents:
            if line.startswith("Benchmarking "):
                experiment = line.replace("Benchmarking ", "").replace("*", "").replace("\n", "").replace("w/", "with")
                if experiment not in results[method]:
                    results[method][experiment] = {}
                if size not in results[method][experiment]:
                    results[method][experiment][size] = 0
            if line.startswith("    Throughput:    "):
                throughput = float(line.replace("    Throughput:    ", "").replace("GB/sec\n", ""))
                results[method][experiment][size] = throughput

for method, experiments in results.items():

    for experiment, sizes in experiments.items():
        sorted_sizes = sorted(sizes.keys())

        x = sorted_sizes
        y = [sizes[i] for i in sorted_sizes]

        plt.plot(x, y, label=experiment)
    plt.xlabel("data size (bytes)")
    plt.ylabel("Throughput (GB/s)")
    plt.title(f"{method}")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"{method}.png"))  
    plt.clf()
