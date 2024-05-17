import csv
import matplotlib.pyplot as plt
import os

from pathlib import Path

RESULTS_DIR = "results"
GRAPHS_DIR = "graphs"
THROUGHPUT = "throughput"
GFLOPS = "GFLOPS"
NAME = "name"
CLOCK = "clock"
BUS = "bus"
MEMORY = "memory"
DEVICES = "devices"

Path(os.path.join(RESULTS_DIR, GRAPHS_DIR)).mkdir(parents=True, exist_ok=True) 
result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".out")]

results = {}
devices = {}
device_counter = -1
longest_devices = -1

for result_file_name in result_files:
    method, device_count, size = result_file_name.replace(".out", "").split('_')
    #size = int(size.replace("KB", "000").replace("MB", "000000").replace("GB", "000000000").replace("GFLOPS", "000000000"))
    if method not in results:
        results[method] = {}

    result_file_path = os.path.join(RESULTS_DIR, result_file_name)
    with open(result_file_path) as result_file:
        result_file_contents = result_file.readlines()
        for line in result_file_contents:
            if line.startswith("Using device: "):
                device_counter = int(line.replace("Using device: ", "").replace("\n", ""))
                devices[device_counter] = {
                    NAME: None,
                    CLOCK: None,
                    BUS: None,
                    MEMORY: None
                }
            if device_counter != -1:
                if line.startswith("  Device name: "):
                    devices[device_counter][NAME] = line.replace("  Device name: ", "").replace("\n", "")
                elif line.startswith("  Memory Clock Rate (KHz): "):
                    devices[device_counter][CLOCK] = line.replace("  Memory Clock Rate (KHz): ", "").replace("\n", "")
                elif line.startswith("  Memory Bus Width (bits): "):
                    devices[device_counter][BUS] = line.replace("  Memory Bus Width (bits): ", "").replace("\n", "")
                elif line.startswith("  Peak Memory Bandwidth (GB/s): "):
                    devices[device_counter][MEMORY] = line.replace("  Peak Memory Bandwidth (GB/s): ", "").replace("\n", "")
                else:
                    device_counter == -1

            if line.startswith("Benchmarking "):
                experiment = line.replace("Benchmarking ", "").replace("*", "").replace("\n", "").replace("w/", "with").strip()
                if experiment not in results[method]:
                    results[method][experiment] = {}
                if size not in results[method][experiment]:
                    results[method][experiment][size] = {}
                if device_count not in results[method][experiment][size]:
                    results[method][experiment][size][device_count] = {
                        THROUGHPUT: 0,
                        GFLOPS: 0,
                        DEVICES: []
                    }

            if line.startswith("    GLFOPS:        "):
                gflops = float(line.replace("    GLFOPS:        ", "").replace("/sec\n", ""))
                results[method][experiment][size][device_count][GFLOPS] = gflops
            if line.startswith("    Throughput:    "):
                throughput = float(line.replace("    Throughput:    ", "").replace("GB/sec\n", ""))
                results[method][experiment][size][device_count][THROUGHPUT] = throughput

                devices_list = []
                sorted_devices = {key: value for key, value in sorted(devices.items())}
                for device, device_dict in sorted_devices.items():
                    devices_list.append(device_dict[NAME])
                    devices_list.append(device_dict[CLOCK])
                    devices_list.append(device_dict[BUS])
                    devices_list.append(device_dict[MEMORY])
                results[method][experiment][size][device_count][DEVICES] = devices_list
    if len(devices) > longest_devices:
        longest_devices = len(devices)
    devices = {}

with open(os.path.join(RESULTS_DIR, "results.csv"), 'w', newline="") as csvFile:
    writer = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    headings = ["method", "experiment", "size(GB)", "device count", "GFLOPS", "throughput(GB/s)"]
    for i in range(longest_devices):
        headings = headings + [f"device {i} name", f"device {i} memory clock rate (KHz)", f"device {i} memory bus width (bits)", f"device {i} peak memory bandwidth (GB/s)"]
    writer.writerow(headings)
    
    sorted_methods = {key: value for key, value in sorted(results.items())}
    for method, method_dict in sorted_methods.items():
        sorted_experiments = {key: value for key, value in sorted(method_dict.items())}
        for experiment, experiment_dict in sorted_experiments.items():
            sorted_sizes = {key: value for key, value in sorted(experiment_dict.items())}
            for size, size_dict in sorted_sizes.items():
                sorted_device_counts = {key: value for key, value in sorted(size_dict.items())}
                for device_count, device_count_dict in sorted_device_counts.items():
                    row = [method, experiment, size, device_count, device_count_dict[GFLOPS], device_count_dict[THROUGHPUT]]
                    if len(results[method][experiment][size][device_count][DEVICES]) > 0:
                        row = row + results[method][experiment][size][device_count][DEVICES]
                    writer.writerow(row)

for method, experiments in results.items():

    size_count = len(experiments[list(experiments.keys())[0]].keys())
    fig, axes = plt.subplots(size_count, 2, figsize=(5, size_count*2.5))
    title = f"{method}"

    for experiment, sizes in experiments.items():
        sorted_sizes = {key: value for key, value in sorted(sizes.items())}

        for i, size in enumerate(sorted_sizes): 
            device_counts = sorted_sizes[size]
            sorted_device_counts = {key: value for key, value in sorted(device_counts.items())}

            for device, base_dict in sorted_device_counts.items():
                devices = results[method][experiment][size][device_count][DEVICES]
                throughput = results[method][experiment][size][device_count][THROUGHPUT]


            x = sorted_device_counts.keys()
            y_throughput = [sorted_device_counts[k][THROUGHPUT] for k in sorted_device_counts]
            y_gflops = [sorted_device_counts[k][GFLOPS] for k in sorted_device_counts]

            axes[i][0].plot(x, y_throughput, label=f"{experiment}")
            axes[i][1].plot(x, y_gflops, label=f"{experiment}")
            
            axes[i][0].set_title(size)
            axes[i][0].set_xlabel("Devices")
            axes[i][0].set_ylabel("Throughput (GB/s)")

            axes[i][1].set_title(size)
            axes[i][1].set_xlabel("Devices")
            axes[i][1].set_ylabel("GFLOPS")

    fig.suptitle(title)
    fig.tight_layout()

    axes[0][1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.savefig(os.path.join(RESULTS_DIR, GRAPHS_DIR, f"{title}.png"), bbox_inches="tight")  
    plt.clf()
