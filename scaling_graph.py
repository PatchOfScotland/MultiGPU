
import matplotlib.pyplot as plt
import statistics
import sys
import os

scaling_dir = sys.argv[1]

data_sizes = []
gigaflops = []
all_results = os.listdir(scaling_dir)
all_results.sort()
for f in all_results:
    n = int(f[f.index('_')+1:f.index('.')])
    gb = n*n*3*4/1e9
    ops = n*n*n*2/1e9
    with open(os.path.join(scaling_dir, f), "r") as result_file:
        lines = result_file.readlines()
        timings = []
        for line in lines:
            try:
                result = float(line)
                timings.append(result)
            except ValueError:
                pass
        avg_s = statistics.mean(timings) / 1e6
        gf = ops / avg_s
        print(f"Timings for {n}x{n} ({gb} GB, {gf} Gflops): {avg_s}")
        data_sizes.append(gb)
        gigaflops.append(gf)

plt.plot(data_sizes, gigaflops)
plt.title("Throughput Scalability")
plt.ylabel("Throughput (GFLOPS)")
plt.xlabel("Total data size (GB)")
plt.savefig("result.png")
