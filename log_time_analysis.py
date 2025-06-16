import re
import statistics
import matplotlib.pyplot as plt

log_file = "./test_data/translated_Dataset/report._log.log"

# 1. Load the log
with open(log_file, 'r', encoding='utf-8') as f:
    log_text = f.read()

# 2. Extract all total‐time values
pattern = re.compile(r'total time:\s*([\d\.]+)s')
times = [float(m.group(1)) for m in pattern.finditer(log_text)]

if not times:
    print("No total-time entries found.")
    exit()

# 3. Quick statistics
count = len(times)
minimum = min(times)
maximum = max(times)
mean = statistics.mean(times)
median = statistics.median(times)
stdev = statistics.pstdev(times)

print(f"Found {count} entries.")
print(f"Min time: {minimum:.2f}s")
print(f"Max time: {maximum:.2f}s")
print(f"Average time: {mean:.2f}s")
print(f"Median time: {median:.2f}s")
print(f"Std. deviation: {stdev:.2f}s")

# 4. Histogram
plt.figure(figsize=(8, 4))
plt.hist(times, bins=10)
plt.xlabel('Total time (s)')
plt.ylabel('Frequency')
plt.title('Distribution of Processing Times')
plt.tight_layout()
plt.show()
