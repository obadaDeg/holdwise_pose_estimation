from collections import Counter
import matplotlib.pyplot as plt
import sympy

accidents = [4, 2, 4, 3, 11, 6, 4, 4, 1, 4]
freq_table = Counter(accidents)

print("Frequency Table (accidents : count):")
for k in sorted(freq_table):
    print(f"{k} : {freq_table[k]}")


# Extract the distinct accident counts and frequencies
values = sorted(freq_table.keys())
frequencies = [freq_table[v] for v in values]

plt.bar(values, frequencies, color="skyblue", edgecolor="black")
plt.xlabel("Number of Accidents")
plt.ylabel("Frequency")
plt.title("Frequency of Yearly Airline Accidents (1985–1994)")
plt.show()


accidents_mean = sympy.Rational(sum(accidents), len(accidents))  # exact rational mean
accidents_mean_float = float(accidents_mean)  # approximate decimal

print("Sample mean (exact) =", accidents_mean)
print("Sample mean (float) =", accidents_mean_float)

n = len(accidents)
mean_val = float(accidents_mean)  # 4.3
variance_num = sum((x - mean_val) ** 2 for x in accidents)
sample_variance = variance_num / (n - 1)

print("Sample variance =", sample_variance)

# We already have freq_table from Counter
sample_mode = max(freq_table, key=freq_table.get)
print("Sample mode =", sample_mode)
