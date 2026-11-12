import time
import numpy as np
import matplotlib.pyplot as plt

VARIANT = 21

print("ВИКОНАННЯ ЧАСТИНИ 1\n")

N_part1 = 10 + VARIANT  # N = 10 + 21 = 31
print(f"Розмірність вхідного вектора N = 10 + {VARIANT} = {N_part1}")

np.random.seed(42)
f_vector = np.random.rand(N_part1) * 10 

print("\nЗгенерований вхідний вектор f:")
print("-" * 75)
for idx, val in enumerate(f_vector):
    print(f"f[{idx:2d}] = {val:6.3f}  ", end="")
    if (idx + 1) % 4 == 0 or idx == N_part1 - 1:
        print()
print("-" * 75)

def compute_dft(f):
    N = len(f)
    C_k = np.zeros(N, dtype=complex)
    A_k = np.zeros(N)
    B_k = np.zeros(N)

    mult_count = 0
    add_count = 0

    start_time = time.perf_counter()
    inv_N = 1.0 / N

    for k in range(N):
        sum_real = 0.0
        sum_imag = 0.0

        for i in range(N):
            angle = (2.0 * np.pi / N) * k * i

            term_cos = f[i] * np.cos(angle)
            term_sin = f[i] * np.sin(angle)
            mult_count += 2
        
            sum_real += term_cos
            sum_imag -= term_sin
            if i>0: 
                add_count += 2


        A_k[k] = sum_real * inv_N
        B_k[k] = sum_imag * inv_N

        C_k[k] = complex(A_k[k], B_k[k])

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    return C_k, A_k, B_k, execution_time, mult_count, add_count

C_k, A_k, B_k, execution_time, mult_count, add_count = compute_dft(f_vector)

amplitudes = np.abs(C_k)
phases = np.arctan2(C_k.imag, C_k.real)

print("\nОбчислені компоненти та коефіцієнти C_k = A_k + j*B_k:")
print("-" * 75)
print(f"{'k':^4} | {'A_k (Re)':^12} | {'B_k (Im)':^12} | {'|C_k| (модуль)':^16} | {'arg C_k (фаза)':^18}")
print("-" * 75)
for k in range(N_part1):
    print(f"{k:^4} | {A_k[k]:^+12.4f} | {B_k[k]:^+12.4f} | {amplitudes[k]:^16.4f} | {phases[k]:^+18.4f}")
print("-" * 75)

print(f"\nЧас обчислення ДПФ: {execution_time * 1000:.4f} мс")
print(f"Кількість операцій множення: {mult_count}")
print(f"Кількість операцій додавання: {add_count}")

# Графіки спектрів
plt.figure(figsize=(12, 5))
plt.suptitle(f"Частина 1: Спектральний аналіз ДПФ (Варіант {VARIANT}, N={N_part1})", fontsize=13)

plt.subplot(1, 2, 1)
plt.stem(range(N_part1), amplitudes, basefmt="k-")
plt.title("Спектр амплітуд |C_k|")
plt.xlabel("k")
plt.ylabel("|C_k|")
plt.grid(True, linestyle=":", alpha=0.6)

plt.subplot(1, 2, 2)
plt.stem(range(N_part1), phases, basefmt="k-")
plt.title("Спектр фаз arg(C_k)")
plt.xlabel("k")
plt.ylabel("Радіани")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

print("\n")
print("ВИКОНАННЯ ЧАСТИНИ 2\n")

N_val = 96 + VARIANT  # 96 + 21 = 117
binary_str = format(N_val, '08b')
s_samples = np.array([float(bit) for bit in binary_str])

if VARIANT % 2 != 0:
    s_samples[0] = 1.0
else:
    s_samples[0] = 0.0

binary_modified = "".join([str(int(b)) for b in s_samples])

print(f"Число N = 96 + {VARIANT} = {N_val} | Двійкове (8 біт) - до заміни: {binary_str} | Двійкове після заміни: {binary_modified}")
print(f"Згенеровані відліки s(nT_delta): {s_samples.astype(int).tolist()}")

N_8 = len(s_samples)
C_k_8 = np.zeros(N_8, dtype=complex)

for k in range(N_8):
    sum_val = 0.0
    for n_idx in range(N_8):
        angle = -2 * np.pi * k * n_idx / N_8
        sum_val += s_samples[n_idx] * np.exp(1j * angle)
    C_k_8[k] = (1 / N_8) * sum_val

print("\nКоефіцієнти C_k для N=8 (перші N/2 + 1 гармонік):")
print("-" * 75)
for k in range(N_8 // 2 + 1):
    print(f"  C_{k} = {C_k_8[k].real:+.4f} {C_k_8[k].imag:+.4f}j | |C_{k}| = {np.abs(C_k_8[k]):.4f} | arg C_{k} = {np.angle(C_k_8[k]):+.4f} рад")
print("-" * 75)

t_vals = np.linspace(0, 1, 400)
def reconstructed_signal(t):
    res = np.abs(C_k_8[0])
    # k = 1, 2, 3 (множник 2)
    for k in range(1, N_8 // 2):
        amplitude = 2 * np.abs(C_k_8[k])
        phase = np.angle(C_k_8[k])
        res += amplitude * np.cos(2 * np.pi * k * t + phase)
    # k = 4 (N/2, множник 1)
    k_last = N_8 // 2
    res += np.abs(C_k_8[k_last]) * np.cos(2 * np.pi * k_last * t + np.angle(C_k_8[k_last]))
    return res

s_reconstructed = reconstructed_signal(t_vals)

plt.figure(figsize=(10, 4.5))
plt.plot(t_vals, s_reconstructed, label="Відтворений сигнал s", color="#0b559f", linewidth=1.8)
sample_points = np.arange(8) / 8.0
plt.scatter(sample_points, s_samples, color="#e6550d", s=50, zorder=5, label="Початкові відліки s(nT_δ)")
plt.title(f"Часова залежність відтвореного сигналу s(t) (Частина 2, Варіант {VARIANT})", fontsize=12)
plt.xlabel("t / T_c")
plt.ylabel("s(t)")
plt.xlim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

print("\n")
print("ВИКОНАННЯ ЧАСТИНИ 3 (ОДПФ)\n")

def inverse_dft(C):
    N = len(C)
    s_recovered = np.zeros(N)
    for n_idx in range(N):
        sum_val = 0.0
        for k in range(N):
            angle = 2 * np.pi * k * n_idx / N
            sum_val += (C[k] * np.exp(1j * angle)).real
        s_recovered[n_idx] = sum_val
    return s_recovered

s_inv_recovered = inverse_dft(C_k_8)

print(f"{'n':^4} | {'Початковий відлік':^18} | {'Відновлений через ОДПФ':^24} | {'Похибка':^12}")
print("-" * 65)
for n_idx in range(N_8):
    err = abs(s_samples[n_idx] - s_inv_recovered[n_idx])
    print(f"{n_idx:^4} | {s_samples[n_idx]:^18.1f} | {s_inv_recovered[n_idx]:^24.4f} | {err:^12.2e}")
print("-" * 65)