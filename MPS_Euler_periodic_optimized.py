import math
import statistics

import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
import ttnumpy as tt
import gc

a = 1.0
L = 1.0
T = 2.0
courant = 0.5

def exact(x: float, t: float) -> float:
    x = np.asarray(x)
    xi = ((x - a * t + L) % (2.0 * L)) - L

    return np.where((xi >= -0.5) & (xi <= 0.5), 1.0, 0.0)


def MPO(Nx: int, d: int, coeff: float) -> tt.TensorTrain:
    c = float(coeff)
    I = np.array([[1.0, 0.0], [0.0, 1.0],])
    J = np.array([[0.0, 1.0],[0.0, 0.0],])
    JT = J.T.copy()
    cores = []

    # первое ядро: G1 = [I, c * (JT + J)], форма: (1, 2, 2, 2)
    G1 = np.zeros((1, 2, 2, 2), dtype=float)
    G1[0, :, :, 0] = I
    G1[0, :, :, 1] = c * (JT + J)
    cores.append(G1)

    # промежут ядра: Gk = [[I, c * JT], [0, J]], форма: (2, 2, 2, 2)
    for _ in range(d - 2):
        Gk = np.zeros((2, 2, 2, 2), dtype=float)
        Gk[0, :, :, 0] = I
        Gk[0, :, :, 1] = c * JT
        Gk[1, :, :, 1] = J
        cores.append(Gk)

    # последнее ядро: Gd = [[(1 - c) * I + c * JT], форма: (2, 2, 2, 1)
    Gd = np.zeros((2, 2, 2, 1), dtype=float)
    Gd[0, :, :, 0] = (1.0 - c) * I + c * JT
    Gd[1, :, :, 0] = J
    cores.append(Gd)

    return tt.TensorTrain(cores)


def MPS(d: int, trunc_tol: float = 1e-8, operator_max_rank = None):
    Nx = 2**d
    dx = 2.0 * L / Nx

    dt_limit = courant * dx / a
    Nt = math.ceil(T / dt_limit)
    dt = T / Nt
    coeff = a * dt / dx

    x = np.array([-L + i * dx for i in range(Nx)], dtype=float)
    u_init_dense = np.array([exact(xi, 0.0) for xi in x], dtype=float)

    u_tensor = u_init_dense.reshape([2] * d) # нач усл вектор -> тензор
    u_tt = tt.tt_svd(u_tensor, error=1e-10)

    M_tt = MPO(Nx, d, coeff)

    #print(f"d: {d}, узлов Nx: {Nx}")
    #print(f"TT-ранги оператора перехода M: {M_tt.bonds}")
    #print(f"Начальные TT-ранги состояния u: {u_tt.bonds}")
    #сразу фиксируем бонд для его отслеживания
    P_tt = M_tt
    #max_ranks = [max(u_tt.bonds)] #бонд состояний
    operator_max_bonds = [int(max(P_tt.bonds))] #бонд оператора
    if operator_max_rank is not None:
        P_tt = P_tt.truncate(error = trunc_tol, max_rank = operator_max_rank)
    for _ in range(d+1):
        P_tt = P_tt.dot(P_tt)
        P_tt = P_tt.truncate(error=trunc_tol, max_rank = operator_max_rank) # !ВОЗМОЖНО СТОИТ ЗАМЕНИТЬ ТОЧНОСТЬ НА ОПЕР

        operator_max_bonds.append(int(max(P_tt.bonds))) #бонд замер

    u_tt = P_tt.dot(u_tt)
    u_tt = u_tt.truncate(error=trunc_tol)
    #print(f"Конечные TT-ранги состояния u: {u_tt.bonds}")

    # свертываем решение в вектор
    u_final_tensor = u_tt.get_full_tensor()
    u_num = u_final_tensor.reshape(-1)

    # аналит реш при t = Nt * dt
    u_ex = np.array([exact(xi, Nt * dt) for xi in x], dtype=float)
    max_err = np.max(np.abs(u_num - u_ex))
    stats = {
        "squaring_steps": np.arange(len(operator_max_bonds)),
        "max_bonds": np.array(operator_max_bonds, dtype=int)
    }

    return x, u_num, u_ex, max_err, stats

# ограничение ранга НА ОПЕРАТОР
def compare_operator_ranks():
    d = 13
    tol = 1e-7
    x, u_base, u_ex, err, stats = MPS(d=d, trunc_tol=tol)
    rmse_base = np.sqrt(np.mean((u_base - u_ex) ** 2)) #ош опорного реш
    print(f"RMSE без ограничения рангов: {rmse_base:.6e}")
    R_max = int(stats["max_bonds"].max())
    print(f"Максимальный ранг базового расчёта: {R_max}")



    # Огр по R_max - 1, R_max - 2, ..., 1.
    for r in range(R_max - 1, 0, -1):
        _, u_num, _, _, stats_r = MPS(
            d=d,
            trunc_tol=tol,
            operator_max_rank=r,
        )
        rmse = np.sqrt(np.mean((u_num - u_ex) ** 2))
        actual_rank = int(stats_r["max_bonds"].max())
        plt.figure(figsize=(9, 5))

        plt.plot(
            x, u_ex,
            "k--",
            linewidth=2,
            label="Аналитическое",
        )

        plt.plot(
            x, u_base,
            color="gray",
            linewidth=1.5,
            label="Без ограничения рангов",
        )

        plt.plot(
            x, u_num,
            color="tab:blue",
            linewidth=1.5,
            label=f"Ограничение r ≤ {r}",
        )

        plt.title(
            f"Ограничение рангов оператора: r ≤ {r}\n"
            f"d={d}, Nx={2 ** d}, T={T}, RMSE_ошибка={rmse:.3e}"
        )
        plt.xlabel("x")
        plt.ylabel("u(x, T)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()


        print(
            f"Ограничение r ≤ {r}: "
            f"максимальный фактический ранг = {actual_rank}, "
            f"RMSE_ошибка = {rmse:.6e}"
        )

    plt.show()

# оганичение ранга НА СОСТОЯНИЕ
'''def max_bond(stats, d):
    times = stats["bond_times"]
    max_bonds = stats["max_bonds"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.step(
        times,
        max_bonds,
        where="post",
        linewidth=2,
        color="darkblue",
        label=r"$r_{\max}(t)$"
    )

    ax.set_title(
        f"Динамика максимального бонда: d = {d}, Nx = {2**d}"
    )
    ax.set_xlabel("Время t")
    ax.set_ylabel("Максимальный бонд (после сжатия)")

    ax.set_yticks(
        np.arange(max_bonds.min(), max_bonds.max() + 1)
    )

    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    return fig, ax'''

# ГРАФИК ДИНАМИКИ РАНГА ОПЕРАТОРА
def max_bond(stats, d):
    steps = stats["squaring_steps"]
    max_bonds = stats["max_bonds"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.step(
        steps,
        max_bonds,
        where="post",
        linewidth=2,
        color="darkblue",
        label=r"$r_{\max}(M^{2^k})$"
    )

    ax.plot(
        steps,
        max_bonds,
        "",
        color="darkblue"
    )

    ax.set_title(
        f"Динамика максимального бонда оператора: "
        f"d = {d}, Nx = {2**d}"
    )
    ax.set_xlabel("Число возведений в квадрат k")
    ax.set_ylabel("Максимальный бонд оператора после сжатия")

    ax.set_xticks(steps)
    ax.set_yticks(
        np.arange(max_bonds.min(), max_bonds.max() + 1)
    )

    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()

    return fig, ax

'''def compare_ranks():
    d = 10
    # здесь ошибки подобраны исходя из ранга которым хочу ограничить
    tolerances = [1e-7, 1e-3, 3e-2, 1e-1]
    labels = ["r≈7", "r≈5", "r≈3", "r≈2"]

    plt.figure(figsize=(10, 6))

    for tol, label in zip(tolerances, labels):
        x, u_num, u_ex, err, stats = MPS(d=d, trunc_tol=tol)
        actual_max_r = stats["max_bonds"].max()
        l2_err = np.sqrt(np.mean((u_num - u_ex) ** 2))
        plt.plot(x, u_num, label=f"{label}, l2_err={l2_err:.3f}")


    plt.plot(x, u_ex, "k--", alpha=0.5, label="Аналитическое")

    plt.title(f"Угудшение решения при ограничении ранга (Nx = 2^{d})")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()'''

# ЗАМЕРЫ
def measure_algorithm(func, *args, repeats=5, **kwargs):
    times = []
    memories = []
    result = None
    # ВРЕМЯ
    for i in range (repeats):
        gc.collect()
        t_start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - t_start

        if i==0:
            result = res
        times.append(elapsed_time)
    # ПАМЯТЬ
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()

    _ = func(*args, **kwargs)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memories.append(peak_bytes / (1024 * 1024))

    avg_time = statistics.mean(times)
    avg_memory = statistics.mean(memories)
    return result, avg_time, avg_memory


def main():
    d = 20
    '''(x, u_num, u_ex, err, stats), elapsed_time, peak_mb = measure_algorithm(MPS, d=d, trunc_tol=1e-7)
    print(f"Сетка: d = {d} (число узлов Nx = {2 ** d})")
    print(f"Макс ош: {err:.6e}")
    print(f"Время алгоритма: {elapsed_time:.4f} сек")
    print(f"Пиковая память: {peak_mb:.2f} МБ")'''
    x, u_num, u_ex, err, stats = MPS(d=d, trunc_tol=1e-7)

    print("=" * 45)
    print(f"Максимальная ошибка: {err:.6e}")
    #print(f"Время расчета:                  {stats['time_sec']:.4f} сек")
    print("=" * 45)


    plt.figure(figsize=(8, 5))
    plt.plot(x, u_ex, "r--", linewidth=2, label="Аналитическое")
    plt.plot(x, u_num, "b-", linewidth=2, label=f"MPS (Nx={2**d})")
    plt.title(f"Сравнение решений при t = {T}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.grid(True)
    plt.legend()


    max_bond(stats, d)
    plt.show()
if __name__ == "__main__":
    compare_operator_ranks()
    #compare_ranks()
    #main()