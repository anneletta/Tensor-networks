import math
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
import ttnumpy as tt

a = 1.0
L = 1.0
T = 0.5
courant = 0.5

def exact(x: float, t: float) -> float:
    xi = x - a * t
    return float((np.cos(np.pi * xi)) ** 2) if -0.5 <= xi <= 0.5 else 0.0


def MPO(Nx: int, d: int, coeff: float) -> tt.TensorTrain:
    M_dense = np.zeros((Nx, Nx), dtype=float)
    for m in range(1, Nx):
        M_dense[m, m] = 1.0 - coeff
        M_dense[m, m - 1] = coeff
    M_tensor = M_dense.reshape([2] * (2 * d))

    pairs = [(2, 2)] * d

    M_tt = tt.tt_svd(M_tensor, error=1e-12, physical_indexes=pairs, ttm=True)
    return M_tt


def MPS(d: int, trunc_tol: float = 1e-8):
    Nx = 2**d
    dx = 2.0 * L / (Nx - 1)
    dt = courant * dx / a
    Nt = int(T / dt)
    coeff = a * dt / dx

    tracemalloc.start()
    t_start = time.perf_counter()

    x = np.array([-L + i * dx for i in range(Nx)], dtype=float)
    u_init_dense = np.array([exact(xi, 0.0) for xi in x], dtype=float)

    u_tensor = u_init_dense.reshape([2] * d) # нач усл вектор -> тензор
    u_tt = tt.tt_svd(u_tensor, error=1e-10)

    M_tt = MPO(Nx, d, coeff)

    print(f"d: {d}, узлов Nx: {Nx}")
    print(f"TT-ранги оператора перехода M: {M_tt.bonds}")
    print(f"Начальные TT-ранги состояния u: {u_tt.bonds}")
    #сразу фиксируем бонд для его отслеживания
    max_ranks = [max(u_tt.bonds)]
    for _ in range(Nt):
        u_tt = M_tt.dot(u_tt)
        u_tt = u_tt.truncate(error=trunc_tol)

        max_ranks.append(int(max(u_tt.bonds))) #бонд замер

    print(f"Конечные TT-ранги состояния u: {u_tt.bonds}")

    elapsed_time = time.perf_counter() - t_start
    tracemalloc.stop()

    # свертываем решение в вектор
    u_final_tensor = u_tt.get_full_tensor()
    u_num = u_final_tensor.reshape(-1)

    # аналит реш при t = Nt * dt
    u_ex = np.array([exact(xi, Nt * dt) for xi in x], dtype=float)
    max_err = np.max(np.abs(u_num - u_ex))
    stats = {
        "time_sec": elapsed_time,
        "bond_times": np.arange(Nt + 1) * dt,
        "max_bonds": np.array(max_ranks, dtype=int)
    }

    return x, u_num, u_ex, max_err, stats


def max_bond(stats, d):
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

    return fig, ax

def main():
    d = 10
    x, u_num, u_ex, err, stats = MPS(d=d, trunc_tol=1e-7)

    print("=" * 45)
    print(f"Максимальная ошибка: {err:.6e}")
    print(f"Время расчета:                  {stats['time_sec']:.4f} сек")
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
    main()