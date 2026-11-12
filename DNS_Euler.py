import math
import gc
import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np

a, L, T, courant = 1.0, 1.0, 2.0, 0.5

def exact(x, t):
    x = np.asarray(x)
    xi = ((x - a * t + L) % (2.0 * L)) - L

    return np.where((xi >= -0.5) & (xi <= 0.5), 1.0, 0.0)


def DNS(Nx, dt = None):
    dx = 2.0 * L / (Nx - 1)

    if dt is None:
        dt = courant * dx / a

    Nt = int(T / dt)
    coeff = a * dt / dx

    x = -L + np.arange(Nx) * dx
    u = exact(x, 0.0)


    for _ in range(Nt):
        u = u - coeff * (u - np.roll(u, 1)) #получ левых соседей т.е. возвр [u3 u0 u1 u2]


    u_ex = exact(x, Nt * dt)
    max_err = np.max(np.abs(u - u_ex))
    return x, u, u_ex, max_err

def measure_time(Nx = 2**19, repeats=1):
    # ЗАМЕР ВРЕМЕНИ
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        DNS(Nx)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    # del x, u_num, u_exact
    # gc.collect()
    mean_time = float(np.mean(timings))

    #ЗАМЕР ПАМЯТИ
    tracemalloc.start()
    tracemalloc.reset_peak()
    x, u_num, u_exact, max_err = DNS(Nx)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()



    print(f"Сетка Nx: {Nx}")
    print(f"Количество запусков: {repeats}")
    print(f"Время выполнения: {elapsed:.4f} с")
    print(f"Память в момент завершения: {current / 1024 ** 2:.2f} МБ")
    print(f"Пиковое потребление памяти:  {peak / 1024 ** 2:.2f} МБ")
    print(f"Макс. ошибка: {max_err:.6e}\n")

    return mean_time, timings, max_err


def main():
    measure_time()
    #график для Nx = ....
    '''x, u_num, u_an, _ = DNS(2**10)
    plt.figure(figsize=(8, 5))
    plt.plot(x, u_an, "r--", linewidth=2, label="Аналитическое")
    plt.plot(x, u_num, "b-", linewidth=2, label="Численное (Nx=1024)")
    plt.title(f"Сравнение решений при t = {T}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.grid(True)
    plt.legend()
    plt.show()'''


if __name__ == "__main__":
    main()