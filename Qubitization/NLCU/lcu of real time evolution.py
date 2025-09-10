import numpy as np
import matplotlib.pyplot as plt

def fx_with_shifted_phase(x, N):
    """
    f(x) = (1/N) ∑_{j=N+1}^{2N} e^{-ixj}
    """
    j_vals = np.arange(N + 1, 2 * N + 1)
    #return (1 / N) * np.sum(np.exp(-1j * x * j_vals))
    return sum(np.exp(-1j * x * i) for i in range(-N, N + 1))/(2*N+1)
    #return sum(np.exp(-1j * x * i) for i in range(-N, N + 1))

# 参数设置
N = 1
x_vals = np.linspace(-10, 10, 500)
fx_vals = np.array([fx_with_shifted_phase(x, N) for x in x_vals])

# 提取三部分
real_vals = fx_vals.real
imag_vals = fx_vals.imag
abs_vals = np.abs(fx_vals)

# 画图
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(x_vals, real_vals, label="Re[f(x)]", color="blue")
axes[0].set_ylabel("Real")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x_vals, imag_vals, label="Im[f(x)]", color="red")
axes[1].set_ylabel("Imag")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(x_vals, abs_vals, label="|f(x)|", color="black")
axes[2].set_xlabel("x")
axes[2].set_ylabel("Magnitude")
axes[2].legend()
axes[2].grid(True)

#plt.suptitle(r"Components of $f(x) = \sum_{j=-N}^{N} e^{-ixj}$ (N = %d)" % N)
plt.suptitle(r"Components of $f(x) =  \frac{1}{2N+1}\sum_{j=-N}^{N} e^{-ixj}$ (N = %d)" % N)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
