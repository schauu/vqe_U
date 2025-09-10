import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def fx(x, N):
    """
    f(x) = (1 / (2N + 1)) ∑_{j=-N}^{N} e^{-ixj}
    """
    j_vals = np.arange(-N, N + 1)
    return np.sum(np.exp(-1j * x * j_vals)) / (2 * N + 1)


# x 取值范围
x_vals = np.linspace(-10, 10, 500)

# 设置绘图
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-10, 10)
ax.set_xticks(np.arange(-10, 11, 1))
ax.set_ylim(0, 1.1)
ax.set_xlabel("x")
ax.set_ylabel(r"$|f(x)|$")
ax.grid(True)

artists = []

# 不同的 N 值
N_vals = range(1, 60 + 1)  # 可以调整帧数
for N in N_vals:
    fx_vals = np.array([fx(x, N) for x in x_vals])
    #fx_vals = fx(1, N)
    abs_vals = np.abs(fx_vals)

    # 绘制当前帧的 artist（注意 plot 返回的是一个 list）
    line, = ax.plot(x_vals, abs_vals, color='black', label=f'N = {N}')
    text = ax.text(0.02, 0.95, f"N = {N}", transform=ax.transAxes)
    artists.append([line, text])

# 创建动画
ani = animation.ArtistAnimation(fig, artists, interval=200)

plt.tight_layout()
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# def fx(x, N):
#     j_vals = np.arange(-N, N + 1)
#     return np.sum(np.exp(-1j * x * j_vals)) / (2 * N + 1)
#
# # 参数设置
# x_fixed = 1.0
# N_max = 20
# N_vals = np.arange(1, N_max + 1)
# amplitudes = [1 - np.abs(fx(x_fixed, N)) for N in N_vals]
#
# # 初始化图
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.set_xlim(0, N_max + 1)
# ax.set_ylim(0, 1.1)
# ax.set_xlabel("N")
# ax.set_ylabel(r"$|f(1)|$")
# ax.set_title(r"Amplitude $|f(1)|$ vs $N$ for $f(x)=\frac{1}{2N+1}\sum_{j=-N}^N e^{-ixj}$")
# line, = ax.plot([], [], 'o-', color='blue')
# text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
#
# # 动画更新函数
# def update(frame):
#     xdata = N_vals[:frame+1]
#     ydata = amplitudes[:frame+1]
#     line.set_data(xdata, ydata)
#     text.set_text(f"N = {N_vals[frame]}\n|f(1)| = {amplitudes[frame]:.4f}")
#     return line, text
#
# # 创建动画
# ani = animation.FuncAnimation(fig, update, frames=len(N_vals), interval=100, blit=True)
#
# plt.tight_layout()

ani.save('fx.mp4', writer='ffmpeg', dpi=400, fps=30)