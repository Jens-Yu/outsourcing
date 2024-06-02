import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
dt = 0.01  # 采样时间
T = 1500  # 总时间
num_steps = int(T / dt)  # 时间步数量
t = np.linspace(0, T, num_steps)  # 时间向量

# 初始化增益值
a = np.array([0.0867, 0.2917, 0.5413, 0.7851, 0.9818, 1.1166, 1.1882, 1.2059, 1.1846, 1.1407,
              1.0886, 1.0394, 1.001, 0.9738, 0.9603, 0.9577, 0.9627, 0.9721, 0.9829, 0.9929]) * 0.31

A = np.zeros((20, 20))
for i in range(20):
    for j in range(i + 1):
        A[i, j] = a[i - j]

q = np.array([0.5, 0.4, 0.3, 0.3, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004,
              0.002, 0.001, 0.001, 0.001, 0.001, 0.001]) * 10.5
r = np.array([0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004,
              0.002, 0.001, 0.001, 0.001, 0.001, 0.001]) * 40
Q = np.diag(q)
R = np.diag(r)
c = np.zeros(20)
c[0] = 1
c1 = c.T
dt_K = np.dot(np.dot(c1, np.linalg.inv(np.dot(A.T, np.dot(Q, A)) + R)), np.dot(A.T, Q))

h = np.ones(20)
v = np.ones(19)
S = np.diag(v, 1)
S[19, 19] = 1

# 初始化输入信号和输出信号
input1 = np.zeros(num_steps)  # 输入1，始终为0
input2 = np.ones(num_steps) * -1  # 输入2，假设为-1的常数
u = np.zeros(num_steps)
intermediate_signal = np.zeros(num_steps)
y = np.zeros(num_steps)
y_final = np.zeros(num_steps)

# 初始化fcn函数所需的持久变量
fcn_x = np.ones(20)
fcn_N = 0

# 更新的fcn函数
def fcn(u):
    global fcn_x, fcn_N
    if fcn_N == 0:
        fcn_N = 1

    fcn_x[fcn_N - 1] = u
    fcn_N += 1
    if fcn_N > 20:
        fcn_N = 1

    return fcn_x[0]  # 只返回第一个元素，保证是标量

# 手动模拟离散系统1: 100s / (s + 100)
y1 = np.zeros(num_steps)
for i in range(1, num_steps):
    y1[i] = y1[i - 1] + 100 * (input2[i - 1] - y1[i - 1]) * dt

# 手动模拟离散系统2: (204s + 22.5) / s
y2 = np.zeros(num_steps)
for i in range(1, num_steps):
    y2[i] = y2[i - 1] + (204 * input2[i - 1] + 22.5) * dt

# 处理溢出问题
y1 = 314 * np.clip(y1, -1e10, 1e10)

# 仿真
for i in range(1, num_steps):
    # 零阶保持器输出
    u_held = input1[i]  # 由于input1始终为0，这里实际为0
    u_func = fcn(u_held)  # 调用fcn函数

    # 延迟一个采样周期的输入信号
    delayed_u = u[i - 1]

    # 计算中间信号
    intermediate_signal[i] = (u_func - np.sum(dt_K) * u_func + np.sum(a) * u_func
                              + np.sum(c1) * u_func - (np.sum(c1) * delayed_u + np.sum(S) * delayed_u + np.sum(h) * u_func)
                              + input2[i])

    # 离散时间积分器
    y[i] = y[i - 1] + intermediate_signal[i]

    # 零阶保持
    y_held = y[i]  # 假设零阶保持直接是当前值

    # 输出信号经过增益K
    y_final[i] = 0.89 * y_held

# 确保所有数组的长度匹配
y_final = y_final[:num_steps]
y1 = y1.flatten()[:num_steps]
y2 = y2.flatten()[:num_steps]

# 最终输出信号
y_final_total = y_final + y1 + y2

# 绘制结果
plt.plot(t, y_final_total, label='Output y_final_total')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.legend()
plt.show()
