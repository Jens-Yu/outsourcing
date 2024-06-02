import numpy as np
import cmath
from iapws import IAPWS97
import matplotlib.pyplot as plt
import pandas as panda
import CoolProp.CoolProp as CP


# 用Python搭建几个积分环节

def integrator(dy, y, step_size, lower_bound=float('-inf'), upper_bound=float('inf')):
    y += dy * step_size
    y = np.clip(y, lower_bound, upper_bound)
    return y

# 汽包蓄积段
'''def fcn2(dsm, pd, v, dxj, dgr, deta_xjs):
    r = 461.5  # 蒸汽气体常数
    vqb = 256  # 汽包体积
    ad = 45.318  # 常数ad

    # 将压力从 Mpa 转换为 Pa
    p = pd * 1e6

    # 使用 CoolProp 获取饱和温度
    t_s = CP.PropsSI('T', 'P', p, 'Q', 0, 'Water')

    # 使用 CoolProp 获取液体和蒸汽的密度
    rho = CP.PropsSI('D', 'P', p, 'Q', 0, 'Water')
    rhos = CP.PropsSI('D', 'P', p, 'Q', 1, 'Water')

    # 计算 Vw 流体的体积变化
    deta_v = (dsm - dxj) / rho

    # 计算蒸汽体积的变化
    deta_vs = (dxj - deta_xjs) / rhos

    # 计算压力变化相关的量
    deta_pd = r * (t_s + 273.15) / (vqb - v) * (dxj - dgr)

    # 计算体积变化率

    dld_dt = 1 / ad * (deta_v + deta_vs)

    return deta_pd, deta_v, dld_dt, t_s, deta_vs'''

def fcn2(dsm, pd, v, dxj, dgr, deta_xjs):
    r = 461.5  # 蒸汽气体常数
    vqb = 256  # 汽包
    # print("pd received:", pd)
    # 液体和蒸汽密度，及饱和温度的计算，使用IAPWS97库
    steam_liquid = IAPWS97(P=pd, x=0)  # x=0 表示液态水
    steam_vapor = IAPWS97(P=pd, x=1)  # x=1 表示饱和蒸汽
    rho = steam_liquid.rho  # 液体密度
    rhos = steam_vapor.rho  # 蒸汽密度
    t_s = steam_liquid.T - 273.15  # 饱和温度转换为摄氏度

    # 流体和蒸汽的体积变化
    deta_v = (dsm - dxj) / rho
    deta_vs = (dxj - deta_xjs) / rhos

    # 压力变化相关的量
    deta_pd = r * (t_s + 273.15) / (vqb - v) * (dxj - dgr)

    # 通过给定的常数计算时间变化率
    ad = 45.318
    dld_dt = 1 / ad * (deta_v + deta_vs)

    return deta_pd, deta_v, dld_dt, t_s, deta_vs


def main():
    dgr_path = 'D:/Programfiles/outsourcing/matlabtopython/dgr1.xlsx'
    # 设置第一个PID参数和滤波器系数
    dxj_path = 'D:/Programfiles/outsourcing/matlabtopython/dxj2.xlsx'
    dsm_path = 'D:/Programfiles/outsourcing/matlabtopython/dsm2.xlsx'
    dgrs = panda.read_excel(dgr_path, header=None)
    dgrs = dgrs.iloc[:, 1]
    dxjs = panda.read_excel(dxj_path, header=None)
    dxjs = dxjs.iloc[:, 1]
    dsms = panda.read_excel(dsm_path, header=None)
    dsms = dsms.iloc[:, 1]
    #dgr = dgrs[0]
    #dsm = dsms[0]
    h = 0.01  # 时间步长

    #dxj = dxjs[0]

    #deta_xjs = dxj

    # 对于汽包蓄积段三个积分环节赋初值

    v = 50
    pd_initial = 18000000
    pd = pd_initial * 0.000001

    output_signal = 0  # 相对液位输出初始值

    # 定义一个大循环体，在0-1500秒内执行

    current_time = 0
    total_time = 150
    time_interval = 5
    output_signals = []
    time_steps = []
    output_signals.append(output_signal)  # 记录输出信号
    time_steps.append(current_time)  # 记录当前模拟时间
    n = int(total_time / h)
    change_point = 0
    for i in range(n + 1):
        if change_point == 500:
            current_time += time_interval
            output_signals.append(output_signal)  # 记录输出信号
            time_steps.append(current_time)  # 记录当前模拟时间
            change_point = 0
        #idx = i // change_point
        dxj = dxjs[i]
        deta_xjs = dxj
        dgr = dgrs[i]
        dsm = dsms[i]
            #print("dld_dt:", dld_dt, "output_signal:", output_signal)

        deta_pd, deta_v, dld_dt, t_s, deta_vs = fcn2(dsm, pd, v, dxj, dgr, deta_xjs)
        #print("dld_dt:", dld_dt)
        pd = integrator(deta_pd, pd, h, 6000000, 30000000)
        gain1 = 0.000001
        pd = pd * gain1  # 应用增益得到Pd的值
        v = integrator(deta_v, v, h, 0, 225)
        gain2 = 1000
        y = integrator(dld_dt, output_signal / gain2, h)
        # 应用增益得到输出信号

        #if i == n:
        output_signal = y * gain2
        #else:
            #output_signal = y

        change_point += 1

    # 循环结束后绘制图形
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, output_signals, marker='o', linestyle='-', color='b')
    plt.title('Output Signal Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Output Signal')
    plt.grid(True)
    plt.show()


# 执行主函数
if __name__ == "__main__":
    main()