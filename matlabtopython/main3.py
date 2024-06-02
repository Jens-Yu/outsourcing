import numpy as np
import cmath
import time
from iapws import IAPWS97
import matplotlib.pyplot as plt
import pandas as panda
# 定义PID
# 创建PID类


class PIDController:
    def __init__(self, kp, ki, kd, filter_coefficient):
        self.kp1 = kp  # 比例增益
        self.ki1 = ki  # 积分增益
        self.kd1 = kd  # 微分增益
        self.filter_coefficient = filter_coefficient  # 滤波器系数
        self.integral = 0  # 积分项初始化
        self.prev_error = 0  # 上一个时间步的误差
        #self.filtered_error = 0  # 滤波后的误差
        self.filtered_d_term = 0


    def update(self, setpoint, process_variable, time_step):
        error = setpoint - process_variable  # 计算当前误差
        # 使用滤波器对误差进行平滑处理
        #self.filtered_error = self.filter_coefficient * error + (1 - self.filter_coefficient) * self.filtered_error
        # 更新积分项
        self.integral += error * time_step
        # 计算微分项
        derivative = (error - self.prev_error) / time_step
        self.filtered_d_term = (self.filter_coefficient * derivative * time_step + self.filtered_d_term) / (1 + self.filter_coefficient * time_step)
        # 存储当前滤波后的误差为上一个时间步的误差
        self.prev_error = error
        # 根据PID公式计算输出
        output = self.kp1 * error + self.ki1 * self.integral + self.kd1 * self.filtered_d_term
        return output

# 用Python搭建几个积分环节

def integrator(dy, y, step_size, lower_bound=float('-inf'), upper_bound=float('inf')):
    y += dy * step_size
    y = np.clip(y, lower_bound, upper_bound)
    return y

# 带有饱和限制的四阶龙格-库塔方法


def runge_kutta_4(f, t0, y0, step_size, n, lower_bound=float('-inf'), upper_bound=float('inf')):
    time_array = np.zeros(n + 1)
    y = np.zeros(n + 1)
    time_array[0] = t0
    y[0] = y0
    #print("y shape:", y.size, "time_array shape:", time_array.size)
    for i in range(n):
        ti = time_array[i]
        yi = y[i]
        k1 = step_size * f(yi, ti)
        #print("k1:", k1)
        ''' k2 = step_size * f(yi + k1 / 2, ti + step_size / 2)
        k3 = step_size * f(yi + k2 / 2, ti + step_size / 2)
        k4 = step_size * f(yi + k3, ti + step_size)
        print("k1:", k1, "k2:", k2, "k3:", k3, "k4:", k4)
        y_next = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6'''
        y_next = yi + k1
        #print("y_next:", y_next)
        y[i + 1] = np.clip(y_next, lower_bound, upper_bound)
        time_array[i + 1] = ti + step_size
    return y, time_array  # 返回整个积分结果和时间数组
# 省煤器段模块


def fcn(ne, d0, ty1, tj, p1, t1):
    mj = 5000
    cj = 0.5
    v = 80
    vy = 100
    ky = 80

    # 计算 P0, T0, Ty0
    p0 = -4.9635549e-8 * (ne ** 3) + 6.63318631e-5 * (ne ** 2) - 0.0153 * ne + 15.5287
    t_0 = -5.552223890598186e-5 * (ne ** 2) + 0.188222106275285 * ne + 187.4638
    ty0 = -0.00024444444444444345 * (ne ** 2) + 0.4033333333333324 * ne + 350.0000000000002

    # 计算 mid1, Dy, ksi1, kc
    mid1 = [-0.0015650666666666663 * (ne ** 2) + 2.424473333333333 * ne - 357.2819999999998,
            -1.6666666664670333e-09 * (ne ** 2) + 1.4033333331143116e-06 * ne - 3.699999995103936e-05,
            -0.00018639250135555523 * (ne ** 2) + 0.16768895880333307 * ne - 29.693576604999947]
    dy = 0.85 * mid1[0]
    ksi1 = 20 * mid1[1]
    kc = 12 * mid1[2]

    # rho1, drho1

    rho1 = 1210.86211811911 - 1.59793418318532 * t1 - 35.8884167229680 * p1 + 0.313417735621854 * p1 * t1 - 0.00045548926374317 * (t1 ** 2) - 0.867613946591206 * (p1 ** 2) + 0.00280501652204646 * (p1 ** 2) * t1 - 0.000615644347189089 * p1 * (t1 **2)

    drho1 = -35.8884167229680 + 0.313417735621854 * t1 - 2 * 0.867613946591206 * p1 + 2 * 0.00280501652204646 * p1 * t1 - 0.000615644347189089 * (t1 ** 2)
    # 计算 H0, mid3, H1, dHT1, dHP1

    h0 = 106.780612309849 + 3.54538497629407 * t_0 - 16.088116133118092 * p0 + 0.063459069518316 * p0 * t_0 + 0.001505531044285 * (t_0 ** 2) + 1.264454190598599 * (p0 ** 2) - 0.007735643634244 * (p0 ** 2) * t_0 + 1.086607348530378e-05 * (p0 ** 2) * (t_0 ** 2)  # noqa

    h1 = 2.91894501846895 + 4.48997695937590 * t1 - 47.8864910940965 * p1 + 0.152989961918212 * p1 * t1 + 0.000109879475056696 * (t1 ** 2) + 6.18027726004829 * (p1 ** 2) - 0.034847848273992 * (p1 ** 2) * t1 + 4.78907792762745e-05 * (p1 ** 2) * (t1 ** 2)
    dht1 = 4.48997695917590 + 2 * 0.000109879475056696 * t1 + 0.152989961918212 * p1 - 0.0348478482739920 * (p1 ** 2) + 2 * 4.78907792762745e-05 * (p1 ** 2) * t1

    dhp1 = -47.8864910940965 + 0.1529899619182152 * t1 + 2 * 6.18027726004829 * p1 - 2 * 0.034847848273992 * p1 *t1 + 2 * 4.78907792762745e-05 * p1 * (t1 **2)
    rhoy1 = 1.799999999999994e-05 * (ty1 ** 2) - 0.015979999999999953 * ty1 + 4.03699999999999
    mid4 = [-1.9999999999988916e-07 * (ty1 ** 2) + 0.0004899999999999016 * ty1 + 0.9900000000000216,
            2.2204460486140494e-20 * (ty1 ** 2) - 0.0009200000000000152 * ty1 + 0.8930000000000026]
    cy0, cy1 = mid4[0], mid4[1]
    pout = 2.85195241345460e-05 * (ne ** 2) - 0.0113109873432667 * ne + 15.9447132979074
    complex_result = complex(rho1 * (p1 - pout) / ksi1)
    sqrt_complex = np.sqrt(complex_result)
    # 取平方根的实部
    d1 = sqrt_complex.real

    dp1 = (d0 - d1) / (v * drho1)
    d_t1 = (d0 * (h0 - h1) + kc * np.real(np.power(complex(d1), 0.8)) * (tj - t1) - v * rho1 * dhp1 * dp1) / (v * rho1 * dht1)
    #d_t1 = (d0 * (h0 - h1) + kc * (d1 ** 0.8).real * (tj - t1) - v * rho1 * dhp1 * dp1) / (v * rho1 * dht1)
    dtj = (ky * np.real(np.power(complex(dy), 0.8)) * (ty1 - tj) - kc * np.real(np.power(complex(d1), 0.8)) * (tj - t1)) / (mj * cj)
    #dtj = (ky * (dy ** 0.8).real * (ty1 - tj) - kc * (d1 ** 0.8).real * (tj - t1)) / (mj * cj)
    dty1 = (dy * cy0 * ty0 - dy * cy1 * ty1 - ky * np.real(np.power(complex(dy), 0.8)) * (ty1 - tj)) / (rhoy1 * vy * cy1)
    #dty1 = (dy * cy0 * ty0 - dy * cy1 * ty1 - ky * (dy ** 0.8).real * (ty1 - tj)) / (rhoy1 * vy * cy1)
    return d1, dp1, d_t1, dty1, dtj, p0, pout, h0, h1, ksi1


# 汽包蓄积段


'''def fcn2(dsm, pd, v, dxj, dgr, deta_xjs):
    r = 461.5  # 蒸汽气体常数
    vqb = 256  # 汽包体积
    ad = 45.318  # 常数ad

    # 将压力从 bar 转换为 Pa
    p = pd * 1e5
    #print("p:", p)
    # 使用 CoolProp 获取饱和温度
    t_s = CP.PropsSI('T', 'P', p, 'Q', 0, 'Water')
    #print("t_s:", t_s)
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
    #print("pd received:", pd)
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

def fcn3(pd1, pt):
    #print("pd1:", pd1, "pt:", pt)
    K = 0.08764256902732083  # 假设的K值
    const = 18.439682689872793  # 图中标注的常数值
    const_c0 = 38 / 2.8  # 图中的c0计算
    plus = K * pd1 ** 2 - 1.8391411470804848 * pd1 +const
    #plus = 0.0876 * pd1 ** 2 - 1.8391 * pd1 + 18.4397
    #c0 = 38 / 2.8
    #ds = c0 * plus ** 0.5 * (abs(pd1 - pt)) ** 0.5 + 200
    ds = const_c0 * plus * np.sqrt(complex(pd1 - pt)) + 200
    if ds > 600:
        ds = 600

    elif ds < 100:
        ds = 100
    else:
        ds = np.real(ds)
    return ds


def adjust_value(process_variable):
    #print("process_variable:", process_variable)
    if process_variable > 50:
        return 50
    elif process_variable < -50:
        return -50
    else:
        return process_variable


def main():
    main_steam_flow_path = 'D:/Programfiles/outsourcing/matlabtopython/main_steam_flow.xls'
    # 设置第一个PID参数和滤波器系数
    dxj_path = 'D:/Programfiles/outsourcing/matlabtopython/dxj.xlsx'
    pt_path = "D:/Programfiles/outsourcing/matlabtopython/pt.xlsx"
    ne_path = "D:/Programfiles/outsourcing/matlabtopython/ne.xlsx"
    main_steam_flows = panda.read_excel(main_steam_flow_path, header=None)
    main_steam_flows = main_steam_flows.iloc[:, 1]
    dxjs = panda.read_excel(dxj_path, header=None)
    dxjs = dxjs.iloc[:, 1]
    pts = panda.read_excel(pt_path, header=None)
    pts = pts.iloc[:, 1]
    nes = panda.read_excel(ne_path, header=None)
    nes = nes.iloc[:, 1]
    kp = 203.8263761763
    ki = 22.4302136842899
    kd = 314.119346774621
    filter_coefficient1 = 0.810831032432681
    # 实例化第一个PID控制器
    pid = PIDController(kp, ki, kd, filter_coefficient1)

    # 第二个PID控制器的参数

    kp2 = 0.00527059631029743
    ki2 = 1.6113586492284e-05
    kd2 = 0.389685831097133
    filter_coefficient2 = 0.0914334420746481
    # 实例化第二个PID控制器
    pid2 = PIDController(kp2, ki2, kd2, filter_coefficient2)
    setpoint1 = 0  # 主控制器的设定值
    main_steam_flow = main_steam_flows[0]  # 主汽流量值，后续导入数据
    ne = nes[0]  # 负荷值，后续导入数据
    #d0 = 0  # 给水流量的初始值
    output2 = 0  # 副控制器的变量

    # 对于省煤器模块的四个积分环节的赋初值

    p1 = 18.5
    t1 = 300
    ty1 = 482
    tj = 300

    #d1, dp1, d_t1, dty1, dtj, _, _, _, _, _, _ = fcn(ne, d0, ty1_initial, tj_initial, p1_initial, t1_initial)

    h = 0.01  # 时间步长
    '''t_0 = 0   # 积分初始时间
    tn = 0.01  # 积分结束时间
    n_steps = int((tn - t_0)/h)  # 计算迭代步数'''
    dxj = dxjs[0]

    deta_xjs = dxj

    # 对于汽包蓄积段三个积分环节赋初值

    v = 50
    pd_initial = 18000000
    #initial_condition = 0
    pd = pd_initial * 0.000001
    #v = v_initial
    #dt1 = 0.01  # 主控制器的时间间隔
    #dt2 = 0.01  # 副控制器的时间间隔
    pd1 = pd_initial
    pt = pts[0]
    print("pd1:", pd1, "pt:", pt)
    ds = fcn3(pd1, pt)
    #print("ds:", ds)
    #ds = 0
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
    #n = total_time // h
    #change_point = time_interval // h
    #i = 0
    print("ne:", ne, "dxj:", dxj, "pt:", pt, "main_steam_flow:", main_steam_flow, "output_signal", output_signal)
    for i in range(n + 1):
        if change_point == 500:
            current_time += time_interval
            idx = i // change_point
            ne = nes[idx]
            dxj = dxjs[idx]
            deta_xjs = dxj
            pt = pts[idx]
            main_steam_flow = main_steam_flows[idx]
            output_signals.append(output_signal)  # 记录输出信号
            time_steps.append(current_time)  # 记录当前模拟时间
            change_point = 0
            print("ne:", ne, "dxj:", dxj, "pt:", pt, "main_steam_flow:", main_steam_flow, "output_signal", output_signal)

        process_variable1 = output_signal
        # 更新第一个PID控制器的输出
        adjusted_process_variable1 = adjust_value(process_variable1)
        output1 = pid.update(setpoint1, adjusted_process_variable1, h)
        print("output1:", output1)
        setpoint2 = output1 + main_steam_flow
        output2 = pid2.update(setpoint2, output2, h)
        #process_variable2 += output2
        d0 = output2
        d1, dp1, d_t1, dty1, dtj, p0, pout, h0, h1, ksi1 = fcn(ne, d0, ty1, tj, p1, t1)
        #print("d1:", d1)
        p1 = integrator(dp1, p1, h)
        t1 = integrator(d_t1, t1, h, 1, 1000)
        ty1 = integrator(dty1, ty1, h, 1, 2000)
        tj = integrator(dtj, tj, 10, 1000)
        '''# 调用龙格-库塔函数进行积分
        p1, _ = runge_kutta_4(lambda p1, t: dp1, t_0, p1_initial, h, n_steps)
        p1_initial = p1[-1]  # 更新p1的初始条件为积分后的最后一个值
        t1, _ = runge_kutta_4(lambda t1, t: d_t1, t_0, t1_initial, h, n_steps, 1, 1000)
        t1_initial = t1[-1]  # 更新t1的初始条件为积分后的最后一个值
        ty1, _ = runge_kutta_4(lambda ty1, t: dty1, t_0, ty1_initial, h, n_steps, 1, 2000)
        ty1_initial = ty1[-1]  # 更新ty1的初始条件为积分后的最后一个值
        tj, _ = runge_kutta_4(lambda tj, t: dtj, t_0, tj_initial, h, n_steps, 10, 1000)
        tj_initial = tj[-1]  # 更新tj的初始条件为积分后的最后一个值'''
        #d1 = fcn(ne, d0, ty1_initial, tj_initial, p1_initial, t1_initial)
        dsm = d1
        dgr = ds
        #print("pd:", pd)
        deta_pd, deta_v, dld_dt, t_s, deta_vs = fcn2(dsm, pd, v, dxj, dgr, deta_xjs)
        #print("dld_dt:", dld_dt)
        # print("pd_initial:", pd_initial)
        #print("deta_pd:", deta_pd)
        pd = integrator(deta_pd, pd * 1e6, h, 6000000, 30000000)
        #pd_integral, _ = runge_kutta_4(lambda pd, t: deta_pd, t_0, pd_initial, h, n_steps, 6000000, 30000000)
        #pd_initial = pd_integral[-1]
        gain1 = 0.000001
        pd = pd * gain1  # 应用增益得到Pd的值
        #v, _ = runge_kutta_4(lambda v, t: deta_v, t_0, v_initial, h, n_steps, 0, 225)
        v = integrator(deta_v, v, h, 0, 225)
        #v_initial = v[-1]
        #y, _ = runge_kutta_4(lambda y, t: dld_dt, t_0, initial_condition, h, n_steps)
        y = integrator(dld_dt, output_signal, h)
        #initial_condition = y[-1]
        # 应用增益得到输出信号
        gain2 = 1000
        if i == 0:
            output_signal = y * gain2
        else:
            output_signal = y
        # 汽包出口段的汽包压力输入值pd1
        pd1 = pd
        ds = fcn3(pd1, pt)
        #print("ds:", ds)
        change_point += 1
        '''if i % change_point == 0:
            output_signals.append(output_signal)  # 记录输出信号
            time_steps.append(current_time)  # 记录当前模拟时间
            current_time += time_interval'''

    #last_record_time = time.time()
    
    '''while current_time <= total_time:
        #start_time = time.time()
        ne = nes[i]
        dxj = dxjs[i]
        pt = pts[i]
        main_steam_flow = main_steam_flows[i]
        process_variable1 = output_signal
        #print("output_signal:", output_signal)
        #print("process_variable1:", process_variable1)
        # 更新第一个PID控制器的输出
        adjusted_process_variable1 = adjust_value(process_variable1)
        #print("adjusted_process_variable1:", adjusted_process_variable1)
        output1 = pid.update(setpoint1, adjusted_process_variable1, dt1)
        #print("output1:", output1)
        setpoint2 = main_steam_flow + output1
        #print("setpoint2:", setpoint2)
        output2 = pid2.update(setpoint2, process_variable2, dt2)
        process_variable2 += output2 * dt2
        #print("process_variable2:", process_variable2)
        d0 = output2
        d1, dp1, d_t1, dty1, dtj, p0, pout, h0, h1, ksi1, ty1 = fcn(ne, d0, ty1_initial, tj_initial, p1_initial, t1_initial)
        # 调用龙格-库塔函数进行积分
        p1, _ = runge_kutta_4(lambda p1, t: dp1, t_0, p1_initial, h, n_steps)
        p1_initial = p1[-1]  # 更新p1的初始条件为积分后的最后一个值
        t1, _ = runge_kutta_4(lambda t1, t: d_t1, t_0, t1_initial, h, n_steps, 1, 1000)
        t1_initial = t1[-1]  # 更新t1的初始条件为积分后的最后一个值
        ty1, _ = runge_kutta_4(lambda ty1, t: dty1, t_0, ty1_initial, h, n_steps, 1, 2000)
        ty1_initial = ty1[-1]  # 更新ty1的初始条件为积分后的最后一个值
        tj, _ = runge_kutta_4(lambda tj, t: dtj, t_0, tj_initial, h, n_steps, 10, 1000)
        tj_initial = tj[-1]  # 更新tj的初始条件为积分后的最后一个值

        dsm = d1
        dgr = ds
        deta_pd, deta_v, dld_dt, t_s, deta_vs = fcn2(dsm, pd, v_initial, dxj, dgr, deta_xjs)
        #print("pd_initial:", pd_initial)
        #print("deta_pd:", deta_pd)
        pd_integral, _ = runge_kutta_4(lambda pd, t: deta_pd, t_0, pd_initial, h, n_steps, 6000000, 30000000)
        pd_initial = pd_integral[-1]
        gain1 = 0.000001
        pd = pd_initial * gain1  # 应用增益得到Pd的值
        v, _ = runge_kutta_4(lambda v, t: deta_v, t_0, v_initial, h, n_steps, 0, 225)
        v_initial = v[-1]
        y, _ = runge_kutta_4(lambda y, t: dld_dt, t_0, initial_condition, h, n_steps)
        initial_condition = y[-1]
        # 应用增益得到输出信号
        gain2 = 1000
        output_signal = initial_condition * gain2
        #print("output_signal:", output_signal)
        # 汽包出口段的汽包压力输入值pd1
        pd1 = pd
        ds = fcn3(pd1, pt)
        i += 1
        # 更新当前时间
        time_steps.append(current_time)
        output_signals.append(output_signal)
        current_time += time_interval'''

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

