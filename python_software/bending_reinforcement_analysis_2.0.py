# Name: Jiaming Yu
# Time:2024/05/10
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from math import pi
import tkinter as tk
from PIL import Image, ImageTk
import sys
import os


def main(max_k, D1, d1, d2, L1, L2, L3, EIp, Es, F, q):
    ls = 10  # 管道总长度
    x = np.linspace(0, ls, 500)
    y_init = np.zeros((2, x.size))
    y_init[0, :] = 1  # 假设解的第一部分始终为1

    def gangdu(x):
        Ds = np.where(x < L1, D1,
                      np.where(x < L1 + L2, D1 - ((D1 - d2) / L2) * (x - L1),
                               np.where(x < L1 + L2 + L3, d2, d1)))
        Is = pi / 64 * (Ds**4 - d1**4)
        return Es * Is + EIp

    def dgangdu(x):
        Ds = np.where(x < L1, D1,
                      np.where(x < L1 + L2, D1 - ((D1 - d2) / L2) * (x - L1), d1))
        dDs_dx = np.where((x >= L1) & (x < L1 + L2), -Es * pi / 64 * 4 * Ds**3 * (D1 - d2) / L2, 0)
        return dDs_dx

    def fangcheng(x, y):
        EI = gangdu(x)
        dEI = dgangdu(x)
        # Ensure EI is not zero to avoid division by zero
        EI = np.where(EI == 0, 1e-10, EI)
        # Calculate the second part of the differential equations
        derivative = -(dEI * y[1] + F * np.sin(q - y[0])) / EI
        # Stack and return the result
        return np.vstack((y[1], derivative))

    def bianjie(ya, yb):
        return np.array([ya[0], yb[0] - q])

    def draw_horizontal_lines_for_ticks(ax, color='gray', linestyle='--'):
        # Get the current y-axis tick locations and draw horizontal lines at these positions
        ticks = ax.get_yticks()
        for tick in ticks:
            if tick != max_k:
                plt.axhline(y=tick, color=color, linestyle=linestyle)

    #solinit = np.zeros((2, x.size))
    sol = solve_bvp(fangcheng, bianjie, x, y_init)

    plt.plot(sol.x, sol.y[1, :], label='Calculated curve')
    plt.axhline(y=max_k, color='red', linestyle='--', label='Max κ')
    plt.plot(sol.x[np.argmax(sol.y[1, :])], np.max(sol.y[1, :]), 'r*', label='Maximum point')
    ax = plt.gca()
    draw_horizontal_lines_for_ticks(ax)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.axvline(x=2, color='gray', linestyle='--')
    plt.axvline(x=4, color='gray', linestyle='--')
    plt.axvline(x=6, color='gray', linestyle='--')
    plt.axvline(x=8, color='gray', linestyle='--')
    plt.axvline(x=10, color='gray', linestyle='--')
    plt.legend()
    plt.show()

#main(0.03, 0.3, 0.1, 0.2, 0.2, 1, 0.1, 80, 80000, 5, 0.1)

def show_frame(frame):
    frame.tkraise()
    if frame == frame1:
        app.geometry('800x500')
    elif frame == frame2:
        app.geometry('500x500')
def run_simulation():
    max_k = float(entry_max_k.get())
    F = float(entry_F.get())
    EIp = float(entry_EIp.get())
    q = float(entry_q.get())
    D1 = float(entry_D1.get())
    d1 = float(entry_d1.get())
    d2 = float(entry_d2.get())
    L1 = float(entry_L1.get())
    L2 = float(entry_L2.get())
    L3 = float(entry_L3.get())
    Es = float(entry_Es.get())
    main(max_k, D1, d1, d2, L1, L2, L3, EIp, Es, F, q)


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

app = tk.Tk()
app.title("bending_reinforcement_analysis_2.0")
#app.geometry('800x400')

# 创建Frame容器
frame1 = tk.Frame(app)
frame2 = tk.Frame(app)
#result_frame = tk.Frame(app)

for frame in (frame1, frame2):
    frame.grid(row=0, column=0, sticky='news')

title_label1 = tk.Label(frame1, text="弯曲加强器分析 2.0", font=('Arial', 16))
title_label1.grid(row=0, column=0, sticky='w')
image3_path = resource_path('Figure_3.png')
image3 = Image.open(image3_path)
photo3 = ImageTk.PhotoImage(image3)
label_image3_1 = tk.Label(frame1, image=photo3)
label_image3_1.image = photo3
label_image3_1.grid(row=0, column=1, sticky='e')

title_label2 = tk.Label(frame2, text="弯曲加强器分析 2.0", font=('Arial', 16))
title_label2.grid(row=0, column=0, sticky='w')
label_image3_2 = tk.Label(frame2, image=photo3)
label_image3_2.image = photo3
label_image3_2.grid(row=0, column=1, sticky='e')


# Frame 1
image1_path = resource_path('Figure_1.png')
#image1 = Image.open('./Figure_1.png')
image1 = Image.open(image1_path)
photo1 = ImageTk.PhotoImage(image1)
label_image1 = tk.Label(frame1, image=photo1)
label_image1.image = photo1
label_image1.grid(row=1, column=0, columnspan=2)

labels1 = ["Maxκ", "F", "EIp", "角度"]
entries1 = []
for i, label in enumerate(labels1, 1):
    tk.Label(frame1, text=label).grid(row=i + 1, column=0)
    entry = tk.Entry(frame1)
    entry.grid(row=i + 1, column=1)
    entries1.append(entry)

entry_max_k, entry_F, entry_EIp, entry_q = entries1

next_page1 = tk.Button(frame1, text="下一页", command=lambda: show_frame(frame2))
next_page1.grid(row=6, column=0, columnspan=2)

# Frame 2
image2_path = resource_path('Figure_2.png')
image2 = Image.open(image2_path)
#image2 = Image.open('./Figure_2.png')
photo2 = ImageTk.PhotoImage(image2)
label_image2 = tk.Label(frame2, image=photo2)
label_image2.image = photo2  # 保留对图片的引用
label_image2.grid(row=1, column=0, rowspan=8, sticky='ns')

labels2 = ["D1", "d1", "d2", "L1", "L2", "L3", "Es"]
entries2 = []
for i, label in enumerate(labels2):
    tk.Label(frame2, text=label).grid(row=i+1, column=1, sticky='e')
    entry = tk.Entry(frame2)
    entry.grid(row=i+1, column=2, sticky='w')
    entries2.append(entry)

entry_D1, entry_d1, entry_d2, entry_L1, entry_L2, entry_L3, entry_Es = entries2

prev_page2 = tk.Button(frame2, text="上一页", command=lambda: show_frame(frame1))
prev_page2.grid(row=8, column=1, sticky='e')

next_page2 = tk.Button(frame2, text="显示结果", command=lambda: [run_simulation()])
next_page2.grid(row=8, column=2, sticky='w')

# Result Frame
'''result_label = tk.Label(result_frame, text="计算结果将在这里显示")
result_label.grid(row=0, column=0)
'''
show_frame(frame1)  # 默认显示第一页

app.mainloop()



