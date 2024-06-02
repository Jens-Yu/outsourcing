from vehicle import Driver

# 创建 Driver 的实例
driver = Driver()

# 设置车辆的目标速度为 60.0 m/s
target_speed = 60.0
driver.setCruisingSpeed(target_speed)

# Webots 中的默认步长，单位是毫秒
time_step = int(driver.getBasicTimeStep())

# 主循环
while driver.step() != -1:
    # 此处可以添加其他车辆控制逻辑
    pass
