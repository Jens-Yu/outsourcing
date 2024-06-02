from vehicle import Driver

# 创建 Driver 的实例
driver = Driver()

# 设置车辆的目标速度（例如 10.0 m/s）
target_speed = 100.0
driver.setCruisingSpeed(target_speed)

# Webots 中的默认步长，单位是毫秒
time_step = int(driver.getBasicTimeStep())

# 旅行的目标距离（米）
target_distance = 600

# 已经旅行的距离
traveled_distance = 0

# 主循环
while driver.step() != -1:
    # 更新行驶的距离
    # 距离 = 速度（米/秒） * 时间（秒）
    traveled_distance += target_speed * (time_step / 1000.0)
    
    # 检查是否达到目标距离
    if traveled_distance >= target_distance:
        break

    # 可以添加其他控制逻辑...
