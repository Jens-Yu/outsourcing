import numpy as np


class EnvCore(object):
    def __init__(self):
        self.agent_num = 2  # Number of agents (drones)
        self.obs_dim = 24  # Observation dimension
        self.action_dim = 3  # Action dimension (distance, angle, user index)
        self.max_steps = 10  # Maximum number of steps per episode
        self.current_step = 0  # Current step counter

        # Initial drone positions
        self.drone_positions = np.array([[0, 0], [1000, 1000]])

        # User positions
        self.user_positions = np.array([
            [180, 800], [720, 280], [880, 820], [150, 200], [270, 580],
            [900, 460], [220, 380], [500, 350], [760, 380], [270, 160]
        ])

    def reset(self):
        self.current_step = 0  # Reset current step counter
        self.drone_positions = np.array([[0, 0], [1000, 1000]])  # Reset drone positions
        self.accessed_users = set()  # Reset the set of accessed users

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.concatenate((self.drone_positions.flatten(), self.user_positions.flatten()))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        self.current_step += 1  # Increment step counter
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        for i in range(self.agent_num):
            move_distance, move_angle, user_index = actions[i]
            user_index = int(user_index)  # 转换user_index为整数

            # 确保 user_index 在 0 到 9 之间
            user_index = max(0, min(user_index, len(self.user_positions) - 1))

            if user_index in self.accessed_users:
                # 如果选择的用户已经被接入，给予惩罚
                reward = -600.0
            else:
                # 正奖励用于接入新用户并更新无人机位置
                reward = 500 - move_distance
                self.accessed_users.add(user_index)
                self.drone_positions[i] = self.user_positions[user_index]
            if len(self.accessed_users) == len(self.user_positions):
                reward += 2000
            sub_agent_obs.append(np.concatenate((self.drone_positions.flatten(), self.user_positions.flatten())))
            sub_agent_reward.append(reward)

            # Info (optional debugging or performance information)
            sub_agent_info.append({"total_accessed_users": len(self.accessed_users)})

        # 检查是否所有用户已被接入或达到最大步数
        done = len(self.accessed_users) == len(self.user_positions) or self.current_step >= self.max_steps
        sub_agent_done = [done for _ in range(self.agent_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
