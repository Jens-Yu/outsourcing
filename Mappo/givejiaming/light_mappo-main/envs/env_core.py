import numpy as np



class EnvCore(object):
    def __init__(self):
        self.agent_num = 2  # Number of agents (drones)
        self.obs_dim = 22  # Observation dimension
        self.action_dim = 10  # Action dimension is just the user index
        self.max_steps = 50  # Maximum number of steps per episode
        self.current_step = 0  # Current step counter
        self.total_distances = []  # 存储每个episode的总移动距离
        # Initial drone positions
        self.drone_positions = np.array([[0, 0], [1000, 1000]])

        # User positions
        self.user_positions = np.array([
            [180, 800], [720, 280], [880, 820], [150, 200], [270, 580],
            [900, 460], [220, 380], [500, 350], [760, 380], [270, 160]
        ])

    def reset(self):
        self.current_step = 0
        self.drone_positions = np.array([[0, 0], [1000, 1000]])
        self.accessed_users = []


        return self._get_obs()

    def step(self, actions):
        self.current_step += 1
        rewards = []
        done = False
        # 在这里减少重复访问的惩罚
        repeat_visit_penalty = -100

        # 设置访问新用户的最小奖励
        min_reward_for_new_user = 100

        for i, user_index_array in enumerate(actions):
            user_index = None
            #print('actions:', actions)
            for j, element in enumerate(user_index_array):
                if element == 1:
                    user_index = j  # 提取单个动作索引
                    break  # 当找到第一个选中的用户时停止循环

            # 计算移动距离并累积
            distance = np.linalg.norm(self.drone_positions[i] - self.user_positions[user_index])

            # 移动无人机到指定用户的位置
            self.drone_positions[i] = self.user_positions[user_index]

            # 更新奖励逻辑：考虑移动距离
            if user_index in self.accessed_users:
                reward = repeat_visit_penalty
            else:
                # 确保即使距离很远也有正的奖励
                reward = max(500.0 - distance, min_reward_for_new_user)
                self.accessed_users.append(user_index)
            rewards.append(reward)

        # 修改完成任务的奖励
        completion_reward = 2000.0

        # 修改每步的惩罚
        step_penalty = 50

        if len(self.accessed_users) == len(self.user_positions):
            done = True
            for i in range(self.agent_num):
                rewards[i] += completion_reward
        elif self.current_step >= self.max_steps:
            done = True

        # 调整每步的惩罚
        rewards = [r - step_penalty for r in rewards]

        obs = self._get_obs()
        info = [{"total_accessed_users": len(self.accessed_users)} for _ in range(self.agent_num)]
        done_flags = [done for _ in range(self.agent_num)]

        return obs, rewards, done_flags, info

    def _get_obs(self):
        """ Get the current observation of the environment. """
        obs = []
        for i in range(self.agent_num):
            obs.append(np.concatenate((self.drone_positions[i], self.user_positions.flatten())))
        return obs

