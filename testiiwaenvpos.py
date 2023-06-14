
import gymnasium as gym
import kuka_env_example
import time
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env 


env = gym.make('iiwaEnvPos-v0')
observation, info = env.reset()
env.render()
print(observation)
print(env.observation_space)
action_spac = env.action_space.sample()
observation_spac = env.observation_space.sample()
print("action_space", action_spac)
print("observation_space", observation_spac)
check_env(env=env)
# model_dir = "models/PPO"
# logdir = "logs"
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# if not os.path.exists(logdir):
#     os.makedirs(logdir)

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/PPO")
# TIMESTEPS=10000
# for i in range(1,50):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
#     model.save(f"{model_dir}/{TIMESTEPS*i}")


#%%
# observation, info = env.reset(seed=123)
# print ("observation", observation, "info", info)
# for episode in range(10):
#     observation, info = env.reset()
#     for t in range(1000):  
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             observation, info = env.reset()
#             print("Episode Finished after {} timesteps".format(t+1))
#             break
# env.close()

