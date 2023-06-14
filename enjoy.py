
import gymnasium as gym
import kuka_env_example
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env 


env = gym.make('iiwaEnvPos-v0')
observation, info = env.reset()
env.render()
model_dir = "models/PPO"
model_path = f"{model_dir}/490000"

model = PPO.load(model_path, env=env)
episode = 10000
terminated=1

# some for loop idk
action, _ = model.predict(obs)
obs, reward, terminated, truncated, info = env.step(action)
if terminated:
    obs, _ = env.reset()
    # time.sleep(0.1)

env.close()
