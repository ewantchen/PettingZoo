# run_gymnasium_env.py

import gymnasium
import gymnasium_env.envs as envs
import time

env = gymnasium.make('gymnasium_env:gymnasium_env/GridWorld', render_mode="human")
#env = gymnasium.make('gymnasium_env:gymnasium_env/GridWorld', render_mode="rgb_array")

obs = env.reset()

obs, info = env.reset()

for _ in range(50):  # Run for 50 steps
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print(obs, env)
        obs, info = env.reset()
    
    time.sleep(0.1)  # Pause so you can see the moves

env.close()