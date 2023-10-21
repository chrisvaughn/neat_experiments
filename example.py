import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")

observation, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
