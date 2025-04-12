import gym
import simple_driving
from stable_baselines3 import DQN
import pybullet as p

# Load the trained model
model = DQN.load("dqn_simple_driving")

# Create environment with GUI
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)

obs, info = env.reset()
# Set camera right after reset
p.resetDebugVisualizerCamera(cameraDistance=17, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()
