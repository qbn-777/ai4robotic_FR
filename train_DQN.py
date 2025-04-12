import gym
import simple_driving
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = make_vec_env(lambda: gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True))

# Initialise the DQN model
model = DQN("MlpPolicy",# Use MLP policy for DQN
            env,
            verbose=1, # Show training logs
            learning_rate=1e-3, 
            buffer_size=10000, 
            learning_starts=1000, 
            batch_size=32)
# Train the model
model.learn(total_timesteps=300_000)

# Save the trained model
model.save("dqn_simple_driving_turn2goal")
