import gym
import simple_driving
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
env = env.unwrapped

# Initialise the DQN model
model = DQN("MlpPolicy",# Use MLP policy for DQN
            env,
            verbose=1, # Show training logs
            learning_rate=1e-3, 
            buffer_size=10000, 
            learning_starts=1000, 
            batch_size=32)

# Train with safe exit
try:
    model.learn(total_timesteps=300_000)
except KeyboardInterrupt:
    print("\n Training interrupted by user.")
    model.save("dqn_simple_driving_interrupted")
    print(" Model saved as 'dqn_simple_driving_interrupted.zip'")
else:
    # Save normally if finished
    model.save("dqn_simple_driving_final")
    print(" Training completed and model saved.")