import gym
import simple_driving
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import pybullet as p


def main():
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
    model.learn(total_timesteps=100_000)

    # Save the trained model
    model.save("dqn_simple_driving")

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


    
    
    
    
    
    
    
    
    



if __name__ == '__main__':
    main()