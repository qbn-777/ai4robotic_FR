Modifications:
* Added third-person and on-board camera rendering modes.
* Made runnable on google colab notebook

# Gym-Medium-Post
Basic OpenAI gym environment. 

Resource for the [Medium series on creating OpenAI Gym Environments with PyBullet](https://medium.com/@gerardmaggiolino/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24). 

Steps are changes, models to be loaded are
dqn_simple_driving

Goals are added, however not succesfully trained to be able to read goal effectively.

* Train the robot to drive to the green goal marker which spawns at random locations (60%)
    Completed, use model dqn_simple_driving, and best to remove most or all obstacles
    Remove obstacles in simple_driving_env.py in reset()

* Modify the epsilon-greedy function to incorporate prior knowledge (20%)
    With prior knownledge that it is better to rotate or aim towards goal before going there directly
    I make step3() to do so, however, after trained, it seems not to be so effective

* Modify the reward function (10%)
    reward functions have multiple attempts in different step() versions

* Add obstacles to the environment (10%)
    As seen, obstacles are added
