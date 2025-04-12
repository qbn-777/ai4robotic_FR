import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
import time
from simple_driving.resources.obstacle import Obstacle


RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40], dtype=np.float32),
            high=np.array([40, 40], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0

    def step(self, action):
        # Convert discrete action into throttle + steering
        if self._isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]

        self.car.apply_action(action)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        # Calculate distance to goal
        dist_to_goal = math.sqrt((carpos[0] - goalpos[0]) ** 2 + (carpos[1] - goalpos[1]) ** 2)
        progress = self.prev_dist_to_goal - dist_to_goal
        reward = progress * 10  # encourage moving closer

        # Small time penalty to encourage speed
        reward -= 0.1


        # Check for collisions with obstacles
        for obs in self.obstacles:
            if len(self._p.getContactPoints(bodyA=self.car.car, bodyB=obs.obstacle)) > 0:
                reward -= 4  # collision penalty
                break

        self.prev_dist_to_goal = dist_to_goal

        # If goal is reached
        if dist_to_goal < 1.5 and not self.reached_goal:
            self.done = True
            self.reached_goal = True
            reward += 150  # bonus for success

        return np.array(car_ob, dtype=np.float32), reward, self.done, {}

    

    #Change this to step when 
    def step2(self, action):
        #Funtion for prioritise facing the goal
        # Feed action to the car and get observation of car's state
        if self._isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]

        self.car.apply_action(action)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        # Compute distance to goal
        dist_to_goal = math.sqrt((carpos[0] - goalpos[0]) ** 2 + (carpos[1] - goalpos[1]) ** 2)
        reward = -dist_to_goal  # Default reward: closer is better

        # ✅ ADDITION: Encourage facing the goal
        car_euler = self._p.getEulerFromQuaternion(carorn)
        car_yaw = car_euler[2]

        dir_to_goal = np.array([goalpos[0] - carpos[0], goalpos[1] - carpos[1]])
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-8  # normalize + avoid div by zero

        car_forward = np.array([math.cos(car_yaw), math.sin(car_yaw)])
        alignment = np.dot(dir_to_goal, car_forward)  # ranges from -1 to 1

        reward += alignment * 5  # bonus for facing goal

        self.prev_dist_to_goal = dist_to_goal

        # Done by reaching goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            self.done = True
            self.reached_goal = True
            reward += 100  # big bonus for success!

        ob = car_ob
    
        return np.array(ob, dtype=np.float32), reward, self.done, False, {}
    
    def step3(self, action):
        # Process discrete actions into throttle and steering
        if self._isDiscrete:
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]

        self.car.apply_action(action)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        # 1. Distance to goal
        dist_to_goal = math.sqrt((carpos[0] - goalpos[0]) ** 2 + (carpos[1] - goalpos[1]) ** 2)
        progress = self.prev_dist_to_goal - dist_to_goal

        # 2. Orientation alignment
        car_euler = self._p.getEulerFromQuaternion(carorn)
        car_yaw = car_euler[2]

        dir_to_goal = np.array([goalpos[0] - carpos[0], goalpos[1] - carpos[1]])
        dir_to_goal /= np.linalg.norm(dir_to_goal) + 1e-8  # avoid division by zero

        car_forward = np.array([math.cos(car_yaw), math.sin(car_yaw)])
        alignment = np.dot(dir_to_goal, car_forward)  # -1 (away) to +1 (facing goal)

        # 3. Reward logic
        reward = 0
        reward += progress * 10             # Encourage moving closer
        reward += alignment * 5             # Encourage facing the goal
        reward -= 0.1                       # Slight time penalty

        # Bonus for reaching the goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            reward += 100
            self.reached_goal = True
            self.done = True

        # Update distance
        self.prev_dist_to_goal = dist_to_goal

        terminated = self.done
        return np.array(car_ob, dtype=np.float32), reward, terminated, {}


        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)

        #ONLY UNCOMMENT FOR THIS WHEN RUN MODEL THAT IS TRAINED WITH OBSTACLES
        # Add obstacles
        self.obstacles = []

        positions = [
            (0, -6), (5, 2), (-5, -4), (3, -6) , (6, 0),
            (-6, 3), (2, -5), (-4, 4), (1, 6), (7, -1), (-2, 2),
            (4, -3), (6, 6), (-3, 0)
        ]
        for pos in positions:
            self.obstacles.append(Obstacle(self._p, position=pos))

        

        # Get observation to return
        carpos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        car_ob = self.getExtendedObservation()

        return np.array(car_ob, dtype=np.float32)



    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        # self._observation = []  #self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        observation = [goalPosInCar[0], goalPosInCar[1]]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()
        
