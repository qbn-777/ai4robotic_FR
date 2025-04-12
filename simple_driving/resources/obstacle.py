import os

class Obstacle:
    def __init__(self, client, position=(0, 0), urdf_name="simpleBlock.urdf"):
        # Get path to the URDF file
        urdf_path = os.path.join(os.path.dirname(__file__), urdf_name)

        # Load the obstacle into the simulation
        self.obstacle = client.loadURDF(
            fileName=urdf_path,
            basePosition=[position[0], position[1], 0],
        )
