# IsaacLab-Simple-Tutorial-Walkthrough
from: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/index.html 

## 0- Pre-Requisites
- Before starting, make sure you have completed the prerequisites i.e. creating a virtual environment and a project with IsaacSim and IsaacLab installed as well as all other libraries e.g. Python 3.11 etc.
- You can find the step by step tutorial for the prerequisites here: https://github.com/marcelpatrick/Step-by-step-IsaacLab-Installation_2/blob/main/README.md 

## Classes and Configs
Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html

**When following the documentation, "isaac_lab_tutorial" should be replaced by your project name, in this case "myProject"**

- Open an Ubuntu terminal
- Activate the conda environment created with Python 3.11 ```conda activate env_isaaclab```
- Navigate to ```cd ~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject```
- Check the environment configurations:
  - Open the config file (using Sublime in this example): ```subl myproject_env_cfg.py```
  - If you don't have sublime enabled, run: ```sudo snap install sublime-text --classic```

**-> you can also navigate to these folders on Windows Explorer by starting at the path: ```\\wsl.localhost\Ubuntu-22.04\root\IsaacSim\myProject```**

## 1- Environment Design
- Defining the environment
Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/technical_env_design.html

### Define the Robot
- Inside ```~/IsaacSim/myProject/source/myProject/myProject``` Create a folder "robots" ```mkdir robots```
- Within this folder create two files: __init__.py and jetbot.py. ```touch __init__.py jetbot.py```
- __init__.py makes the folder a Python package, and jetbot.py will be your module file.
- Open jetbot.py ```nano jetbot.py``` copy the following code inside:
```
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
```
- Save it with Ctrl X, Yes.
- The only purpose of this file is to define a unique path in which to save our configurations.

### Environment Configuration
- Navigate to ```~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject```
- Open myproject_env_cfg.py and replace its content with
```
from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]
```
- **-> replace ```class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):``` with the name of your project, in this case ```class MyprojectEnvCfg(DirectRLEnvCfg):```**
- **-> replace ```from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG``` with the name of your project, in this case: ```from myProject.robots.jetbot import JETBOT_CONFIG```
- Save and close
- Here we have, effectively, the same environment configuration as before, but with the Jetbot instead of the cartpole

### Setup Environment Details
- Navigate to ```~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject#``` and open myproject_env.py
-  replace the contents of the __init__ and _setup_scene methods with the following
```
class MyprojectEnv(DirectRLEnv):
    cfg: MyprojectEnvCfg

    def __init__(self, cfg: MyprojectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
```
- Notice that the _setup_scene method doesn’t change and the _init__ method is simply grabbing the joint indices from the robot (remember, setup is called in super).

- The next thing our environment needs is the definitions for how to handle actions, observations, and rewards. First, replace the contents of _pre_physics_step and _apply_action with the following.
```
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self.actions = actions.clone()

def _apply_action(self) -> None:
    self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
```
- _pre_physics_step: getting data from the policy being trained and applying updates to the physics simulation ( lets us detach the process of getting data from the policy being trained and applying updates to the physics simulation)
- The _apply_action method is where those actions are actually applied to the robots on the stage

- Replace the contents of _get_observations and _get_rewards with the following
```
def _get_observations(self) -> dict:
    self.velocity = self.robot.data.root_com_lin_vel_b
    observations = {"policy": self.velocity}
    return observations

def _get_rewards(self) -> torch.Tensor:
    total_reward = torch.linalg.norm(self.velocity, dim=-1, keepdim=True)
    return total_reward
```

- ```def _get_observations(self) -> dict:```
  - We are applying actions to all robots on the stage at once! Here, when we need to get the observations, we need the body frame velocity for all robots on the stage, and so access ```self.robot.data``` to get that information.
  - root_com_lin_vel_b is a property of the ArticulationData that handles the conversion of the center-of-mass linear velocity from the world frame to the body frame.
  - it returns a dictionary composed of a policy: velocity value pair
 
- ```def _get_rewards(self) -> torch.Tensor:```
  - For each clone of the scene, we need to compute a reward value and return it as a tensor of shape [num_envs, 1]
  - As a place holder, we will make the reward the magnitude of the linear velocity of the Jetbot in the body frame. With this reward and observation space, the agent should learn to drive the Jetbot forward or backward, with the direction determined at random shortly after training starts.

- Replace the contents of _get_dones and _reset_idx with the following.
```
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    time_out = self.episode_length_buf >= self.max_episode_length - 1

    return False, time_out

def _reset_idx(self, env_ids: Sequence[int] | None):
    if env_ids is None:
        env_ids = self.robot._ALL_INDICES
    super()._reset_idx(env_ids)

    default_root_state = self.robot.data.default_root_state[env_ids]
    default_root_state[:, :3] += self.scene.env_origins[env_ids]

    self.robot.write_root_state_to_sim(default_root_state, env_ids)
```

-```def _get_dones(self)```
  - mark which environments need to be reset and why
  -  an “episode” ends in one of two ways: either the agent reaches a terminal state, or the episode reaches a maximum duration
    
-```def _reset_idx(self, env_ids: Sequence[int] | None):```
  -   indicating which scenes need to be reset, and resets them

## 2- Training the Jetbot: Ground Truth
- Modify rewards in order to train a policy to act as a controller for the Jetbot.
- As a user, we would like to use Reinforcement Learning be able to specify the desired direction for the Jetbot to drive, and have the wheels turn such that the robot drives in that specified direction as fast as possible

### Expanding the environment
- Create the logic for setting commands for each Jetbot on the stage
- Each command will be a unit vector, and we need one for every clone of the robot on the stage, which means a tensor of shape [num_envs, 3] (3D vectors)

#### Visualization Markers
  - Set up ```VisualizationMarkers``` to visualize the training in action. like Debug visuals
  - define the marker config and then instantiate the markers with that config
  - Add the following to the global scope of isaac_lab_tutorial_env.py:
    
  - include import libraries in the beginning
```
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
```
- include a new function **outside and before** the MyprojectEnv class:
```
def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)
```

- Setup data for tracking the commands as well as the marker positions and rotations. Replace the contents of _setup_scene with the following
```
def _setup_scene(self):
    self.robot = Articulation(self.cfg.robot_cfg)
    # add ground plane
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    # clone and replicate
    self.scene.clone_environments(copy_from_source=False)
    # add articulation to scene
    self.scene.articulations["robot"] = self.robot
    # add lights
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    self.visualization_markers = define_markers()

    # setting aside useful variables for later
    self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
    self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
    self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
    self.commands[:,-1] = 0.0
    self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

    # offsets to account for atan range and keep things on [-pi, pi]
    ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
    gzero = torch.where(self.commands > 0, True, False)
    lzero = torch.where(self.commands < 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
    self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
    self.marker_offset[:,-1] = 0.5
    self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
    self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
```
  - Define the visualize markers function **inside** the ```class MyprojectEnv(DirectRLEnv)``` and after def _setup_scene
```
 def _visualize_markers(self):
    # get marker locations and orientations
    self.marker_locations = self.robot.data.root_pos_w
    self.forward_marker_orientations = self.robot.data.root_quat_w
    self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

    # offset markers so they are above the jetbot
    loc = self.marker_locations + self.marker_offset
    loc = torch.vstack((loc, loc))
    rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

    # render the markers
    all_envs = torch.arange(self.cfg.scene.num_envs)
    indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
    self.visualization_markers.visualize(loc, rots, marker_indices=indices)
```

- call _visualize_markers on the pre physics step
- paste ```self._visualize_markers()``` inside
```
def _pre_physics_step(self, actions: torch.Tensor) -> None:
  self.actions = actions.clone()
  self._visualize_markers()
```

- update the _reset_idx method to account for the commands and markers
- Replace the contents of _reset_idx with the following:
```
def _reset_idx(self, env_ids: Sequence[int] | None):
    if env_ids is None:
        env_ids = self.robot._ALL_INDICES
    super()._reset_idx(env_ids)

    # pick new commands for reset envs
    self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
    self.commands[env_ids,-1] = 0.0
    self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

    # recalculate the orientations for the command markers with the new commands
    ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
    gzero = torch.where(self.commands[env_ids] > 0, True, False)
    lzero = torch.where(self.commands[env_ids]< 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    # set the root state for the reset envs
    default_root_state = self.robot.data.default_root_state[env_ids]
    default_root_state[:, :3] += self.scene.env_origins[env_ids]

    self.robot.write_root_state_to_sim(default_root_state, env_ids)
    self._visualize_markers()
```

### Exploring the RL problem


- 🎯 The Core Goal: Teaching the AI to *Drive*

- The main point is to set up the problem for our Reinforcement Learning (RL) AI (the agent).
- The agent is the AI brain, like the robot driver that observes (information about the current state + the goal) and acts based on the observation. It runs its internal logic (policy) and decides what action to take based on the difference (error) between the current state (current linear and angular velocities) and the future state (goal/command).
- We don't want to just "teleport" the robot to a new spot. We want to teach it *how* to take the correct **actions** (like spinning its wheels) to move from its **current state** (where it is now) to a **desired state** (the goal).
- The AI learns by trial and error. It needs to see the "error" (the difference between its current state and its goal) to learn which actions reduce that error.
- With **Reinforcement Learning**, it tells the agent what to do by: if error (difference between initial state and command) increases after action -> negative reinforcement. If error decreases after action -> positive reinforcement. 

#### 0\. The "Current Observation Space": (a 6-dimensional Observation vector)
Before updating this observation vector, the AI only knew how it was currently moving. This was its **"6-dimensional velocity vector"**:

- Linear Velocity X (how fast it's moving forward/backward)
- Linear Velocity Y (how fast it's moving left/right)
- Linear Velocity Z (how fast it's moving up/down)

- Angular Velocity X (how fast it's spinning around the X-axis)
- Angular Velocity Y (how fast it's spinning around the Y-axis)
- Angular Velocity Z (how fast it's spinning around the Z-axis)

#### 1\. The "Desired Future State"/Command Vector: The AI's Goal (a 3-dimensional Command vector)
  * **The Command:** Go to a new location, update robots linear velocity (x, y, z)
  * This is just a "goal" or "GPS instruction" we give the AI. In this case, it's a **"unit vector,"** which is simply a 3-number list (e.g., `[1, 0, 0]`) that points in a specific direction with a length of 1 - just one arrow pointing in one direction.

#### 2\. The "Future Observation Space": The AI's Dashboard (a 9-dimensional Observation vector)
  * This is the *entire set of information* the AI gets to see at every single step: it's current state and the desired state. Think of it as the AI's dashboard. This is the *only* thing it knows about the world.
  * we need to change the observation space to 9 numbers.
  * Initial observation vector: linear velocity (3 dimensions) + angular velocity (3 dimension) + command vector (3 dimension) = updated observation vector (9 dimensions)
  * If we were to update both linear and angular velocity it would result in a new observation vector with 12 dimensions. 

#### 3\. (The "Error" Calculation)
To learn, the AI needs to compare two separate things:

1.  **"Where am I *right now*?" (The Current State)**
2.  **"Where am I *supposed to* go?" (The Goal State)**

The AI's "brain" (the policy) is like the thermostat's logic: it calculates the "error" between these two and takes an action to fix it.

  * **The First 6 Numbers (Current State):** or the **"world velocity vector"**. This is the robot's "speedometer." It's a list of 6 values:

      * Linear Velocity (X, Y, Z) - How fast it's moving forward/back, left/right, up/down.
      * Angular Velocity (X, Y, Z) - How fast it's *spinning* or *tumbling*.
      * **What if we only had this?** The AI would be "numb." It would know it's moving, but it would be "blind" to the goal. It has no "error" to correct.

  * **The Next 3 Numbers (Goal State):** This is our **"command"** vector.

      * **What if we only had this?** The AI would be "blind" to its own state. It would know the goal but wouldn't be able to see if its actions were actually moving it closer to that goal.

By "appending the command to this vector," we are **gluing** these two lists together.

**`Total Observation = [6-number Current State] + [3-number Goal State] = 9-number vector`**

This 9-number "dashboard" gives the AI *everything* it needs to learn. It can now see both its state and its goal, calculate the error, and learn which actions (wheel movements) make its "Current State" numbers look more like its "Goal State" numbers.

-----

#### 4\. The Code, 

Navigate to ~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject# and open myproject_env.py
Replace the _get_observations method with the following:

```python
# This function's job is to gather all the data (observations) 
# our AI policy needs to make a decision.
def _get_observations(self) -> dict:

    # 1. GET THE "CURRENT STATE" (Numbers 1-6)
    #    `root_com_vel_w` stands for "root center-of-mass velocity in the world frame."
    #    This is our 6-number "speedometer reading" (linear + angular velocity).
    self.velocity = self.robot.data.root_com_vel_w 

    # 2. A SIDE-QUEST FOR LATER (Calculating the robot's "Forward" direction)
    #    This line is NOT part of the 9-number observation.
    #    It's used later to calculate the REWARD.
    #    `root_link_quat_w`: A "quaternion" is a 4-number value for a 3D rotation. This gets the robot's current rotation.
    #    `FORWARD_VEC_B`: This is just the vector `[1, 0, 0]` (the robot's "front").
    #    `math_utils.quat_apply`: This function applies the 3D rotation to the "front" vector.
    #    In short: This line figures out which way the robot is *actually* pointing in the 3D world.
    self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

    # 3. BUILD THE "DASHBOARD" (The 9-Number Vector)
    #    `torch.hstack` means "horizontal stack." It's the "glue."
    #    It takes our 6-number `self.velocity` vector and our 3-number `self.commands`
    #    vector and sticks them together, end-to-end, to make one 9-number list.
    obs = torch.hstack((self.velocity, self.commands)) 

    # 4. PACKAGE THE DATA
    #    The RL framework expects the observations in a "dictionary" (a labeled box).
    #    We put our 9-number list `obs` into a box labeled "policy"
    #    so the AI's "brain" knows where to find it.
    observations = {"policy": obs} 
    
    # 5. SEND IT TO THE AI
    return observations
```

### 🎯 The Core Goal: Creating a "Reward"
This whole section is about designing the reward signal for our AI agent. The reward is like giving the AI a "cookie" (a positive number) when it does something good and a "penalty" (a negative number) when it does something bad.

The AI's entire goal is to learn how to take actions that get it the most rewards over time.

Objective: just give the AI a reward for two things:

- 1. Forward Reward: Driving Forward: We want it to move fast towards the goal.
     - It's the robot's speed specifically along its own "forward" X-axis.
     - by taking the linear velocity of the robot measured taking the robot's center as a reference point: the robot's body frame (robots x,y,z coordinates) center of mass (where it can be balanced on the tip of a pencil)
     - The robot's linear velocity is the angle between the movement of its body frame center of mass vs the world's frame (inner product between the two)
       
- 2. Alignment Reward: Facing the Goal: We want it to point in the direction of the command.
     - The alignment term is the inner product between the forward vector and the command vector: when they are pointing in the same direction this term will be 1, but in the opposite direction it will be -1. We add them together to get the combined reward.
     - If two vectors are perfectly aligned (pointing the same way), the inner product is +1.
     - If they are perfectly misaligned (pointing opposite ways), the inner product is -1.
     - If they are at a 90° angle (perpendicular), the inner product is 0.

The hope is that by rewarding both of these at the same time, the AI will figure out that the best way to drive fast towards the goal.
