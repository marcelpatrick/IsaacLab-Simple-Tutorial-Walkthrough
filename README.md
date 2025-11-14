# IsaacLab-Simple-Tutorial-Walkthrough
from: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/index.html 

### Project Settings
- This example is running on Anaconda Prompt CLI running on Windows11. 

### Pre-Requisites
- Before starting, make sure you have completed the prerequisites i.e. creating a virtual environment and a project with IsaacSim and IsaacLab installed as well as all other libraries e.g. Python 3.11 etc.
- You can find the step by step tutorial for the prerequisites for Windows11 native here: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial/blob/main/README.md

### Activate the environment
- Open Conda CLI: (on a conda cli: click on Windows search option, type “anaconda prompt”, click on it to open the cli)
- Activate the conda environment created with Python 3.11 ```conda activate env_isaaclab```

### Clone the Repo
- Navigate to ```(env_isaaclab) C:\Users\[YOUR INTERNAL PATH]\IsaacLab\source>``` (either on CLI or Windows explorer)
- Clone repo there ```git clone https://github.com/isaac-sim/IsaacLabTutorial.git isaac_lab_tutorial```
- Run ```dir``` and check that isaac_lab_tutorial is there

## 1- Understanding the main project files/scripts:

### 1.0- File Architecture: 

a) **CONFIG** file (IsaacLabTutorialEnvCfg): What is in the world and how it behaves?
- Defines what the simulation contains, the world and how it works: robot, physics, scene size, action/obs spaces.
- Defines what gets rendered in the simulation
- Allows to swap robots by just swapping this file.
- Allows to change environment conditions easily from this file: eg. reducing timesteps (for faster prototyping); apply stronger gravity to test with accurate conditions (full scale staging testing).
- Allows you to run simulations with different robots in different environments, but keeping the same RL logic among them.

b) **TASKS** file (IsaacLabTutorialEnv): What tasks and actions to perform to learn?
- Defines how the simulation runs: spawn robot, apply actions, compute rewards, reset.
- Defines the tasks the agents should accomplish and what is considered success (reward)
- Rewards are defined by what tasks the agent should accomplish - and this is defined by the env. Eg:
  - “balance the pole”
  - “move forward”
  - “stay upright”
  - “avoid collisions”
- Define the Episodes: one full attempt at the task.
  - Episode ends when robot accomplishes the goal (eg: arm grabs a defined object, robot reaches a location)
  - Episode ends when robot fails or violates a condition (eg: coliding with forbiden objects. too much deviation from a path)
  - Episode ends after X mins
  - Constrains: eg: accumulates too much torque (eg: trying to lift a box that is too heavy for him, keeps adding more torque and the arm doesn't move)

c) **LEARN*** Training Hyperparameters: How to learn it?
- if running SKRL: leave in a trainign script under ```scripts/skrl/train.py```
- if running RSL-RL: live in a YAML file usually ```rsl_rl_cfg/ppo.yaml```


### 1.1- Classes and Configs: Environment Configuration: isaac_lab_tutorial_env_cfg.py
Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html

- This file defines the configuration class for the Jetbot environment
- tells Isaac Lab how to build the virtual world where the Jetbot robot will live, move, and learn. defines what the world looks like
- It doesn’t run the simulation or training itself — it just describes all the parts of that world.
- It defines
  - which robot to load (Jetbot),
  - how the physics simulation should behave,
  - how long each episode should last,
  - how many parallel environments to create for training, and
  - what the robot’s control and observation spaces look like.
  
- Navigate to ```C:\Users\[YOUR USER]\IsaacLab\source\isaac_lab_tutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial```
- Check the environment configurations:
  - Open the config file. ```isaac_lab_tutorial_env_cfg```
  - Here is a description of each component of this file:
```python
# --- Imports ---
# Bring in Isaac Lab modules and the predefined Cartpole robot configuration
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG          # Cartpole robot configuration (predefined)
from isaaclab.assets import ArticulationCfg                       # Describes robot joints and physical setup
from isaaclab.envs import DirectRLEnvCfg                          # Base configuration for RL environments
from isaaclab.scene import InteractiveSceneCfg                    # Describes how the simulation world is arranged
from isaaclab.sim import SimulationCfg                             # Configures physics simulation settings
from isaaclab.utils import configclass                             # Marks this class as a configuration class for Isaac Lab


@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    """
    Defines the configuration for the Cartpole training environment.
    Specifies how the simulation behaves, which robot to use,
    and how the world is replicated for large-scale reinforcement learning.
    """

    # --- General / placeholder section ---
    # Other configuration fields can be defined here (reward, sensors, etc.)
    .
    .
    .


    # --- Simulation configuration ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,              # Physics time step: 1/120 sec per step (smooth simulation)
        render_interval=2        # Render every 2 simulation steps (improves speed)
    )
    # WHY IMPORTANT:
    # Defines how time and rendering progress in the simulated world.
    # A smaller dt = more accurate physics, render_interval controls visualization frequency.


    # --- Robot configuration ---
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"  # Location pattern where each robot is placed in the scene
    )
    # WHY IMPORTANT:
    # Loads the Cartpole robot model and tells Isaac Lab where to spawn it for each environment instance.
    # Without this, the environment would have no physical robot for the RL agent to control.


    # --- Scene configuration ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,           # Creates 4,096 environments in parallel
        env_spacing=4.0,         # Distance between each environment (to prevent overlap)
        replicate_physics=True   # All environments share the same physics configuration
    )
    # WHY IMPORTANT:
    # Controls large-scale parallel training, a major strength of Isaac Lab.
    # This setup allows thousands of Cartpoles to be trained at once for faster learning.


    # --- General / placeholder section ---
    # More configuration parameters (e.g., sensors, goals) can be added here later.
    .
    .
    .

```

**-> For custom project**
  
  **When following the documentation with a custom project, "isaac_lab_tutorial" should be replaced by your project name, eg. "myProject"**
  
  - Navigate to ```cd .../myProject/source/myProject/myProject/tasks/direct/myproject```
  - Check the environment configurations:
    - Open the config file (using Sublime in this example): ```subl myproject_env_cfg.py```
    - If you don't have sublime enabled, run: ```sudo snap install sublime-text --classic```


### 1.2- The Environment: isaac_lab_tutorial_env.py

- This file is where the logic of the training environment lives.
- While the config file (IsaacLabTutorialEnvCfg) defines what the world looks like,
this file (IsaacLabTutorialEnv) defines what the robot does during reinforcement learning.
- tells Isaac Lab how to use that simulation (configured by IsaacLabTutorialEnvCfg) for reinforcement learning.
- What it does:
  - It builds the scene (ground, lights, robot, clones of the environment).
  - It manages each training step, like applying robot actions and collecting sensor data.
  - It calculates rewards (how well the robot performed).
  - It handles resets when an episode ends or fails.

- In the same path ```C:\Users\[YOUR USER]\IsaacLab\source\isaac_lab_tutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial``` open ```isaac_lab_tutorial_env.py```

```python
# --- Imports ---
# Bring in the environment configuration we defined earlier (with sim, robot, and scene setup)
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg


# --- Environment Definition ---
class IsaacLabTutorialEnv(DirectRLEnv):
    # cgf Links this environment to its configuration class
    # cfg: IsaacLabTutorialEnvCfg is just an annotation that says “cfg is expected to be an instance of the class IsaacLabTutorialEnvCfg.” It throws warnings If you’re using a type checker like mypy, Pyright, or an IDE (VSCode, PyCharm)
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize the parent DirectRLEnv (sets up sim, scene, etc.)
        super().__init__(cfg, render_mode, **kwargs)
        # WHY IMPORTANT:
        # The DirectRLEnv base class provides the interface Isaac Lab and RL libraries expect.
        # Passing cfg ensures the environment uses the exact sim, robot, and scene we defined earlier.
        # This ties configuration and runtime behavior together cleanly.


    def _setup_scene(self):
        # Create the robot using its configuration (loads model into the scene)
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add a flat ground plane under the robot
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Register the robot in the scene so Isaac Lab tracks it
        self.scene.articulations["robot"] = self.robot

        # Clone and replicate the base environment to create all parallel copies
        self.scene.clone_environments(copy_from_source=False)

        # Add lighting so the scene is visible (useful for visualization)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # WHY IMPORTANT:
        # This builds the actual 3D world described in the config:
        # - loads the robot
        # - sets up physics entities (ground)
        # - duplicates the environment for parallel training
        # This is the foundation for scalable robot learning.


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Called before each physics step — prepares and validates actions
        # WHY IMPORTANT:
        # Separates action handling from simulation to keep updates consistent
        # and avoid conflicts during physics calculations.


    def _apply_action(self) -> None:
        # Actually applies the agent’s actions to the robot’s joints or motors
        # WHY IMPORTANT:
        # Translates the RL agent’s outputs into physical motion commands.
        # This is where the robot interacts with the simulated world.


    def _get_observations(self) -> dict:
        # Gathers sensory data or state info from the robot (e.g. position, velocity)
        # WHY IMPORTANT:
        # Provides feedback for the RL policy — the “eyes and ears” of the agent.


    def _get_rewards(self) -> torch.Tensor:
        # Calculates the reward signal based on task performance
        total_reward = compute_rewards(...)
        return total_reward
        # WHY IMPORTANT:
        # Defines the learning objective — tells the agent what behaviors are desirable.


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Checks if an episode is finished (success, failure, or time limit)
        # WHY IMPORTANT:
        # Ensures the simulation resets environments correctly and avoids infinite runs.


    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Resets the specified environments (e.g. after failure or episode end)
        # WHY IMPORTANT:
        # Refreshes robot and environment states for the next training episode.
        # Keeps training stable and continuous across thousands of environments.


# --- Reward Function ---
@torch.jit.script
def compute_rewards(...):
    # Computes total reward based on robot behavior and environment state
    return total_reward
    # WHY IMPORTANT:
    # Encapsulates reward logic in a compiled (JIT) function for speed and clarity.
    # Critical for defining what “success” means in robot learning.
```

**In Brief**:
1. The config class (IsaacLabTutorialEnvCfg) defines what exists
– simulation settings
– robot model
– scene layout

2. The environment class (IsaacLabTutorialEnv) defines how it behaves
– how to apply actions
– how to calculate rewards
– how to reset the world

The cfg variable connects those two worlds. It’s how the environment knows what world to build. It tells the new environment: 
- what robot to spawn,
- what physics settings to use,
- or how many parallel copies to make.


## 2- Environment Design
- Defining the environment
Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/technical_env_design.html

### 2.0- Define the Robot
- Inside ```C:\Users\[YOUR USER]\IsaacLab\source\isaac_lab_tutorial\source\isaac_lab_tutorial\isaac_lab_tutorial>``` Create a folder "robots" ```mkdir robots``` if it doesn't yet exist
- Within this folder create two files: __init__.py and jetbot.py. ```touch __init__.py jetbot.py```
- __init__.py makes the folder a Python package, and jetbot.py will be your module file.
- Open jetbot.py ```jetbot.py``` copy the following code inside:
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

### 2.1- Specifying Environment Configuration: isaac_lab_tutorial_env_cfg
- Navigate back to ```C:\Users\[YOUR USER]\IsaacLab\source\isaac_lab_tutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial```
- Open again ```isaac_lab_tutorial_env_cfg``` and replace its content with:
- 
```python
# --- Imports ---
# Bring in the Jetbot configuration and Isaac Lab modules needed for environment setup
from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG        # Predefined Jetbot model with its actuators
from isaaclab.assets import ArticulationCfg                       # Used to describe robot structure and joints
from isaaclab.envs import DirectRLEnvCfg                          # Base config for reinforcement learning environments
from isaaclab.scene import InteractiveSceneCfg                    # Defines how many environments and how they’re arranged
from isaaclab.sim import SimulationCfg                             # Defines the physics simulation settings
from isaaclab.utils import configclass                             # Marks this as a config class usable by Isaac Lab


@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    """
    Defines the configuration for the Jetbot training environment.
    This tells Isaac Lab what simulation to build, which robot to use,
    how long each episode lasts, and how many environments to train in parallel.
    """

    # --- Environment-level settings ---
    decimation = 2                    # Simulation renders every 2 steps (reduces computation cost)
    episode_length_s = 5.0            # Each training episode lasts 5 seconds
    # WHY IMPORTANT:
    # Controls simulation efficiency and how frequently the agent interacts with the world.


    # --- RL space definitions ---
    action_space = 2                  # Two actions: left and right wheel velocity
    observation_space = 3             # Three values observed (e.g. dot, cross, forward speed)
    state_space = 0                   # No internal state tracking needed
    # WHY IMPORTANT:
    # Defines how the RL agent sees and interacts with the world.
    # These must match the neural network input/output dimensions during training.


    # --- Simulation configuration ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,                   # Physics timestep (120 updates per second)
        render_interval=decimation    # Renders every 2 steps for faster training
    )
    # WHY IMPORTANT:
    # Creates the physical simulation environment — defines time stepping and rendering speed.


    # --- Robot setup ---
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"   # Places each Jetbot in its environment instance on the stage
    )
    # WHY IMPORTANT:
    # Loads the Jetbot model and tells Isaac Lab where to spawn it in each cloned environment.
    # The robot configuration defines what the agent will control.


    # --- Scene setup ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100,                 # Create 100 parallel environments
        env_spacing=4.0,              # Space each one 4 meters apart
        replicate_physics=True        # All share the same physics configuration
    )
    # WHY IMPORTANT:
    # Enables large-scale parallel training, speeding up RL data collection.


    # --- Robot joint names ---
    dof_names = ["left_wheel_joint", "right_wheel_joint"]
    # WHY IMPORTANT:
    # Defines which joints are controlled by the RL agent.
    # These correspond to the robot’s physical actuators in simulation.

```
- Save and close
- Here we have, effectively, the same environment configuration as before, but with the Jetbot instead of the cartpole


**-> for Custom Project**
  ### Environment Configuration
  - Navigate to ```~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject```
  - insert new code as above
  - **-> replace ```class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):``` with the name of your project, in this case ```class MyprojectEnvCfg(DirectRLEnvCfg):```**
  - **-> replace ```from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG``` with the name of your project, in this case: ```from myProject.robots.jetbot import JETBOT_CONFIG```
  - Save and close

---------- STOPPED HERE ----------

### 2.2- Setup Environment Details / Attack of the clones
- Navigate to ```C:\Users\[YOUR USER]\IsaacLab\source\isaac_lab_tutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial``` and open ```isaac_lab_tutorial_env.py```
- replace the contents of the __init__ and _setup_scene methods with the following.
```python
class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
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
- This part of the file defines the runtime logic for the robot training environment in Isaac Lab.
It extends DirectRLEnv, which is Isaac Lab’s base class for reinforcement learning environments.

- The configuration file (IsaacLabTutorialEnvCfg) defines what to simulate (robot type, physics, number of environments, etc.). This file (isaacLabTutorialEnv) defines how the simulation is built and initialized at runtime.

- Explaining each major component in this code:
| Part                                  | Purpose                                                | Why it’s important                                                        |
| ------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| **Class inheritance (`DirectRLEnv`)** | Provides RL-compatible lifecycle (step, reset, reward) | Integrates with Isaac Lab’s RL pipeline                                   |
| **`cfg` field**                       | Brings in simulation setup defined elsewhere           | Connects the environment logic with the configuration                     |
| **`self.dof_idx`**                    | Finds robot joints to control                          | Allows the RL agent to send motor commands to specific parts of the robot |
| **`_setup_scene()`**                  | Builds the virtual world and clones it                 | Creates the actual 3D simulation where learning happens                   |
| **Ground plane and lighting**         | Add realism and stability to the scene                 | Needed for physics accuracy and visualization                             |

- Joint (DoF) indexing means finding which specific joints of the robot you want to control based on their names (e.g., "left_wheel_joint", "right_wheel_joint").
Isaac Lab internally numbers all joints, so you must map your joint names → their numeric indices.

It’s relevant because the RL agent sends actions to joints by index, not by name.
Without this mapping, the environment wouldn’t know which motors the actions should affect — the robot wouldn’t move, or the wrong joints would be controlled.

**-> For Custom Project**
- Navigate to ```~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject#``` and open myproject_env.py
-  replace the contents of the __init__ and _setup_scene methods with the following
```python
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
```python
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self.actions = actions.clone()

def _apply_action(self) -> None:
    self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
```
- _pre_physics_step: getting data from the policy being trained and applying updates to the physics simulation ( lets us detach the process of getting data from the policy being trained and applying updates to the physics simulation)
- The _apply_action method is where those actions are actually applied to the robots on the stage

- Replace the contents of _get_observations and _get_rewards with the following
```python
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
```python
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
```python
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
```
- include a new function **outside and before** the MyprojectEnv class:
```python
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
```python
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
```python
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
```python
def _pre_physics_step(self, actions: torch.Tensor) -> None:
  self.actions = actions.clone()
  self._visualize_markers()
```

- update the _reset_idx method to account for the commands and markers
- Replace the contents of _reset_idx with the following:
```python
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
https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/training_jetbot_reward_exploration.html 

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

## Run
-> ```python scripts/skrl/train.py --task=Template-Isaac-Lab-Tutorial-Direct-v0```

- open an Ubuntu terminal and activate the conda environment, in this case: ```conda activate env_isaaclab"
- Navigate to ```cd /root/IsaacSim``` and run ```source _build/linux-x86_64/release/setup_conda_env.sh``` to link the IsaacLab project to the IsaacSim build so the terminal can find the IsaacSim module.
- navigate to the projects root folder ```cd /root/IsaacSim/myProject```
- run the training script.
  - Replace "Template-Isaac-Lab-Tutorial-Direct-v0" with the name of your project, in this case "MyProject": ```python scripts/skrl/train.py --task=Template-Myproject-Direct-v0```
