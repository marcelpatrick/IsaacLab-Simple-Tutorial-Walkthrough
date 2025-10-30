# IsaacLab-Simple-Tutorial-Walkthrough
from: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/index.html 

## Classes and Configs
Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/api_env_design.html

** When following the documentation, "isaac_lab_tutorial" should be replaced by your project name, in this case "myProject"

- Open an Ubuntu terminal
- Activate the conda environment created with Python 3.11 ```conda activate env_isaaclab```
- Navigate to ```cd ~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject```
- Check the environment configurations:
  - Open the config file (using Sublime in this example): ```subl myproject_env_cfg.py```
  - If you don't have sublime enabled, run: ```sudo snap install sublime-text --classic```

**-> you can also navigate to these folders on windows explorer by starting at the path: ```\\wsl.localhost\Ubuntu-22.04\root\IsaacSim\myProject```**

## Environment Design
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
- Save it with Ctrl X, Yes. The only purpose of this file is to define a unique scope in which to save our configurations.

### Environment Configuration
- Navigate to ```~/IsaacSim/myProject/source/myProject/myProject/tasks/direct/myproject#```
- Open myproject_env_cfg.py and replace its contenxt with
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
- **-> replace ```class MyprojectEnvCfg(DirectRLEnvCfg):``` with the name of your project, in this case ```class MyprojectEnvCfg(DirectRLEnvCfg):```**
- **-> replace ```from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG``` with 
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
- _pre_physics_step: getting data from the policy being trained and applying updates to the physics simulation
- The _apply_action method is where those actions are actually applied to the robots on the stage

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

### Training the Jetbot: Ground Truth
