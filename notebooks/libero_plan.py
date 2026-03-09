# closed loop planning for libero with VJEPA2-AC
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio

# ---- imports from vjepa2 ----
sys.path.insert(0, "..")
from app.vjepa_droid.transforms import make_transforms
from utils.world_model_wrapper import WorldModel

# ---- imports from libero (installed or sibling repo) ----
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    sys.path.insert(0, "../../lerobot-libero")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv


def resolve_bddl_path(bddl_path_from_meta: str) -> str:
    if os.path.exists(bddl_path_from_meta):
        return bddl_path_from_meta

    benchmark_root = get_libero_path("benchmark_root")
    candidate = os.path.join(benchmark_root, bddl_path_from_meta)
    if os.path.exists(candidate):
        return candidate

    if "bddl_files/" in bddl_path_from_meta:
        rel_under_bddl = bddl_path_from_meta.split("bddl_files/", 1)[1]
        candidate = os.path.join(get_libero_path("bddl_files"), rel_under_bddl)
        if os.path.exists(candidate):
            return candidate

    bddl_root = Path(get_libero_path("bddl_files"))
    basename = Path(bddl_path_from_meta).name
    matches = list(bddl_root.rglob(basename))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"Could not resolve BDDL file: {bddl_path_from_meta}")


class LiberoGymLikeEnv:
    def __init__(
        self,
        task,
        task_id,
        benchmark_instance,
        camera_name="droidview",
        fallback_camera="sideview",
        camera_height=256,
        camera_width=256,
    ):
        self.task = task
        self.task_id = task_id
        self.benchmark_instance = benchmark_instance
        self.camera_name = camera_name
        self.fallback_camera = fallback_camera
        self.camera_key = f"{camera_name}_image"

        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        if os.path.exists(task_bddl_file):
            bddl_file = task_bddl_file
        else:
            bddl_file = resolve_bddl_path(task.bddl_file)

        env_kwargs = {
            "bddl_file_name": bddl_file,
            "camera_heights": camera_height,
            "camera_widths": camera_width,
            "camera_depths": False,
            "use_camera_obs": True,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
        }

        try:
            self.env = OffScreenRenderEnv(
                **env_kwargs, camera_names=[self.fallback_camera, self.camera_name]
            )
            self.env.reset()
            print(f"Using preferred camera: {self.camera_name}")
        except ValueError as err:
            if self.camera_name in str(err):
                print(
                    f"Preferred camera '{self.camera_name}' unavailable. "
                    f"Falling back to '{self.fallback_camera}'."
                )
                self.env = OffScreenRenderEnv(
                    **env_kwargs, camera_names=[self.fallback_camera]
                )
                self.env.reset()
                self.camera_key = f"{self.fallback_camera}_image"
            else:
                raise

        self.init_states = None
        self.last_obs = None

    def reset(self, init_state_idx=0, seed=0):
        self.env.seed(seed)
        self.env.reset()
        self.init_states = self.benchmark_instance.get_task_init_states(self.task_id)
        self.last_obs = self.env.set_init_state(self.init_states[init_state_idx])
        return self.last_obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected action shape (7,), got {action.shape}")
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, done, info

    def get_image(self, obs=None):
        obs = self.last_obs if obs is None else obs
        if self.camera_key not in obs:
            image_keys = sorted([k for k in obs.keys() if k.endswith("_image")])
            raise KeyError(f"{self.camera_key} not in obs. Available image keys: {image_keys}")
        return np.ascontiguousarray(obs[self.camera_key][::-1])

    def get_pose_7d(self, obs=None):
        obs = self.last_obs if obs is None else obs
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)

        gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", [0.0]), dtype=np.float32)
        gripper = np.array([float(np.mean(gripper_qpos))], dtype=np.float32)
        return np.concatenate([pos, euler, gripper], axis=0)

    def close(self):
        self.env.close()


def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- task setup ----
    benchmark_name = "libero_spatial"
    task_id = 0
    init_state_idx = 0

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    task = benchmark_instance.get_task(task_id)
    print(f"Task {task_id}: {task.name}")
    print(f"Language: {task.language}")

    env = LiberoGymLikeEnv(
        task=task,
        task_id=task_id,
        benchmark_instance=benchmark_instance,
        camera_name="droidview",
        fallback_camera="sideview",
        camera_height=256,
        camera_width=256,
    )

    # ---- model setup ----
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    mpc_args = {
        "rollout": 1,
        "samples": 800,
        "topk": 10,
        "cem_steps": 10,
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": 0.075,
        "verbose": True,
    }

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args=mpc_args,
        normalize_reps=True,
        device=device,
    )
    # V-JEPA actions are interpreted as larger-time-step xyz deltas; map to LIBERO OSC input.
    hold_steps = 5
    xyz_scale = 4.0

    exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    planning_root = Path(__file__).resolve().parents[1] / "experiments" / "planning"
    run_dir = planning_root / exp_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    def to_libero_action(action):
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        action[:3] *= xyz_scale
        np.clip(action, -1.0, 1.0, out=action)
        return action

    # ---- build start + synthetic goal ----
    obs_start = env.reset(init_state_idx=init_state_idx, seed=0)
    start_frame = env.get_image(obs_start)
    start_pose = env.get_pose_7d(obs_start)

    probe_action = np.array([0.00, 0.00, -0.30, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    probe_repeat = 50
    obs_goal = None
    for _ in range(probe_repeat):
        obs_goal, _, _, _ = env.step(probe_action)
    goal_frame = env.get_image(obs_goal)
    goal_pose = env.get_pose_7d(obs_goal)

    print(f"Init pose (xyz,euler,gripper): {np.round(start_pose, 4)}")
    print(f"Goal pose (xyz,euler,gripper): {np.round(goal_pose, 4)}")

    init_goal_strip = np.concatenate([start_frame, goal_frame], axis=1)
    init_goal_path = run_dir / "init_goal.png"
    imageio.imwrite(init_goal_path, init_goal_strip)

    # reset again for planning rollout
    obs_start = env.reset(init_state_idx=init_state_idx, seed=0)
    start_frame = env.get_image(obs_start)
    start_pose = env.get_pose_7d(obs_start)

    # ---- first MPC step ----
    with torch.no_grad():
        z_n = world_model.encode(start_frame)
        z_goal = world_model.encode(goal_frame)
        s_n = torch.tensor(start_pose, dtype=torch.float32, device=device)[None, None]
        print("Starting MPC planning...")
        action_plan = world_model.infer_next_action(z_n, s_n, z_goal).detach().cpu().numpy()

    planned_action = to_libero_action(action_plan[0])
    for _ in range(hold_steps):
        obs_after, reward, done, info = env.step(planned_action)
    after_frame = env.get_image(obs_after)

    # ---- closed-loop replanning + frame recording ----
    current_obs = obs_after
    trajectory_frames = [start_frame, after_frame]
    trajectory_fps = 20
    closed_loop_steps = 5

    for t in range(closed_loop_steps):
        current_frame = env.get_image(current_obs)
        current_pose = env.get_pose_7d(current_obs)

        with torch.no_grad():
            z_n = world_model.encode(current_frame)
            s_n = torch.tensor(current_pose, dtype=torch.float32, device=device)[None, None]
            action_plan_t = world_model.infer_next_action(z_n, s_n, z_goal).detach().cpu().numpy()
            jepa_a = action_plan_t[0]
            libero_a = to_libero_action(jepa_a)

        for _ in range(hold_steps):
            current_obs, reward, done, info = env.step(libero_a)
            trajectory_frames.append(env.get_image(current_obs))
        print(f"step={t:02d} action={np.round(jepa_a, 4)} reward={reward} done={done}")

    # ---- save MP4 + GIF ----
    video_path = run_dir / "libero_planned_trajectory.mp4"
    gif_path = run_dir / "libero_planned_trajectory.gif"

    writer = imageio.get_writer(str(video_path), fps=trajectory_fps)
    for frame in trajectory_frames:
        writer.append_data(frame)
    writer.close()

    imageio.mimsave(
        str(gif_path),
        trajectory_frames,
        format="GIF",
        duration=1.0 / trajectory_fps,
        loop=0,
    )

    config_path = run_dir / "mpc_config.json"
    run_config = {
        "timestamp": exp_timestamp,
        "benchmark_name": benchmark_name,
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "init_state_idx": init_state_idx,
        "device": device,
        "camera_name": env.camera_name,
        "camera_key": env.camera_key,
        "probe_action": probe_action.tolist(),
        "probe_repeat": probe_repeat,
        "mpc_args": mpc_args,
        "hold_steps": hold_steps,
        "xyz_scale": xyz_scale,
        "trajectory_fps": trajectory_fps,
        "closed_loop_steps": closed_loop_steps,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved run directory: {run_dir}")
    print(f"Saved config: {config_path}")
    print(f"Saved init-goal image: {init_goal_path}")
    print(f"Saved video: {video_path}")
    print(f"Saved gif: {gif_path}")

    env.close()


if __name__ == "__main__":
    main()