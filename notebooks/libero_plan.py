# closed loop planning for libero with VJEPA2-AC
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---- imports from vjepa2 ----
sys.path.insert(0, "..")
from app.vjepa_droid.transforms import make_transforms
from utils.world_model_wrapper import WorldModel
from utils.mpc_utils import compute_new_pose

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

    def reset_with_settle(self, init_state_idx=0, seed=0, settle_steps=0, settle_action=None):
        obs = self.reset(init_state_idx=init_state_idx, seed=seed)
        if settle_steps <= 0:
            return obs
        if settle_action is None:
            settle_action = np.zeros(7, dtype=np.float32)
        for _ in range(int(settle_steps)):
            obs, _, _, _ = self.step(settle_action)
        return obs

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
        camera_name="droidview2",
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
        "rollout": 8,
        "samples": 100,
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
    xyz_scale = 4.005
    settle_steps = 10
    settle_action = np.zeros(7, dtype=np.float32)

    exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    planning_root = Path(__file__).resolve().parents[1] / "experiments" / "planning"
    run_dir = planning_root / exp_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    def to_libero_action(action):
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        action[:3] *= xyz_scale
        np.clip(action, -1.0, 1.0, out=action)
        return action

    def latent_energy(rep, goal_rep):
        return float(torch.mean(torch.abs(rep - goal_rep)).item())

    def predict_next_rep_from_action(rep, pose_7d, action_7d):
        action_t = torch.tensor(action_7d, dtype=torch.float32, device=device)[None, None]
        pose_t = torch.tensor(pose_7d, dtype=torch.float32, device=device)[None, None]
        rep_t = rep[:, None]  # [B, T=1, N_tokens, D]
        next_rep = predictor(rep_t.flatten(1, 2), action_t, pose_t)[:, -tokens_per_frame:]
        if world_model.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        return next_rep

    def predict_rep_after_plan(rep, pose_7d, action_plan_7d):
        rep_curr = rep
        pose_curr = torch.tensor(pose_7d, dtype=torch.float32, device=device)[None, None]
        for h in range(action_plan_7d.shape[0]):
            action_t = torch.tensor(action_plan_7d[h], dtype=torch.float32, device=device)[None, None]
            rep_t = rep_curr[:, None]
            rep_next = predictor(rep_t.flatten(1, 2), action_t, pose_curr)[:, -tokens_per_frame:]
            if world_model.normalize_reps:
                rep_next = F.layer_norm(rep_next, (rep_next.size(-1),))
            pose_curr = compute_new_pose(pose_curr, action_t)
            rep_curr = rep_next
        return rep_curr

    # ---- build start + synthetic goal ----
    obs_start = env.reset_with_settle(
        init_state_idx=init_state_idx,
        seed=0,
        settle_steps=settle_steps,
        settle_action=settle_action,
    )
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

    # reset again for planning rollout (same settled start)
    obs_start = env.reset_with_settle(
        init_state_idx=init_state_idx,
        seed=0,
        settle_steps=settle_steps,
        settle_action=settle_action,
    )
    start_frame = env.get_image(obs_start)
    start_pose = env.get_pose_7d(obs_start)

    # ---- MPC diagnostics + closed-loop replanning + frame recording ----
    with torch.no_grad():
        z_goal = world_model.encode(goal_frame)
    current_obs = obs_start
    trajectory_frames = [start_frame]
    trajectory_fps = 20
    closed_loop_steps = 3*int(probe_repeat/hold_steps)
    planning_steps = 1 + closed_loop_steps
    action_history_jepa = []
    action_history_libero = []
    energy_debug = []

    print("Starting MPC planning with latent-energy diagnostics...")
    for t in range(planning_steps):
        current_frame = env.get_image(current_obs)
        current_pose = env.get_pose_7d(current_obs)

        with torch.no_grad():
            z_n = world_model.encode(current_frame)
            energy_before = latent_energy(z_n, z_goal)
            s_n = torch.tensor(current_pose, dtype=torch.float32, device=device)[None, None]
            action_plan_t = world_model.infer_next_action(z_n, s_n, z_goal).detach().cpu().numpy()
            jepa_a = action_plan_t[0]
            libero_a = to_libero_action(jepa_a)
            z_pred_after = predict_next_rep_from_action(z_n, current_pose, jepa_a)
            energy_pred_after = latent_energy(z_pred_after, z_goal)
            predicted_drop = energy_before - energy_pred_after
            z_pred_after_horizon = predict_rep_after_plan(z_n, current_pose, action_plan_t)
            energy_pred_after_horizon = latent_energy(z_pred_after_horizon, z_goal)
            predicted_drop_horizon = energy_before - energy_pred_after_horizon

        for _ in range(hold_steps):
            current_obs, reward, done, info = env.step(libero_a)
            trajectory_frames.append(env.get_image(current_obs))
        realized_frame = env.get_image(current_obs)
        with torch.no_grad():
            z_real_after = world_model.encode(realized_frame)
            energy_real_after = latent_energy(z_real_after, z_goal)
            realized_drop = energy_before - energy_real_after

        action_history_jepa.append(jepa_a.astype(np.float32))
        action_history_libero.append(libero_a.astype(np.float32))
        energy_debug.append(
            {
                "step": int(t),
                "energy_before": energy_before,
                "energy_pred_after": energy_pred_after,
                "predicted_drop": predicted_drop,
                "energy_pred_after_horizon": energy_pred_after_horizon,
                "predicted_drop_horizon": predicted_drop_horizon,
                "energy_real_after": energy_real_after,
                "realized_drop": realized_drop,
                "jepa_action": jepa_a.astype(np.float32).tolist(),
                "jepa_action_plan": action_plan_t.astype(np.float32).tolist(),
                "libero_action": libero_a.astype(np.float32).tolist(),
                "reward": float(reward),
                "done": bool(done),
            }
        )
        print(
            "step="
            f"{t:02d} "
            f"a_z(jepa)={jepa_a[2]:+.4f} "
            f"a_z(libero)={libero_a[2]:+.4f} "
            f"pred_drop_1={predicted_drop:+.6f} "
            f"pred_drop_H={predicted_drop_horizon:+.6f} "
            f"real_drop={realized_drop:+.6f} "
            f"reward={reward} done={done}"
        )

    # ---- save MP4 + GIF ----
    video_path = run_dir / "libero_planned_trajectory.mp4"
    gif_path = run_dir / "libero_planned_trajectory.gif"
    energy_json_path = run_dir / "energy_diagnostics.json"
    energy_plot_path = energy_json_path.with_name("energy_curve_with_predictions.png")
    actions_jepa_path = run_dir / "inferred_actions_jepa.npy"
    actions_libero_path = run_dir / "executed_actions_libero.npy"

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

    with open(energy_json_path, "w", encoding="utf-8") as f:
        json.dump(energy_debug, f, indent=2)

    actions_jepa = np.asarray(action_history_jepa, dtype=np.float32)
    actions_libero = np.asarray(action_history_libero, dtype=np.float32)
    np.save(actions_jepa_path, actions_jepa)
    np.save(actions_libero_path, actions_libero)

    n_steps = len(energy_debug)
    x_real = np.arange(n_steps + 1, dtype=np.float32)
    y_real = np.array(
        [energy_debug[0]["energy_before"]] + [d["energy_real_after"] for d in energy_debug],
        dtype=np.float32,
    )
    x_pred_1 = np.arange(1, n_steps + 1, dtype=np.float32)
    y_pred_1 = np.array([d["energy_pred_after"] for d in energy_debug], dtype=np.float32)
    y_pred_h = np.array(
        [d.get("energy_pred_after_horizon", d["energy_pred_after"]) for d in energy_debug],
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(x_real, y_real, marker="o", linewidth=2.0, label="Realized latent energy")
    plt.plot(
        x_pred_1,
        y_pred_1,
        linestyle=":",
        marker="x",
        linewidth=1.5,
        label="WM predicted energy after first action",
    )
    plt.plot(
        x_pred_1,
        y_pred_h,
        linestyle="--",
        marker="s",
        markersize=4,
        linewidth=1.2,
        label=f"WM predicted energy after horizon={mpc_args['rollout']}",
    )
    for i in range(n_steps):
        x = float(i + 1)
        plt.plot([x, x], [y_real[i + 1], y_pred_1[i]], linestyle=":", linewidth=1.0, color="gray")
    plt.xlabel("State index (after each MPC action)")
    plt.ylabel("Latent energy")
    plt.title("Realized energy curve with one-step WM projections")
    plt.legend()
    plt.tight_layout()
    plt.savefig(energy_plot_path, dpi=180)
    plt.close()

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
        "settle_steps": settle_steps,
        "settle_action": settle_action.tolist(),
        "trajectory_fps": trajectory_fps,
        "closed_loop_steps": closed_loop_steps,
        "planning_steps": planning_steps,
        "energy_diagnostics_json": str(energy_json_path),
        "energy_drop_plot": str(energy_plot_path),
        "inferred_actions_jepa": str(actions_jepa_path),
        "executed_actions_libero": str(actions_libero_path),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved run directory: {run_dir}")
    print(f"Saved config: {config_path}")
    print(f"Saved init-goal image: {init_goal_path}")
    print(f"Saved video: {video_path}")
    print(f"Saved gif: {gif_path}")
    print(f"Saved energy diagnostics: {energy_json_path}")
    print(f"Saved energy plot: {energy_plot_path}")
    print(f"Saved JEPA actions: {actions_jepa_path}")
    print(f"Saved executed LIBERO actions: {actions_libero_path}")

    env.close()


if __name__ == "__main__":
    main()