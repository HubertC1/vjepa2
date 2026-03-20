# closed loop planning for libero with VJEPA2-AC
import argparse
import os
import sys
import re
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

    def reset_to_state(self, state):
        self.env.reset()
        self.last_obs = self.env.set_init_state(state)
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


def collect_subgoal_pairs(debug_dir: str):
    frame_regex = re.compile(
        r"^subgoal_(?P<sid>\d+)_rel_(?P<rel>\d+)_abs_(?P<abs>\d+)\.(png|jpg|jpeg)$",
        flags=re.IGNORECASE,
    )
    root = Path(debug_dir)
    if not root.exists():
        raise FileNotFoundError(f"subgoal debug dir does not exist: {root}")

    entries = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        m = frame_regex.match(p.name)
        if m is None:
            continue
        sid = int(m.group("sid"))
        state_path = root / f"state_{p.stem}.npy"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing state file for {p.name}: {state_path}")
        entries.append({"sid": sid, "frame_path": p, "state_path": state_path})

    if len(entries) < 2:
        raise ValueError(
            f"Need at least 2 subgoal frames to form pairs, found {len(entries)} in {root}"
        )
    entries.sort(key=lambda e: e["sid"])

    pairs = []
    for i in range(len(entries) - 1):
        a = entries[i]
        b = entries[i + 1]
        pairs.append(
            {
                "pair_index": i,
                "init_sid": a["sid"],
                "goal_sid": b["sid"],
                "init_frame_path": str(a["frame_path"]),
                "init_state_path": str(a["state_path"]),
                "goal_frame_path": str(b["frame_path"]),
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO planning with V-JEPA2-AC.")
    parser.add_argument("--benchmark_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--init_state_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera_name", type=str, default="droidview2")
    parser.add_argument("--fallback_camera", type=str, default="sideview")
    parser.add_argument("--hold_steps", type=int, default=5)
    parser.add_argument("--xyz_scale", type=float, default=4.005)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--planning_steps", type=int, default=None)
    parser.add_argument("--probe_repeat", type=int, default=50)
    parser.add_argument(
        "--subgoal_debug_dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing subgoal frames and per-frame state_*.npy files. "
            "If set, planner runs pairwise over consecutive subgoals."
        ),
    )
    parser.add_argument(
        "--subgoal_init_frame",
        type=str,
        default=None,
        help="Optional path to subgoal start frame PNG/JPG.",
    )
    parser.add_argument(
        "--subgoal_goal_frame",
        type=str,
        default=None,
        help="Optional path to next-subgoal goal frame PNG/JPG.",
    )
    parser.add_argument(
        "--subgoal_init_state",
        type=str,
        default=None,
        help="Optional .npy containing flattened MuJoCo state for subgoal start.",
    )
    args, _ = parser.parse_known_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- task setup ----
    benchmark_name = args.benchmark_name
    task_id = int(args.task_id)
    init_state_idx = int(args.init_state_idx)

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    task = benchmark_instance.get_task(task_id)
    print(f"Task {task_id}: {task.name}")
    print(f"Language: {task.language}")

    env = LiberoGymLikeEnv(
        task=task,
        task_id=task_id,
        benchmark_instance=benchmark_instance,
        camera_name=args.camera_name,
        fallback_camera=args.fallback_camera,
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
        "rollout": 4,
        "samples": 10,
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
    hold_steps = max(int(args.hold_steps), 1)
    xyz_scale = float(args.xyz_scale)
    settle_steps = max(int(args.settle_steps), 0)
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
        rep = rep.flatten(1)
        goal_rep = goal_rep.flatten(1)
        return torch.mean(torch.abs(goal_rep - rep), dim=-1).item()

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

    probe_action = np.array([0.00, 0.00, -0.30, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    probe_repeat = int(max(args.probe_repeat, 1))

    def run_single_plan(
        run_dir_local: Path,
        goal_frame: np.ndarray,
        reset_start_fn,
        start_frame_for_strip: np.ndarray,
        planning_steps_local: int,
        pair_meta: dict,
    ):
        obs_start = reset_start_fn()
        start_frame_runtime = env.get_image(obs_start)
        start_pose = env.get_pose_7d(obs_start)

        init_goal_strip = np.concatenate([start_frame_for_strip, goal_frame], axis=1)
        init_goal_path = run_dir_local / "init_goal.png"
        imageio.imwrite(init_goal_path, init_goal_strip)

        with torch.no_grad():
            z_goal = world_model.encode(goal_frame)
        current_obs = obs_start
        trajectory_frames = [start_frame_runtime]
        trajectory_fps = 20
        action_history_jepa = []
        action_history_libero = []
        energy_debug = []

        print("Starting MPC planning with latent-energy diagnostics...")
        for t in range(planning_steps_local):
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

        video_path = run_dir_local / "libero_planned_trajectory.mp4"
        gif_path = run_dir_local / "libero_planned_trajectory.gif"
        energy_json_path = run_dir_local / "energy_diagnostics.json"
        energy_plot_path = energy_json_path.with_name("energy_curve_with_predictions.png")
        actions_jepa_path = run_dir_local / "inferred_actions_jepa.npy"
        actions_libero_path = run_dir_local / "executed_actions_libero.npy"

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

        run_config = {
            "timestamp": exp_timestamp,
            "benchmark_name": benchmark_name,
            "task_id": task_id,
            "task_name": task.name,
            "task_language": task.language,
            "init_state_idx": init_state_idx,
            "seed": args.seed,
            "device": device,
            "camera_name": env.camera_name,
            "camera_key": env.camera_key,
            "probe_action": None if pair_meta.get("is_subgoal", False) else probe_action.tolist(),
            "probe_repeat": None if pair_meta.get("is_subgoal", False) else probe_repeat,
            "subgoal_debug_dir": pair_meta.get("subgoal_debug_dir"),
            "subgoal_init_frame": pair_meta.get("subgoal_init_frame"),
            "subgoal_goal_frame": pair_meta.get("subgoal_goal_frame"),
            "subgoal_init_state": pair_meta.get("subgoal_init_state"),
            "subgoal_pair_index": pair_meta.get("pair_index"),
            "subgoal_init_sid": pair_meta.get("init_sid"),
            "subgoal_goal_sid": pair_meta.get("goal_sid"),
            "mpc_args": mpc_args,
            "hold_steps": hold_steps,
            "xyz_scale": xyz_scale,
            "settle_steps": settle_steps,
            "settle_action": settle_action.tolist(),
            "trajectory_fps": trajectory_fps,
            "planning_steps": planning_steps_local,
            "energy_diagnostics_json": str(energy_json_path),
            "energy_drop_plot": str(energy_plot_path),
            "inferred_actions_jepa": str(actions_jepa_path),
            "executed_actions_libero": str(actions_libero_path),
        }
        config_path = run_dir_local / "mpc_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        print(f"Saved run directory: {run_dir_local}")
        print(f"Saved config: {config_path}")
        print(f"Saved init-goal image: {init_goal_path}")
        print(f"Saved video: {video_path}")
        print(f"Saved gif: {gif_path}")
        print(f"Saved energy diagnostics: {energy_json_path}")
        print(f"Saved energy plot: {energy_plot_path}")
        print(f"Saved JEPA actions: {actions_jepa_path}")
        print(f"Saved executed LIBERO actions: {actions_libero_path}")

    if args.subgoal_debug_dir is not None:
        pairs = collect_subgoal_pairs(args.subgoal_debug_dir)
        print(f"Using subgoal debug directory: {args.subgoal_debug_dir}")
        print(f"Found {len(pairs)} consecutive subgoal pairs")
        default_steps = max(int(args.planning_steps), 1) if args.planning_steps is not None else 31

        for pair in pairs:
            pair_name = (
                f"pair_{pair['pair_index']:03d}_"
                f"subgoal_{pair['init_sid']:03d}_to_{pair['goal_sid']:03d}"
            )
            pair_run_dir = run_dir / pair_name
            pair_run_dir.mkdir(parents=True, exist_ok=True)

            subgoal_state = np.load(pair["init_state_path"])
            start_frame_strip = imageio.imread(pair["init_frame_path"])
            goal_frame = imageio.imread(pair["goal_frame_path"])
            pair_meta = {
                "is_subgoal": True,
                "subgoal_debug_dir": args.subgoal_debug_dir,
                "subgoal_init_frame": pair["init_frame_path"],
                "subgoal_goal_frame": pair["goal_frame_path"],
                "subgoal_init_state": pair["init_state_path"],
                "pair_index": pair["pair_index"],
                "init_sid": pair["init_sid"],
                "goal_sid": pair["goal_sid"],
            }
            print(
                f"\n=== Running {pair_name} "
                f"({Path(pair['init_frame_path']).name} -> {Path(pair['goal_frame_path']).name}) ==="
            )
            run_single_plan(
                run_dir_local=pair_run_dir,
                goal_frame=goal_frame,
                reset_start_fn=lambda st=subgoal_state: env.reset_to_state(st),
                start_frame_for_strip=start_frame_strip,
                planning_steps_local=default_steps,
                pair_meta=pair_meta,
            )
    else:
        use_subgoal_pair = args.subgoal_init_frame is not None and args.subgoal_goal_frame is not None
        if use_subgoal_pair:
            if args.subgoal_init_state is None:
                raise ValueError(
                    "When using --subgoal_init_frame/--subgoal_goal_frame, "
                    "--subgoal_init_state is also required."
                )
            subgoal_state = np.load(args.subgoal_init_state)
            start_frame_strip = imageio.imread(args.subgoal_init_frame)
            goal_frame = imageio.imread(args.subgoal_goal_frame)
            reset_start_fn = lambda st=subgoal_state: env.reset_to_state(st)
            planning_steps_local = max(int(args.planning_steps), 1) if args.planning_steps is not None else 31
            pair_meta = {
                "is_subgoal": True,
                "subgoal_debug_dir": None,
                "subgoal_init_frame": args.subgoal_init_frame,
                "subgoal_goal_frame": args.subgoal_goal_frame,
                "subgoal_init_state": args.subgoal_init_state,
                "pair_index": 0,
                "init_sid": None,
                "goal_sid": None,
            }
        else:
            obs_start = env.reset_with_settle(
                init_state_idx=init_state_idx,
                seed=args.seed,
                settle_steps=settle_steps,
                settle_action=settle_action,
            )
            start_frame_strip = env.get_image(obs_start)
            obs_goal = None
            for _ in range(probe_repeat):
                obs_goal, _, _, _ = env.step(probe_action)
            goal_frame = env.get_image(obs_goal)
            reset_start_fn = (
                lambda: env.reset_with_settle(
                    init_state_idx=init_state_idx,
                    seed=args.seed,
                    settle_steps=settle_steps,
                    settle_action=settle_action,
                )
            )
            closed_loop_steps = 3 * int(probe_repeat / hold_steps)
            planning_steps_local = (
                max(int(args.planning_steps), 1)
                if args.planning_steps is not None
                else 1 + closed_loop_steps
            )
            pair_meta = {"is_subgoal": False}

        run_single_plan(
            run_dir_local=run_dir,
            goal_frame=goal_frame,
            reset_start_fn=reset_start_fn,
            start_frame_for_strip=start_frame_strip,
            planning_steps_local=planning_steps_local,
            pair_meta=pair_meta,
        )

    env.close()


if __name__ == "__main__":
    main()