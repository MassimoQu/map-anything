import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from data_processing.opv2v_pose_utils import (
    CARLA_TO_CAMERA_CV,
    cords_to_pose,
    get_camera_poses_in_ego,
    load_frame_metadata,
)
from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.geometry import get_pointmaps_and_rays_info


def _convert_pose_to_opencv(pose: np.ndarray) -> np.ndarray:
    pose_cv = pose.copy()
    basis = CARLA_TO_CAMERA_CV[:3, :3]
    pose_cv[:3, :3] = basis @ pose[:3, :3] @ basis.T
    pose_cv[:3, 3] = basis @ pose[:3, 3]
    return pose_cv


class CylindricalPanoramaBuilder:
    """Utility that warps multi-camera rigs into cylindrical panoramas."""

    def __init__(
        self,
        panorama_resolution: Sequence[int] = (1008, 252),
        vertical_fov_deg: float = 90.0,
        elevation_center_deg: float = 0.0,
        view_selection_margin: float = 0.05,
    ):
        pano_w, pano_h = panorama_resolution
        self.panorama_resolution = (int(pano_w), int(pano_h))
        v_fov = math.radians(vertical_fov_deg)
        elev_center = math.radians(elevation_center_deg)
        self.virtual_camera = dict(
            model="cyl",
            h_fov=2 * math.pi,
            v_fov=v_fov,
            elevation_center=elev_center,
        )
        self.virtual_intrinsics = np.array(
            [
                [pano_w / (2 * math.pi), 0.0, pano_w / 2.0],
                [0.0, pano_w / (2 * math.pi), pano_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.min_view_cos = float(view_selection_margin)
        self.virtual_rays = self._precompute_virtual_rays()

    def _precompute_virtual_rays(self) -> np.ndarray:
        pano_w, pano_h = self.panorama_resolution
        dummy_depth = np.ones((pano_h, pano_w), dtype=np.float32)
        (
            _,
            _,
            _,
            ray_directions_world,
            _,
            _,
            _,
        ) = get_pointmaps_and_rays_info(
            camera_model="cyl",
            depthmap=dummy_depth,
            camera_pose=np.eye(4, dtype=np.float32),
            virtual_camera=self.virtual_camera,
        )
        return ray_directions_world.astype(np.float32)

    @staticmethod
    def _bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        xs = np.clip(xs, 0.0, w - 1.001)
        ys = np.clip(ys, 0.0, h - 1.001)
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx = xs - x0
        wy = ys - y0
        wa = (1 - wx) * (1 - wy)
        wb = wx * (1 - wy)
        wc = (1 - wx) * wy
        wd = wx * wy

        Ia = image[y0, x0]
        Ib = image[y0, x1]
        Ic = image[y1, x0]
        Id = image[y1, x1]
        return (
            wa[..., None] * Ia
            + wb[..., None] * Ib
            + wc[..., None] * Ic
            + wd[..., None] * Id
        )

    def fuse(self, camera_views: List[Dict]) -> Dict[str, np.ndarray | Image.Image]:
        if not camera_views:
            raise ValueError("Need at least one camera view to build panorama")

        pano_w, pano_h = self.panorama_resolution
        rays_agent = self.virtual_rays

        fused_rgb = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        fused_depth = np.zeros((pano_h, pano_w), dtype=np.float32)
        weights = np.zeros((pano_h, pano_w), dtype=np.float32)
        valid_mask = np.zeros((pano_h, pano_w), dtype=bool)

        for cam in camera_views:
            img_np = np.asarray(cam["img"], dtype=np.float32)
            depth_np = cam["depthmap"].astype(np.float32)
            H_src, W_src = depth_np.shape
            intr = cam["camera_intrinsics"]
            rot_cam2agent = cam["camera_pose"][:3, :3]
            trans_cam2agent = cam["camera_pose"][:3, 3]
            rot_agent2cam = rot_cam2agent.T

            dir_cam = rays_agent @ rot_agent2cam.T
            z = dir_cam[..., 2]
            visible = z > self.min_view_cos

            x = dir_cam[..., 0] / np.clip(z, 1e-6, None)
            y = dir_cam[..., 1] / np.clip(z, 1e-6, None)
            u = x * intr[0, 0] + intr[0, 2]
            v = y * intr[1, 1] + intr[1, 2]
            inside = (
                (u >= 0.0)
                & (u <= W_src - 1.001)
                & (v >= 0.0)
                & (v <= H_src - 1.001)
            )
            valid = visible & inside
            if not np.any(valid):
                continue

            coords = np.where(valid)
            u_valid = u[coords]
            v_valid = v[coords]
            w = z[coords]
            sampled_rgb = self._bilinear_sample(img_np, u_valid, v_valid)
            sampled_depth = self._bilinear_sample(depth_np[..., None], u_valid, v_valid)[
                ..., 0
            ]

            fused_rgb[coords] += sampled_rgb * w[:, None]
            weights[coords] += w
            valid_mask[coords] = True

            unit_dir_cam = dir_cam[coords]
            points_cam = unit_dir_cam * sampled_depth[:, None]
            points_agent = points_cam @ rot_cam2agent.T + trans_cam2agent[None]
            depth_along_ray = np.maximum(
                np.sum(points_agent * rays_agent[coords], axis=1), 0.0
            )
            fused_depth[coords] += depth_along_ray * w

        denom = np.clip(weights, 1e-6, None)
        fused_rgb = fused_rgb / denom[..., None]
        fused_depth = fused_depth / denom
        fused_depth[~valid_mask] = 0.0

        panorama = np.clip(fused_rgb, 0.0, 255.0).astype(np.uint8)
        depthmap = fused_depth.astype(np.float32)

        return {
            "img": Image.fromarray(panorama),
            "depthmap": depthmap,
            "valid_mask": valid_mask,
        }


class _BaseOPV2VCylindricalDataset(BaseDataset):
    """Shared utilities for cylindrical OPV2V datasets."""

    def __init__(
        self,
        *args,
        split: str,
        depth_root: str,
        panorama_resolution: Sequence[int] = (1008, 252),
        panorama_vertical_fov_deg: float = 90.0,
        panorama_elevation_center_deg: float = 0.0,
        view_selection_margin: float = 0.05,
        **kwargs,
    ):
        if "resolution" in kwargs:
            kwargs["resolution"] = self._normalize_resolution(kwargs["resolution"])
        super().__init__(*args, split=split, **kwargs)
        self.split = split
        self.depth_root = Path(depth_root)
        self._panorama_builder = CylindricalPanoramaBuilder(
            panorama_resolution=panorama_resolution,
            vertical_fov_deg=panorama_vertical_fov_deg,
            elevation_center_deg=panorama_elevation_center_deg,
            view_selection_margin=view_selection_margin,
        )
        self.panorama_resolution = self._panorama_builder.panorama_resolution
        self.virtual_camera = dict(self._panorama_builder.virtual_camera)
        self._virtual_intrinsics = self._panorama_builder.virtual_intrinsics.copy()

    @staticmethod
    def _normalize_resolution(resolution):
        from collections.abc import Sequence

        if isinstance(resolution, Sequence) and not isinstance(resolution, (str, bytes)):
            if len(resolution) > 0 and isinstance(resolution[0], Sequence):
                return [tuple(res) for res in resolution]
            return tuple(resolution)
        return resolution

    def _load_camera_view(
        self,
        *,
        sequence: str,
        agent: str,
        frame_id: str,
        frame_meta: Dict,
        camera_poses: Dict[str, np.ndarray],
        cam_key: str,
        resolution,
        image_dir: Path,
    ) -> Dict[str, np.ndarray | Image.Image]:
        img_path = image_dir / f"{frame_id}_{cam_key}.png"
        depth_path = (
            self.depth_root
            / self.split
            / sequence
            / agent
            / f"{frame_id}_{cam_key}_depth.npy"
        )

        if not img_path.exists():
            raise FileNotFoundError(f"Image missing: {img_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth map missing: {depth_path}")

        image = Image.open(img_path).convert("RGB")
        depthmap = np.load(depth_path).astype(np.float32)
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        intrinsics = np.array(frame_meta[cam_key]["intrinsic"], dtype=np.float32)
        camera_pose = camera_poses[cam_key].astype(np.float32)
        camera_pose = _convert_pose_to_opencv(camera_pose)

        image, depthmap, intrinsics = self._crop_resize_if_necessary(
            image=image,
            resolution=resolution,
            depthmap=depthmap,
            intrinsics=intrinsics,
            additional_quantities=None,
        )

        return {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": camera_pose,
            "camera_intrinsics": intrinsics.astype(np.float32),
        }

    def _fuse_camera_views(self, camera_views: List[Dict]) -> Dict[str, np.ndarray | Image.Image]:
        return self._panorama_builder.fuse(camera_views)


class OPV2VCylindricalDataset(_BaseOPV2VCylindricalDataset):
    """Agent-centric OPV2V dataset that fuses 4 cameras into a panorama per sample."""

    def __init__(
        self,
        *args,
        ROOT: str,
        depth_root: str,
        split: str,
        camera_ids: Sequence[int] = (0, 1, 2, 3),
        panorama_resolution: Sequence[int] = (1024, 256),
        panorama_vertical_fov_deg: float = 90.0,
        panorama_elevation_center_deg: float = 0.0,
        view_selection_margin: float = 0.05,
        include_agents: Sequence[str] | None = None,
        max_scenes: int | None = None,
        **kwargs,
    ):
        kwargs.setdefault("num_views", 1)
        super().__init__(
            *args,
            split=split,
            depth_root=depth_root,
            panorama_resolution=panorama_resolution,
            panorama_vertical_fov_deg=panorama_vertical_fov_deg,
            panorama_elevation_center_deg=panorama_elevation_center_deg,
            view_selection_margin=view_selection_margin,
            **kwargs,
        )
        self.root = Path(ROOT)
        self.split = split
        self.camera_ids = [f"camera{cid}" for cid in camera_ids]
        self.include_agents = set(include_agents) if include_agents else None
        self.max_scenes = max_scenes
        self.dataset_name = "OPV2V"

        self._load_data()

        self.is_metric_scale = True
        self.is_synthetic = True

    def _load_data(self):
        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        scenes: List[Dict] = []
        for sequence_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
            for agent_dir in sorted(d for d in sequence_dir.iterdir() if d.is_dir()):
                agent_name = agent_dir.name
                if self.include_agents and agent_name not in self.include_agents:
                    continue

                yaml_files = sorted(agent_dir.glob("*.yaml"))
                for yaml_path in yaml_files:
                    frame_id = yaml_path.stem
                    if not frame_id.isdigit():
                        continue
                    scenes.append(
                        dict(
                            sequence=sequence_dir.name,
                            agent=agent_name,
                            frame=frame_id,
                            yaml_path=yaml_path,
                            image_dir=agent_dir,
                        )
                    )
                    if self.max_scenes and len(scenes) >= self.max_scenes:
                        break
                if self.max_scenes and len(scenes) >= self.max_scenes:
                    break
            if self.max_scenes and len(scenes) >= self.max_scenes:
                break

        if not scenes:
            raise RuntimeError(
                f"No OPV2V scenes found in split {self.split} under {split_root}"
            )

        self.scenes = scenes
        self.num_of_scenes = len(scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        if tuple(resolution) != tuple(self.panorama_resolution):
            raise ValueError(
                f"Configured panorama resolution {self.panorama_resolution} does not match requested resolution {resolution}."
            )

        scene_info = self.scenes[sampled_idx]
        frame_meta = load_frame_metadata(scene_info["yaml_path"])
        camera_poses = get_camera_poses_in_ego(frame_meta)

        camera_views = []
        for cam_key in self.camera_ids:
            if cam_key not in frame_meta:
                continue
            camera_views.append(
                self._load_camera_view(
                    sequence=scene_info["sequence"],
                    agent=scene_info["agent"],
                    frame_id=scene_info["frame"],
                    frame_meta=frame_meta,
                    camera_poses=camera_poses,
                    cam_key=cam_key,
                    resolution=resolution,
                    image_dir=scene_info["image_dir"],
                )
            )

        fused = self._fuse_camera_views(camera_views)
        lidar_pose = cords_to_pose(frame_meta["lidar_pose"]).astype(np.float32)
        agent_pose = _convert_pose_to_opencv(lidar_pose)

        view = dict(
            img=fused["img"],
            depthmap=fused["depthmap"],
            camera_pose=agent_pose.astype(np.float32),
            camera_intrinsics=self._virtual_intrinsics.copy(),
            camera_model="cyl",
            virtual_camera=dict(self.virtual_camera),
            dataset=self.dataset_name,
            label=os.path.join(scene_info["sequence"], scene_info["agent"]),
            instance=scene_info["frame"],
            non_ambiguous_mask=fused["valid_mask"].astype(np.uint8),
        )
        return [view]


class OPV2VCoopCylindricalDataset(_BaseOPV2VCylindricalDataset):
    """Multi-agent OPV2V loader returning one cylindrical panorama per vehicle."""

    def __init__(
        self,
        *args,
        ROOT: str,
        depth_root: str,
        split: str,
        camera_ids: Sequence[int] = (0, 1, 2, 3),
        include_agents: Sequence[str] | None = None,
        main_agent: str | None = None,
        main_agent_policy: str = "first",
        min_agents: int = 2,
        min_num_views: int | None = None,
        max_num_views: int | None = None,
        panorama_resolution: Sequence[int] = (1024, 256),
        panorama_vertical_fov_deg: float = 90.0,
        panorama_elevation_center_deg: float = 0.0,
        view_selection_margin: float = 0.05,
        max_scenes: int | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            split=split,
            depth_root=depth_root,
            panorama_resolution=panorama_resolution,
            panorama_vertical_fov_deg=panorama_vertical_fov_deg,
            panorama_elevation_center_deg=panorama_elevation_center_deg,
            view_selection_margin=view_selection_margin,
            **kwargs,
        )
        self.root = Path(ROOT)
        self.split = split
        self.camera_ids = [f"camera{cid}" for cid in camera_ids]
        self.include_agents = set(include_agents) if include_agents else None
        self.main_agent = main_agent
        self.main_agent_policy = main_agent_policy
        self.min_agents = max(1, int(min_agents))
        self.min_dynamic_views = max(1, int(min_num_views or 1))
        self.requested_max_num_views = max_num_views
        self.max_scenes = max_scenes
        self.dataset_name = "OPV2VCoopCyl"
        self.allow_variable_view_count = bool(self.variable_num_views)
        self.min_num_views_allowed = self.min_dynamic_views

        self._load_data()

        self.is_metric_scale = True
        self.is_synthetic = True
        self.max_views_per_scene = max(len(scene["agents"]) for scene in self.scenes)
        self.max_dynamic_views = (
            min(self.max_views_per_scene, self.requested_max_num_views)
            if self.requested_max_num_views is not None
            else self.max_views_per_scene
        )
        if isinstance(self.num_views, int):
            self.num_views = max(self.num_views, self.max_dynamic_views)
        else:
            self.num_views = list(
                range(self.min_dynamic_views, self.max_dynamic_views + 1)
            )

    def _load_data(self):
        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        scenes: List[Dict] = []
        for sequence_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
            agent_dirs = [d for d in sequence_dir.iterdir() if d.is_dir()]
            if self.include_agents:
                agent_dirs = [d for d in agent_dirs if d.name in self.include_agents]
            if len(agent_dirs) < self.min_agents:
                continue

            frame_to_agents: Dict[str, List[str]] = {}
            for agent_dir in agent_dirs:
                for yaml_path in agent_dir.glob("*.yaml"):
                    frame_id = yaml_path.stem
                    if not frame_id.isdigit():
                        continue
                    frame_to_agents.setdefault(frame_id, []).append(agent_dir.name)

            for frame_id, agents in sorted(frame_to_agents.items()):
                if len(agents) < self.min_agents:
                    continue
                agents_sorted = sorted(agents)
                agent_dir_map = {agent_id: sequence_dir / agent_id for agent_id in agents_sorted}
                scenes.append(
                    dict(
                        sequence=sequence_dir.name,
                        frame=frame_id,
                        agents=agents_sorted,
                        agent_dirs=agent_dir_map,
                    )
                )
                if self.max_scenes and len(scenes) >= self.max_scenes:
                    break
            if self.max_scenes and len(scenes) >= self.max_scenes:
                break

        if not scenes:
            raise RuntimeError(
                f"No cooperative OPV2V scenes found in split {self.split}"
            )

        self.scenes = scenes
        self.num_of_scenes = len(scenes)

    def _choose_main_agent(self, agents: List[str]) -> str:
        if self.main_agent and self.main_agent in agents:
            return self.main_agent
        if self.main_agent_policy == "random":
            return agents[int(self._rng.integers(0, len(agents)))]
        return sorted(agents)[0]

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        if tuple(resolution) != tuple(self.panorama_resolution):
            raise ValueError(
                f"Configured panorama resolution {self.panorama_resolution} does not match requested resolution {resolution}."
            )

        scene_info = self.scenes[sampled_idx]
        sequence = scene_info["sequence"]
        frame_id = scene_info["frame"]
        agent_dirs = scene_info["agent_dirs"]
        agents = list(scene_info["agents"])

        frame_meta_by_agent: Dict[str, Dict] = {}
        for agent_id in agents:
            yaml_path = agent_dirs[agent_id] / f"{frame_id}.yaml"
            if not yaml_path.exists():
                continue
            frame_meta_by_agent[agent_id] = load_frame_metadata(yaml_path)

        if len(frame_meta_by_agent) < self.min_agents:
            raise RuntimeError(
                f"Frame {sequence}/{frame_id} does not have enough agents after filtering"
            )

        available_agents = sorted(frame_meta_by_agent.keys())
        main_agent = self._choose_main_agent(available_agents)
        T_world_main = cords_to_pose(frame_meta_by_agent[main_agent]["lidar_pose"])
        T_main_world = np.linalg.inv(T_world_main)

        fused_entries = []
        for agent_id in available_agents:
            frame_meta = frame_meta_by_agent[agent_id]
            camera_poses = get_camera_poses_in_ego(frame_meta)
            camera_views = []
            for cam_key in self.camera_ids:
                if cam_key not in frame_meta:
                    continue
                try:
                    camera_views.append(
                        self._load_camera_view(
                            sequence=sequence,
                            agent=agent_id,
                            frame_id=frame_id,
                            frame_meta=frame_meta,
                            camera_poses=camera_poses,
                            cam_key=cam_key,
                            resolution=resolution,
                            image_dir=agent_dirs[agent_id],
                        )
                    )
                except FileNotFoundError:
                    camera_views = []
                    break
            if not camera_views:
                continue

            fused = self._fuse_camera_views(camera_views)
            lidar_pose = cords_to_pose(frame_meta["lidar_pose"]).astype(np.float32)
            agent_pose_main = T_main_world @ lidar_pose
            agent_pose_cv = _convert_pose_to_opencv(agent_pose_main)
            fused_entries.append(
                dict(
                    agent_id=agent_id,
                    fused=fused,
                    camera_pose=agent_pose_cv.astype(np.float32),
                )
            )

        total_available = len(fused_entries)
        if total_available < self.min_dynamic_views:
            raise ValueError(
                f"Need at least {self.min_dynamic_views} fused agents but got {total_available}"
            )

        if self.allow_variable_view_count:
            desired = int(num_views_to_sample)
            max_allowed = min(self.max_dynamic_views, total_available)
            if desired > max_allowed:
                raise ValueError(
                    f"Requested {desired} views but only {total_available} available for frame {scene_info}"
                )
            actual_num_views = desired
        else:
            if num_views_to_sample > total_available:
                raise ValueError(
                    f"Requested {num_views_to_sample} views but only {total_available} available"
                )
            actual_num_views = num_views_to_sample

        idx_perm = self._rng.permutation(total_available)
        selected_entries = [fused_entries[i] for i in idx_perm[:actual_num_views]]

        views = []
        for entry in selected_entries:
            fused = entry["fused"]
            view = dict(
                img=fused["img"],
                depthmap=fused["depthmap"],
                camera_pose=entry["camera_pose"],
                camera_intrinsics=self._virtual_intrinsics.copy(),
                camera_model="cyl",
                virtual_camera=dict(self.virtual_camera),
                dataset=self.dataset_name,
                label=os.path.join(sequence, main_agent),
                instance=os.path.join(frame_id, entry["agent_id"]),
                agent_id=entry["agent_id"],
                non_ambiguous_mask=fused["valid_mask"].astype(np.uint8),
            )
            views.append(view)

        return views
