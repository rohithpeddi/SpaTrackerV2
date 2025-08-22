#!/usr/bin/env python3
"""
Rerun-based visualizer for SpaTrackerV2 results.npz files.
This visualizer provides interactive 3D visualization of tracking results using the rerun library.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerunVisualizer:
    """Interactive 3D visualizer for SpaTrackerV2 results using rerun."""

    def __init__(self, npz_path: str, fps: int = 10):
        """
        Initialize the rerun visualizer.
        
        Args:
            npz_path: Path to the results.npz file
            fps: Frames per second for playback
        """
        self.npz_path = Path(npz_path)
        self.fps = fps
        self.data = None
        self.frame_count = 0
        self.track_count = 0

        # Initialize rerun
        rr.init("SpaTrackerV2 Results", spawn=True)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load data from the npz file."""
        try:
            self.data = dict(np.load(self.npz_path, allow_pickle=True))
            logger.info(f"Loaded data from {self.npz_path}")

            # Log data structure
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"{key}: {value.shape} {value.dtype}")
                else:
                    logger.info(f"{key}: {type(value)}")

        except Exception as e:
            logger.error(f"Failed to load data from {self.npz_path}: {e}")
            raise

    def _setup_scene(self):
        """Set up the 3D scene with coordinate systems and cameras."""
        # Set up coordinate system
        rr.log("world", rr.Transform3D(translation=[0, 0, 0]))

        # Log camera intrinsics and extrinsics
        if "intrinsics" in self.data and "extrinsics" in self.data:
            intrinsics = self.data["intrinsics"]
            extrinsics = self.data["extrinsics"]

            # Log camera parameters for the first frame
            if len(intrinsics) > 0 and len(extrinsics) > 0:
                translation = extrinsics[0, :3, 3]  # Extract translation from extrinsics
                rotation_quat = R.from_matrix(extrinsics[0, :3, :3]).as_quat()
                rr.log("world/camera", rr.Transform3D(
                    translation=translation.tolist(),
                    rotation=rotation_quat.tolist()
                ))

                # Log camera intrinsics
                fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
                cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]
                width, height = intrinsics[0, 0, 2] * 2, intrinsics[0, 1, 2] * 2

                logger.info(f"Camera: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                logger.info(f"Image dimensions: {width:.0f}x{height:.0f}")

    def _log_trajectories(self, frame_idx: int):
        """Log 3D trajectories for a specific frame."""
        if "coords" not in self.data:
            return

        coords = self.data["coords"]
        if frame_idx >= len(coords):
            return

        # Get trajectories for current frame
        frame_trajs = coords[frame_idx]  # Shape: (N, 3)

        # Filter out invalid trajectories (NaN or inf)
        valid_mask = np.isfinite(frame_trajs).all(axis=1)
        valid_trajs = frame_trajs[valid_mask]

        if len(valid_trajs) == 0:
            return

        # Log trajectories as points
        rr.log(f"world/trajectories/frame_{frame_idx}", rr.Points3D(
            positions=valid_trajs,
            colors=[[255, 100, 100] for _ in range(len(valid_trajs))],  # Red color
            radii=[0.01] * len(valid_trajs)
        ))

        # Log trajectory IDs
        for i, traj in enumerate(valid_trajs):
            rr.log(f"world/trajectories/frame_{frame_idx}/traj_{i}", rr.TextDocument(
                text=f"Track {i}",
                media_type="text/plain"
            ))

    def _log_camera_pose(self, frame_idx: int):
        """Log camera pose for a specific frame."""
        if "extrinsics" not in self.data:
            return

        extrinsics = self.data["extrinsics"]
        if frame_idx >= len(extrinsics):
            return

        # Get camera pose for current frame
        camera_pose = extrinsics[frame_idx]

        # Log camera transform
        translation = camera_pose[:3, 3]  # Extract translation
        rotation_matrix = camera_pose[:3, :3]
        rotation_quat = R.from_matrix(rotation_matrix).as_quat()
        rr.log(f"world/camera/frame_{frame_idx}", rr.Transform3D(
            translation=translation.tolist(),
            rotation=rotation_quat.tolist()
        ))

        # Log camera as a coordinate system
        rr.log(f"world/camera/frame_{frame_idx}/axes", rr.Arrows3D(
            origins=[[0, 0, 0]],
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],  # X, Y, Z axes
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red, Green, Blue
        ))

    def _log_depth_map(self, frame_idx: int):
        """Log depth map for a specific frame."""
        if "depths" not in self.data:
            return

        depths = self.data["depths"]
        if frame_idx >= len(depths):
            return

        depth_map = depths[frame_idx]

        # Normalize depth for visualization
        valid_depths = depth_map[depth_map > 0]
        if len(valid_depths) == 0:
            return

        min_depth, max_depth = valid_depths.min(), valid_depths.max()
        if max_depth > min_depth:
            normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
            normalized_depth = np.clip(normalized_depth, 0, 1)

            # Convert to RGB colormap
            depth_rgb = cv2.applyColorMap(
                (normalized_depth * 255).astype(np.uint8),
                cv2.COLORMAP_INFERNO
            )

            # Log depth map as image
            rr.log(f"world/depth/frame_{frame_idx}", rr.Image(depth_rgb))

    def _log_video_frame(self, frame_idx: int):
        """Log RGB video frame for a specific frame."""
        if "video" not in self.data:
            return

        video = self.data["video"]
        if frame_idx >= len(video):
            return

        # Get frame and convert to uint8
        frame = video[frame_idx]  # Shape: (C, H, W)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        # Convert from (C, H, W) to (H, W, C) for rerun
        frame = np.transpose(frame, (1, 2, 0))

        # Log video frame
        rr.log(f"world/video/frame_{frame_idx}", rr.Image(frame))

    def _log_visibility_info(self, frame_idx: int):
        """Log visibility information for trajectories."""
        if "visibs" not in self.data:
            return

        visibs = self.data["visibs"]
        if frame_idx >= len(visibs):
            return

        visibility = visibs[frame_idx]  # Shape: (N,) or (N, 1)
        if visibility.ndim > 1:
            visibility = visibility.squeeze()

        # Count visible tracks
        visible_count = np.sum(visibility)
        total_count = len(visibility)

        # Log visibility statistics
        rr.log(f"world/stats/frame_{frame_idx}", rr.TextDocument(
            text=f"Frame {frame_idx}: {visible_count}/{total_count} tracks visible",
            media_type="text/plain"
        ))

    def visualize_frame(self, frame_idx: int):
        """Visualize a specific frame with all its data."""
        logger.info(f"Visualizing frame {frame_idx}")

        # Set the time for this frame
        rr.set_time_sequence("frame", frame_idx)

        # Log all components for this frame
        self._log_trajectories(frame_idx)
        self._log_camera_pose(frame_idx)
        self._log_depth_map(frame_idx)
        self._log_video_frame(frame_idx)
        self._log_visibility_info(frame_idx)

    def visualize_all_frames(self):
        """Visualize all frames in the dataset."""
        if "video" in self.data:
            total_frames = len(self.data["video"])
        elif "coords" in self.data:
            total_frames = len(self.data["coords"])
        else:
            logger.error("No frame data found")
            return

        logger.info(f"Visualizing {total_frames} frames")

        # Set up the scene
        self._setup_scene()

        # Visualize each frame
        for frame_idx in range(total_frames):
            self.visualize_frame(frame_idx)

        logger.info("Visualization complete!")

    def interactive_visualization(self):
        """Start interactive visualization with frame navigation."""
        if "video" in self.data:
            total_frames = len(self.data["video"])
        elif "coords" in self.data:
            total_frames = len(self.data["coords"])
        else:
            logger.error("No frame data found")
            return

        logger.info(f"Starting interactive visualization with {total_frames} frames")
        logger.info("Use the rerun viewer to navigate through frames")

        # Set up the scene
        self._setup_scene()

        # Log all frames at once for interactive navigation
        for frame_idx in range(total_frames):
            rr.set_time_sequence("frame", frame_idx)
            self._log_trajectories(frame_idx)
            self._log_camera_pose(frame_idx)
            self._log_depth_map(frame_idx)
            self._log_video_frame(frame_idx)
            self._log_visibility_info(frame_idx)

        logger.info("Interactive visualization ready! Use the rerun viewer to explore.")
        logger.info("Press Ctrl+C to exit")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Visualize SpaTrackerV2 results using rerun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        help="Path to the results.npz file"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for playback"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive visualization mode"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Visualize specific frame only"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.npz_path).exists():
        logger.error(f"File not found: {args.npz_path}")
        return

    try:
        # Create visualizer
        visualizer = RerunVisualizer(args.npz_path, args.fps)

        if args.frame is not None:
            # Visualize specific frame
            visualizer.visualize_frame(args.frame)
        elif args.interactive:
            # Start interactive mode
            visualizer.interactive_visualization()
        else:
            # Visualize all frames
            visualizer.visualize_all_frames()

    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
