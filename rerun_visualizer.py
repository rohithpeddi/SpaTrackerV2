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
        self.data = dict(np.load(self.npz_path, allow_pickle=True))

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

    def _create_point_cloud_from_depth(self, depth_map, intrinsics, extrinsics, max_points=10000):
        """
        Create a point cloud from depth map using camera intrinsics and extrinsics.
        
        Args:
            depth_map: Depth map (H, W)
            intrinsics: Camera intrinsics matrix (3, 3)
            extrinsics: Camera extrinsics matrix (4, 4)
            max_points: Maximum number of points to sample
            
        Returns:
            points_3d: 3D points in world coordinates (N, 3)
            colors: RGB colors for points (N, 3)
        """
        if depth_map is None or intrinsics is None or extrinsics is None:
            return None, None
            
        height, width = depth_map.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create grid of pixel coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Filter valid depth values
        valid_mask = (depth_map > 0) & np.isfinite(depth_map)
        
        if not np.any(valid_mask):
            return None, None
            
        # Sample points if too many
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) > max_points:
            sample_indices = np.random.choice(len(valid_indices[0]), max_points, replace=False)
            y = y[valid_indices][sample_indices]
            x = x[valid_indices][sample_indices]
            depths = depth_map[valid_indices][sample_indices]
        else:
            y = y[valid_indices]
            x = x[valid_indices]
            depths = depth_map[valid_indices]
        
        # Convert to camera coordinates
        z = depths
        x_cam = (x - cx) * z / fx
        y_cam = (y - cy) * z / fy
        
        # Stack into homogeneous coordinates
        points_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
        
        # Transform to world coordinates
        points_world = (extrinsics @ points_cam.T).T
        points_3d = points_world[:, :3]
        
        # Create colors (you can modify this to use actual RGB values)
        # For now, using depth-based coloring
        normalized_depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
        colors = cv2.applyColorMap((normalized_depths * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        colors = colors.squeeze()  # Remove extra dimension
        
        return points_3d, colors

    def _log_point_cloud(self, frame_idx: int):
        """Log point cloud for a specific frame."""
        if "depths" not in self.data or "intrinsics" not in self.data or "extrinsics" not in self.data:
            return

        depths = self.data["depths"]
        intrinsics = self.data["intrinsics"]
        extrinsics = self.data["extrinsics"]
        
        if frame_idx >= len(depths):
            return

        depth_map = depths[frame_idx]
        frame_intrinsics = intrinsics[frame_idx]
        frame_extrinsics = extrinsics[frame_idx]
        
        # Create point cloud
        points_3d, colors = self._create_point_cloud_from_depth(
            depth_map, frame_intrinsics, frame_extrinsics, max_points=5000
        )
        
        if points_3d is not None and len(points_3d) > 0:
            # Log point cloud
            rr.log(f"world/point_cloud/frame_{frame_idx}", rr.Points3D(
                positions=points_3d,
                colors=colors,
                radii=[0.005] * len(points_3d)  # Small radius for points
            ))
            
            logger.info(f"Logged point cloud for frame {frame_idx}: {len(points_3d)} points")

    def visualize_frame(self, frame_idx: int):
        """Visualize a specific frame with all its data."""
        logger.info(f"Visualizing frame {frame_idx}")

        # Set the time for this frame
        rr.set_time_sequence("frame", frame_idx)
        self._log_point_cloud(frame_idx)  # Add point cloud visualization

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
