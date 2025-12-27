#!/usr/bin/env python3
"""
Visualize the registered slice position within the volume.

This script shows where the registered slice is located in the volume
by displaying volume slices with an overlay line/plane indicating the slice position.
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Add NH-RS2V paths to Python path
NH_RS2V_BASELINES_PATH = Path("/scratch/users/Proulx-S/tools/NH-RS2V-baselines")
NH_RS2V_DATASET_PATH = Path("/scratch/users/Proulx-S/tools/NH-RS2V-dataset")

for path in [NH_RS2V_DATASET_PATH, NH_RS2V_BASELINES_PATH]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Try to import NH-RS2V modules
NH_RS2V_AVAILABLE = False
S2VTransform = None
sample_slice = None

try:
    # Set up module structure first
    import types
    if 'nh_rs2v_dataset' not in sys.modules:
        sys.modules['nh_rs2v_dataset'] = types.ModuleType('nh_rs2v_dataset')
    if 'nh_rs2v_dataset.utils' not in sys.modules:
        sys.modules['nh_rs2v_dataset.utils'] = types.ModuleType('nh_rs2v_dataset.utils')
    
    import importlib.util
    
    # Import invert_4x4 first (needed by transforms and g_s_t)
    invert_path = NH_RS2V_DATASET_PATH / "nh_rs2v_dataset" / "utils" / "invert_4x4.py"
    if invert_path.exists():
        spec_inv = importlib.util.spec_from_file_location("nh_rs2v_dataset.utils.invert_4x4", invert_path)
        inv_module = importlib.util.module_from_spec(spec_inv)
        inv_module.np = np
        sys.modules['nh_rs2v_dataset.utils.invert_4x4'] = inv_module
        spec_inv.loader.exec_module(inv_module)
    
    # Import g_s_t (needed by sampling)
    g_s_t_path = NH_RS2V_DATASET_PATH / "nh_rs2v_dataset" / "utils" / "g_s_t.py"
    if g_s_t_path.exists():
        spec_gst = importlib.util.spec_from_file_location("nh_rs2v_dataset.utils.g_s_t", g_s_t_path)
        gst_module = importlib.util.module_from_spec(spec_gst)
        gst_module.np = np
        sys.modules['nh_rs2v_dataset.utils.g_s_t'] = gst_module
        spec_gst.loader.exec_module(gst_module)
    
    # Import transforms
    transforms_path = NH_RS2V_DATASET_PATH / "nh_rs2v_dataset" / "utils" / "transforms.py"
    if transforms_path.exists():
        spec = importlib.util.spec_from_file_location("nh_rs2v_dataset.utils.transforms", transforms_path)
        transforms_module = importlib.util.module_from_spec(spec)
        transforms_module.np = np
        from scipy.spatial.transform import Rotation
        transforms_module.Rotation = Rotation
        sys.modules['nh_rs2v_dataset.utils.transforms'] = transforms_module
        spec.loader.exec_module(transforms_module)
        S2VTransform = transforms_module.S2VTransform
    
    # Import sampling
    sampling_path = NH_RS2V_DATASET_PATH / "nh_rs2v_dataset" / "utils" / "sampling.py"
    if sampling_path.exists():
        spec = importlib.util.spec_from_file_location("nh_rs2v_dataset.utils.sampling", sampling_path)
        sampling_module = importlib.util.module_from_spec(spec)
        sampling_module.np = np
        sys.modules['nh_rs2v_dataset.utils.sampling'] = sampling_module
        spec.loader.exec_module(sampling_module)
        sample_slice = sampling_module.sample_slice
    
    if S2VTransform is not None and sample_slice is not None:
        NH_RS2V_AVAILABLE = True
except Exception as e:
    print(f"Warning: NH-RS2V modules not available: {e}")
    NH_RS2V_AVAILABLE = False


def load_registration_result(result_file="registration_result_quick.json"):
    """Load registration result from JSON file."""
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    R = np.array(result['rotation_matrix'])
    T = np.array(result['translation_vector'])
    
    return R, T


def load_data(volume_path="volume_3d.npy", metadata_path="data_metadata.json"):
    """Load volume and metadata."""
    volume_data = np.load(volume_path)
    
    metadata = None
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return volume_data, metadata


def get_slice_plane_points(R, T, size=256, volume_shape=None):
    """
    Get points that define the slice plane in volume coordinates.
    
    Returns the corners of the slice plane in volume space.
    Uses identity_2_volume to properly transform from slice identity space.
    """
    if not NH_RS2V_AVAILABLE:
        return None
    
    # Create S2VTransform from R, T
    S_gt = np.array([1.0, 1.0], dtype=np.float32)
    s2v_transform = S2VTransform.from_RTS(size, R, T, S_gt)
    
    # Define corners of the slice in slice identity space (2D, centered at origin)
    # Slice coordinates go from 0.5 to size-0.5
    slice_corners_2d = np.array([
        [0.5, 0.5],      # Top-left
        [size-0.5, 0.5],  # Top-right
        [size-0.5, size-0.5],  # Bottom-right
        [0.5, size-0.5],  # Bottom-left
    ], dtype=np.float64)
    
    # Convert to 3D points in slice identity space (z=0)
    slice_corners_3d = np.hstack([
        slice_corners_2d,
        np.zeros((4, 1), dtype=np.float64)  # z = 0
    ])
    
    # Transform to volume space using identity_2_volume
    # This properly handles the canonical coordinate transformations
    volume_corners = s2v_transform.identity_2_volume(slice_corners_3d)
    
    return volume_corners


def get_slice_plane_intersection(volume_shape, R, T, size=256, axis='z', slice_idx=None):
    """
    Get the intersection of the slice plane with a volume slice.
    
    Parameters:
    -----------
    volume_shape : tuple
        Shape of the volume (nx, ny, nz)
    R : np.ndarray
        3x3 rotation matrix
    T : np.ndarray
        3x1 translation vector
    size : int
        Slice sampling size
    axis : str
        'x', 'y', or 'z' - which axis to slice along
    slice_idx : int
        Index of the slice along the axis (if None, uses middle)
    
    Returns:
    --------
    intersection_points : np.ndarray
        Points where the slice plane intersects the volume slice
    """
    if not NH_RS2V_AVAILABLE:
        return None
    
    # Get slice plane corners
    plane_corners = get_slice_plane_points(R, T, size, volume_shape)
    if plane_corners is None:
        return None
    
    # Get the normal vector of the plane (from two edges)
    edge1 = plane_corners[1] - plane_corners[0]
    edge2 = plane_corners[3] - plane_corners[0]
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    # Get a point on the plane (center)
    plane_point = np.mean(plane_corners, axis=0)
    
    # Determine which axis to slice along
    if axis == 'z':
        if slice_idx is None:
            slice_idx = volume_shape[2] // 2
        z_coord = slice_idx
        
        # Find intersection of plane with z = slice_idx
        # Plane equation: normal . (p - plane_point) = 0
        # For z = z_coord: normal[2] * (z_coord - plane_point[2]) + normal[0] * (x - plane_point[0]) + normal[1] * (y - plane_point[1]) = 0
        # Solve for y given x, or find intersection with volume boundaries
        
        # Get intersection with volume boundaries
        intersections = []
        # Check intersections with each edge of the volume slice
        nx, ny, nz = volume_shape
        
        # Intersection with x=0, x=nx-1, y=0, y=ny-1 boundaries
        for x in [0, nx-1]:
            if abs(normal[0]) > 1e-6:
                y = plane_point[1] - (normal[0] * (x - plane_point[0]) + normal[2] * (z_coord - plane_point[2])) / normal[1]
                if 0 <= y < ny:
                    intersections.append([x, y, z_coord])
        
        for y in [0, ny-1]:
            if abs(normal[1]) > 1e-6:
                x = plane_point[0] - (normal[1] * (y - plane_point[1]) + normal[2] * (z_coord - plane_point[2])) / normal[0]
                if 0 <= x < nx:
                    intersections.append([x, y, z_coord])
        
        if len(intersections) >= 2:
            return np.array(intersections[:2])  # Return first two points (line segment)
    
    elif axis == 'y':
        if slice_idx is None:
            slice_idx = volume_shape[1] // 2
        y_coord = slice_idx
        
        intersections = []
        nx, ny, nz = volume_shape
        
        for x in [0, nx-1]:
            if abs(normal[0]) > 1e-6:
                z = plane_point[2] - (normal[0] * (x - plane_point[0]) + normal[1] * (y_coord - plane_point[1])) / normal[2]
                if 0 <= z < nz:
                    intersections.append([x, y_coord, z])
        
        for z in [0, nz-1]:
            if abs(normal[2]) > 1e-6:
                x = plane_point[0] - (normal[1] * (y_coord - plane_point[1]) + normal[2] * (z - plane_point[2])) / normal[0]
                if 0 <= x < nx:
                    intersections.append([x, y_coord, z])
        
        if len(intersections) >= 2:
            return np.array(intersections[:2])
    
    elif axis == 'x':
        if slice_idx is None:
            slice_idx = volume_shape[0] // 2
        x_coord = slice_idx
        
        intersections = []
        nx, ny, nz = volume_shape
        
        for y in [0, ny-1]:
            if abs(normal[1]) > 1e-6:
                z = plane_point[2] - (normal[0] * (x_coord - plane_point[0]) + normal[1] * (y - plane_point[1])) / normal[2]
                if 0 <= z < nz:
                    intersections.append([x_coord, y, z])
        
        for z in [0, nz-1]:
            if abs(normal[2]) > 1e-6:
                y = plane_point[1] - (normal[0] * (x_coord - plane_point[0]) + normal[2] * (z - plane_point[2])) / normal[1]
                if 0 <= y < ny:
                    intersections.append([x_coord, y, z])
        
        if len(intersections) >= 2:
            return np.array(intersections[:2])
    
    return None


def visualize_slice_position(volume_data, R, T, output_prefix="slice_position_vis", size=256):
    """Create visualization showing slice position in volume."""
    
    volume_shape = volume_data.shape
    print(f"Volume shape: {volume_shape}")
    
    # Normalize volume for display
    def normalize_for_display(img):
        """Normalize image to [0, 1] for display."""
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img
    
    # Get middle slices along each axis
    mid_x = volume_shape[0] // 2
    mid_y = volume_shape[1] // 2
    mid_z = volume_shape[2] // 2
    
    # Get slice plane intersections
    # If plane is outside volume, try to find where it would intersect if extended
    intersection_z = get_slice_plane_intersection(volume_shape, R, T, size, axis='z', slice_idx=mid_z)
    intersection_y = get_slice_plane_intersection(volume_shape, R, T, size, axis='y', slice_idx=mid_y)
    intersection_x = get_slice_plane_intersection(volume_shape, R, T, size, axis='x', slice_idx=mid_x)
    
    # If no intersections found, try to get plane corners and project them
    plane_corners = None
    if NH_RS2V_AVAILABLE:
        plane_corners = get_slice_plane_points(R, T, size, volume_shape)
        if plane_corners is not None:
            # Check if any corners are within volume bounds
            corners_in_bounds = np.all((plane_corners >= 0) & (plane_corners < np.array(volume_shape)), axis=1)
            if not np.any(corners_in_bounds):
                print("Warning: Slice plane is completely outside volume bounds.")
                print("This indicates the registration likely failed.")
                # Still try to show the plane direction by using the plane normal
    
    # Create figure with 3 views
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial view (xy plane, z = mid_z)
    ax = axes[0]
    axial_slice = volume_data[:, :, mid_z]
    ax.imshow(normalize_for_display(axial_slice), cmap='gray', origin='lower')
    if intersection_z is not None:
        ax.plot(intersection_z[:, 0], intersection_z[:, 1], 'r-', linewidth=3, label='Slice plane')
        ax.plot(intersection_z[:, 0], intersection_z[:, 1], 'ro', markersize=8)
    ax.set_title(f'Axial View (z={mid_z})\nRed line: Registered slice plane', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Coronal view (xz plane, y = mid_y)
    ax = axes[1]
    coronal_slice = volume_data[:, mid_y, :]
    ax.imshow(normalize_for_display(coronal_slice), cmap='gray', origin='lower', aspect='auto')
    if intersection_y is not None:
        ax.plot(intersection_y[:, 0], intersection_y[:, 2], 'r-', linewidth=3, label='Slice plane')
        ax.plot(intersection_y[:, 0], intersection_y[:, 2], 'ro', markersize=8)
    ax.set_title(f'Coronal View (y={mid_y})\nRed line: Registered slice plane', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sagittal view (yz plane, x = mid_x)
    ax = axes[2]
    sagittal_slice = volume_data[mid_x, :, :]
    ax.imshow(normalize_for_display(sagittal_slice), cmap='gray', origin='lower', aspect='auto')
    if intersection_x is not None:
        ax.plot(intersection_x[:, 1], intersection_x[:, 2], 'r-', linewidth=3, label='Slice plane')
        ax.plot(intersection_x[:, 1], intersection_x[:, 2], 'ro', markersize=8)
    ax.set_title(f'Sagittal View (x={mid_x})\nRed line: Registered slice plane', fontsize=12)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_prefix}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_file}")
    
    # Print slice plane information
    if NH_RS2V_AVAILABLE:
        plane_corners = get_slice_plane_points(R, T, size, volume_shape)
        if plane_corners is not None:
            plane_center = np.mean(plane_corners, axis=0)
            print(f"\nSlice Plane Information:")
            print(f"  Center position (NH-RS2V coordinates): ({plane_center[0]:.1f}, {plane_center[1]:.1f}, {plane_center[2]:.1f})")
            print(f"  Volume shape (voxels): {volume_shape}")
            
            # Check if coordinates are within volume bounds
            in_bounds = (0 <= plane_center[0] < volume_shape[0] and 
                        0 <= plane_center[1] < volume_shape[1] and 
                        0 <= plane_center[2] < volume_shape[2])
            
            if in_bounds:
                print(f"  ✓ Slice plane center is within volume bounds")
                print(f"  Center as % of volume: ({100*plane_center[0]/volume_shape[0]:.1f}%, "
                      f"{100*plane_center[1]/volume_shape[1]:.1f}%, {100*plane_center[2]/volume_shape[2]:.1f}%)")
            else:
                print(f"  ✗ Slice plane center is OUTSIDE volume bounds!")
                print(f"  This suggests the registration may have failed or the coordinate")
                print(f"  system transformation needs adjustment.")
                
                # Try to find the closest point within bounds
                clamped_center = np.clip(plane_center, [0, 0, 0], 
                                       [volume_shape[0]-1, volume_shape[1]-1, volume_shape[2]-1])
                print(f"  Closest valid point: ({clamped_center[0]:.1f}, {clamped_center[1]:.1f}, {clamped_center[2]:.1f})")
    
    return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize registered slice position within volume"
    )
    parser.add_argument(
        "--result",
        type=str,
        default="registration_result_quick.json",
        help="Registration result JSON file"
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="volume_3d.npy",
        help="Path to volume numpy array"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data_metadata.json",
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="slice_position_vis",
        help="Output prefix for visualization files"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Size parameter used for registration (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading registration result...")
    R, T = load_registration_result(args.result)
    
    print("Loading volume data...")
    volume_data, metadata = load_data(args.volume, args.metadata)
    
    print(f"Volume shape: {volume_data.shape}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_slice_position(volume_data, R, T, output_prefix=args.output, size=args.size)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

