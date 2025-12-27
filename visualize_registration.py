#!/usr/bin/env python3
"""
Visualize slice-to-volume registration results.

This script:
1. Loads the registration result (R, T)
2. Samples a slice from the volume using the transformation
3. Compares it with the original slice
4. Creates visualization plots
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
    else:
        raise ImportError("Could not load S2VTransform or sample_slice")
except Exception as e1:
    # Try importing directly from files (similar to run_s2v_registration.py)
    try:
        import importlib.util
        
        # Import dependencies first
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            print("Warning: scipy.spatial.transform.Rotation not available")
            Rotation = None
        
        dataset_utils_path = NH_RS2V_DATASET_PATH / "nh_rs2v_dataset" / "utils"
        if dataset_utils_path.exists():
            # Import sampling module
            spec = importlib.util.spec_from_file_location(
                "sampling", dataset_utils_path / "sampling.py"
            )
            sampling_module = importlib.util.module_from_spec(spec)
            # Add necessary dependencies to the module
            sampling_module.np = np
            if hasattr(sampling_module, 'g_s_t_slice'):
                # Import g_s_t if needed
                try:
                    from nh_rs2v_dataset.utils.g_s_t import g_s_t_slice
                    sampling_module.g_s_t_slice = g_s_t_slice
                except:
                    pass
            spec.loader.exec_module(sampling_module)
            sample_slice = sampling_module.sample_slice
            
            # Import transforms module
            spec = importlib.util.spec_from_file_location(
                "transforms", dataset_utils_path / "transforms.py"
            )
            transforms_module = importlib.util.module_from_spec(spec)
            # Add necessary dependencies
            transforms_module.np = np
            if Rotation is not None:
                transforms_module.Rotation = Rotation
            spec.loader.exec_module(transforms_module)
            S2VTransform = transforms_module.S2VTransform
            NH_RS2V_AVAILABLE = True
    except Exception as e2:
        print(f"Warning: NH-RS2V modules not available.")
        print(f"  Direct import error: {e1}")
        print(f"  File import error: {e2}")
        print("Some features (like sampling slices from volume) may be limited.")
        NH_RS2V_AVAILABLE = False


def load_registration_result(result_file="registration_result_quick.json"):
    """Load registration result from JSON file."""
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    R = np.array(result['rotation_matrix'])
    T = np.array(result['translation_vector'])
    
    return R, T


def load_data(slice_path="slice_2d.npy", volume_path="volume_3d.npy", 
              metadata_path="data_metadata.json"):
    """Load slice, volume, and metadata."""
    slice_data = np.load(slice_path)
    volume_data = np.load(volume_path)
    
    metadata = None
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return slice_data, volume_data, metadata


def create_transformation_matrix(R, T):
    """Create 4x4 transformation matrix from R and T."""
    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = R
    A[:3, 3] = T
    return A


def sample_slice_from_volume(volume, R, T, size=256):
    """Sample a slice from the volume using the registration transformation."""
    if not NH_RS2V_AVAILABLE:
        raise ImportError("NH-RS2V modules not available")
    
    # The R and T returned from solve_for_RT are from S2VTransform.get_R() and get_T()
    # We need to create an S2VTransform from R, T, and S_gt to get the full transformation
    S_gt = np.array([1.0, 1.0], dtype=np.float32)
    
    # Create S2VTransform from R, T, S using from_RTS
    # This will give us the full transformation matrix A
    s2v_transform = S2VTransform.from_RTS(size, R, T, S_gt)
    
    # Get the full transformation matrix A
    A = s2v_transform.get_A()
    
    # Sample slice using NH-RS2V's sample_slice function
    # Use inverted_A=True because we want to transform from slice space to volume space
    sampled_slice = sample_slice(volume, A, size, interpolation_order=1, inverted_A=True)
    
    return sampled_slice


def visualize_registration(slice_data, volume_data, R, T, 
                          output_prefix="registration_vis",
                          size=256):
    """Create visualization of registration result."""
    
    # Resample original slice to match size if needed
    if slice_data.shape[0] != size or slice_data.shape[1] != size:
        zoom_factors = (size / slice_data.shape[0], size / slice_data.shape[1])
        slice_resampled = zoom(slice_data, zoom_factors, order=1, mode='nearest')
    else:
        slice_resampled = slice_data.copy()
    
    # Normalize for visualization
    def normalize_for_display(img):
        """Normalize image to [0, 1] for display."""
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img
    
    slice_norm = normalize_for_display(slice_resampled)
    
    # Try to sample slice from volume
    try:
        if NH_RS2V_AVAILABLE:
            sampled_slice = sample_slice_from_volume(volume_data, R, T, size=size)
            sampled_norm = normalize_for_display(sampled_slice)
            has_sampled = True
        else:
            has_sampled = False
            sampled_norm = None
    except Exception as e:
        print(f"Warning: Could not sample slice from volume: {e}")
        has_sampled = False
        sampled_norm = None
    
    # Create figure
    fig, axes = plt.subplots(1, 3 if has_sampled else 2, figsize=(15, 5))
    
    # Original slice (movable)
    ax = axes[0]
    im = ax.imshow(slice_norm, cmap='gray', interpolation='nearest')
    ax.set_title('Original Slice\n(Movable)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    if has_sampled:
        # Sampled slice from volume (fixed, at registered location)
        ax = axes[1]
        im = ax.imshow(sampled_norm, cmap='gray', interpolation='nearest')
        ax.set_title('Slice from Volume\n(Fixed, at registered location)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Difference
        ax = axes[2]
        diff = np.abs(slice_norm - sampled_norm)
        im = ax.imshow(diff, cmap='hot', interpolation='nearest')
        ax.set_title('Absolute Difference', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Print statistics
        print(f"\nRegistration Quality Metrics:")
        print(f"  Mean absolute difference: {np.mean(diff):.4f}")
        print(f"  Max absolute difference: {np.max(diff):.4f}")
        print(f"  Mean squared difference: {np.mean(diff**2):.4f}")
    else:
        # Show volume middle slice as reference
        ax = axes[1]
        mid_slice_idx = volume_data.shape[2] // 2
        volume_slice = volume_data[:, :, mid_slice_idx]
        if volume_slice.shape[0] != size or volume_slice.shape[1] != size:
            zoom_factors = (size / volume_slice.shape[0], size / volume_slice.shape[1])
            volume_slice = zoom(volume_slice, zoom_factors, order=1, mode='nearest')
        volume_norm = normalize_for_display(volume_slice)
        im = ax.imshow(volume_norm, cmap='gray', interpolation='nearest')
        ax.set_title(f'Volume Middle Slice (z={mid_slice_idx})\n(Reference)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_prefix}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_file}")
    
    return fig


def print_translation_info(T, metadata=None, size=256):
    """Print information about translation vector."""
    print(f"\nTranslation Vector T:")
    print(f"  T = [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")
    print(f"  Magnitude: {np.linalg.norm(T):.2f}")
    
    print(f"\nCoordinate System Information:")
    print(f"  The translation is in NH-RS2V's internal coordinate system.")
    print(f"  This coordinate system uses a grid from 0.5 to (size-0.5) = {size-0.5}")
    print(f"  where 'size' = {size} (the slice sampling size parameter).")
    print(f"  The values are NOT directly in voxel coordinates or physical mm.")
    
    if metadata:
        print(f"\nVolume Information:")
        print(f"  Volume shape: {metadata['volume_shape']}")
        print(f"  Volume voxel size: {metadata['volume_voxel_size']} mm")
        volume_size_mm = [s * v for s, v in zip(metadata['volume_shape'], metadata['volume_voxel_size'])]
        print(f"  Volume physical size: {volume_size_mm} mm")
        
        # Rough estimate: if we assume the coordinate system scales with volume
        # (this is approximate - the actual transformation is more complex)
        print(f"\nApproximate Interpretation (ROUGH ESTIMATE):")
        print(f"  The internal coordinate system spans roughly the volume dimensions.")
        print(f"  A rough scaling factor might be: volume_size / size")
        if len(metadata['volume_shape']) >= 3:
            scale_estimate = [s / size for s in metadata['volume_shape'][:3]]
            print(f"  Estimated scaling: {scale_estimate}")
            T_scaled = [T[i] * scale_estimate[i] for i in range(3)]
            print(f"  Roughly scaled T: [{T_scaled[0]:.2f}, {T_scaled[1]:.2f}, {T_scaled[2]:.2f}]")
            T_mm = [T_scaled[i] * metadata['volume_voxel_size'][i] for i in range(3)]
            print(f"  Very rough estimate in mm: [{T_mm[0]:.2f}, {T_mm[1]:.2f}, {T_mm[2]:.2f}] mm")
            print(f"  (WARNING: This is a very rough approximation!)")
    
    print(f"\nNote: The actual coordinate transformation involves:")
    print(f"      1. Internal coordinate system (0.5 to size-0.5)")
    print(f"      2. Canonical coordinate system transformations")
    print(f"      3. Volume coordinate system")
    print(f"      4. Physical space (via affine matrices)")
    print(f"      To get accurate physical coordinates, use the full transformation pipeline.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize slice-to-volume registration results"
    )
    parser.add_argument(
        "--result",
        type=str,
        default="registration_result_quick.json",
        help="Registration result JSON file"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="slice_2d.npy",
        help="Path to slice numpy array"
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
        default="registration_vis",
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
    
    print("Loading slice and volume data...")
    slice_data, volume_data, metadata = load_data(args.slice, args.volume, args.metadata)
    
    print(f"Slice shape: {slice_data.shape}")
    print(f"Volume shape: {volume_data.shape}")
    
    # Print translation information
    print_translation_info(T, metadata, args.size)
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_registration(slice_data, volume_data, R, T, 
                          output_prefix=args.output, size=args.size)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

