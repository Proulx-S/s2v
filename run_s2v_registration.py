#!/usr/bin/env python3
# Note: This script should be run with the conda environment's Python
# Use: /home/sebp/miniconda3/envs/nh-rs2v/bin/python3 run_s2v_registration.py
# Or: conda activate nh-rs2v && python3 run_s2v_registration.py
"""
Script to run NH-RS2V baseline methods on preprocessed MRI data.

This script loads the preprocessed slice and volume data and applies
either optimization-based or LoFTR-S2V registration methods.
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import zoom

# Add NH-RS2V paths to Python path
NH_RS2V_BASELINES_PATH = Path("/scratch/users/Proulx-S/tools/NH-RS2V-baselines")
NH_RS2V_DATASET_PATH = Path("/scratch/users/Proulx-S/tools/NH-RS2V-dataset")

for path in [NH_RS2V_DATASET_PATH, NH_RS2V_BASELINES_PATH]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Try to import NH-RS2V optimization modules
NH_RS2V_AVAILABLE = False
try:
    # First try normal import (if packages are properly installed)
    from nh_rs2v_baseline.optim.solve import solve_for_RT
    from nh_rs2v_baseline.optim.refine import refine_RT
    from nh_rs2v_baseline.optim.utils import normalize_slice_volume
    from joblib import Parallel
    NH_RS2V_AVAILABLE = True
    print(f"✓ NH-RS2V optimization modules loaded")
except ImportError:
    # If that fails, try importing directly from files
    try:
        import importlib.util
        
        # Set up module structure
        sys.modules["nh_rs2v_baseline"] = type(sys)("nh_rs2v_baseline")
        sys.modules["nh_rs2v_baseline.optim"] = type(sys)("nh_rs2v_baseline.optim")
        sys.modules["nh_rs2v_dataset"] = type(sys)("nh_rs2v_dataset")
        sys.modules["nh_rs2v_dataset.utils"] = type(sys)("nh_rs2v_dataset.utils")
        
        # Import dataset utils first (needed by optim modules)
        # Import in dependency order
        dataset_utils_path = NH_RS2V_DATASET_PATH / "nh_rs2v_dataset" / "utils"
        if dataset_utils_path.exists():
            # Import order: invert_4x4 -> g_s_t -> sampling -> transforms
            # Skip sampling_cupy (only needed for cupy backend)
            utils_files = ["invert_4x4.py", "g_s_t.py", "sampling.py", "transforms.py"]
            for utils_file in utils_files:
                utils_file_path = dataset_utils_path / utils_file
                if utils_file_path.exists():
                    module_name = utils_file.replace(".py", "")
                    spec = importlib.util.spec_from_file_location(module_name, utils_file_path)
                    utils_module = importlib.util.module_from_spec(spec)
                    sys.modules[f"nh_rs2v_dataset.utils.{module_name}"] = utils_module
                    spec.loader.exec_module(utils_module)
            
            # Create dummy sampling_cupy module (not needed for numpy backend)
            class DummySamplingCupy:
                pass
            sys.modules["nh_rs2v_dataset.utils.sampling_cupy"] = DummySamplingCupy()
        
        # Now import optim modules (in dependency order)
        optim_path = NH_RS2V_BASELINES_PATH / "nh_rs2v_baseline" / "optim"
        
        # Import utils first (needed by solve_routines and others)
        utils_path = optim_path / "utils.py"
        if utils_path.exists():
            spec = importlib.util.spec_from_file_location("utils", utils_path)
            utils_module = importlib.util.module_from_spec(spec)
            sys.modules["nh_rs2v_baseline.optim.utils"] = utils_module
            spec.loader.exec_module(utils_module)
            normalize_slice_volume = utils_module.normalize_slice_volume
        
        # Import utils_cupy (optional, only needed for cupy backend)
        # Skip if cupy not available - we're using numpy backend anyway
        try:
            import cupy
            utils_cupy_path = optim_path / "utils_cupy.py"
            if utils_cupy_path.exists():
                spec = importlib.util.spec_from_file_location("utils_cupy", utils_cupy_path)
                utils_cupy_module = importlib.util.module_from_spec(spec)
                sys.modules["nh_rs2v_baseline.optim.utils_cupy"] = utils_cupy_module
                spec.loader.exec_module(utils_cupy_module)
        except ImportError:
            # Create a dummy module for utils_cupy when cupy is not available
            class DummyUtilsCupy:
                pass
            sys.modules["nh_rs2v_baseline.optim.utils_cupy"] = DummyUtilsCupy()
        
        # Import solve_routines (needed by solve)
        solve_routines_path = optim_path / "solve_routines.py"
        if solve_routines_path.exists():
            spec = importlib.util.spec_from_file_location("solve_routines", solve_routines_path)
            solve_routines_module = importlib.util.module_from_spec(spec)
            sys.modules["nh_rs2v_baseline.optim.solve_routines"] = solve_routines_module
            spec.loader.exec_module(solve_routines_module)
        
        # Import refine
        refine_path = optim_path / "refine.py"
        if refine_path.exists():
            spec = importlib.util.spec_from_file_location("refine", refine_path)
            refine_module = importlib.util.module_from_spec(spec)
            sys.modules["nh_rs2v_baseline.optim.refine"] = refine_module
            spec.loader.exec_module(refine_module)
            refine_RT = refine_module.refine_RT
        
        # Import solve
        solve_path = optim_path / "solve.py"
        if solve_path.exists():
            spec = importlib.util.spec_from_file_location("solve", solve_path)
            solve_module = importlib.util.module_from_spec(spec)
            sys.modules["nh_rs2v_baseline.optim.solve"] = solve_module
            spec.loader.exec_module(solve_module)
            solve_for_RT = solve_module.solve_for_RT
        
        from joblib import Parallel
        NH_RS2V_AVAILABLE = True
        print(f"✓ NH-RS2V optimization modules loaded (direct import)")
    except Exception as e:
        print(f"Warning: NH-RS2V modules not available: {e}")
        print(f"Looking for packages at:")
        print(f"  Baselines: {NH_RS2V_BASELINES_PATH}")
        print(f"  Dataset: {NH_RS2V_DATASET_PATH}")
        if not NH_RS2V_BASELINES_PATH.exists():
            print(f"  ✗ Baselines path does not exist!")
        if not NH_RS2V_DATASET_PATH.exists():
            print(f"  ✗ Dataset path does not exist!")
        if NH_RS2V_BASELINES_PATH.exists() and NH_RS2V_DATASET_PATH.exists():
            print(f"  ✓ Paths exist, but module import failed.")
            print(f"  This might be due to missing dependencies.")
            print(f"")
            print(f"  To fix this, run:")
            print(f"    ./setup_environment.sh")


def load_preprocessed_data(slice_path="slice_2d.npy", volume_path="volume_3d.npy"):
    """Load preprocessed slice and volume data."""
    print(f"Loading slice from {slice_path}...")
    slice_data = np.load(slice_path, allow_pickle=False)
    
    print(f"Loading volume from {volume_path}...")
    volume_data = np.load(volume_path, allow_pickle=False)
    
    print(f"Slice shape: {slice_data.shape}")
    print(f"Volume shape: {volume_data.shape}")
    
    # Ensure slice is 2D
    if slice_data.ndim != 2:
        raise ValueError(f"Slice must be 2D, got shape {slice_data.shape}")
    
    # Ensure volume is 3D
    if volume_data.ndim != 3:
        raise ValueError(f"Volume must be 3D, got shape {volume_data.shape}")
    
    return slice_data, volume_data


def run_optimization_method(slice_data, volume_data, method="scipy_SLSQP", 
                           loss="MAE", population=2048, backend="numpy", size=None,
                           translation_constraint_factor=0.3, rotation_constraint_factor=1.0):
    """
    Run optimization-based registration method.
    
    Parameters:
    -----------
    slice_data : np.ndarray
        2D slice image
    volume_data : np.ndarray
        3D volume image
    method : str
        Optimization method: 'scipy_SLSQP', 'scipy_LBFGSB', 'nlopt_SBPLX', 
        'nlopt_COBYLA', or 'random'
    loss : str
        Loss function: 'MAE', 'MSE', or 'ZNCC'
    population : int
        Number of random initializations for local optimizers
    backend : str
        Backend: 'numpy' or 'cupy'
    """
    if not NH_RS2V_AVAILABLE:
        raise ImportError("NH-RS2V modules not available")
    
    print(f"\nRunning optimization method: {method}")
    print(f"Loss function: {loss}")
    print(f"Population size: {population}")
    print(f"Backend: {backend}")
    
    # Use actual slice dimensions instead of downsampling
    # The size parameter should match the slice dimensions for proper coordinate system
    if size is None or size <= 0:
        # Auto-detect: use the maximum dimension of the slice (assuming square or near-square)
        size = max(slice_data.shape[0], slice_data.shape[1])
        print(f"Auto-detected slice sampling size: {size} (from slice dimensions {slice_data.shape})")
        print(f"This avoids downsampling and preserves full resolution.")
    else:
        print(f"Slice sampling size: {size}")
    
    # Clear NH-RS2V coordinate grid cache - CRITICAL: cache is size-specific
    # The g_s_t.py code checks if cache exists and uses it WITHOUT verifying size
    # This causes "mmap length is greater than file size" errors if cache was created for different size
    # We MUST clear it unconditionally before optimization - it will be recreated with correct size
    import os
    cache_slice = "/tmp/g_s_slice"
    cache_volume = "/tmp/g_s_volume"
    
    # Always clear cache unconditionally - it's cheap to recreate and prevents errors
    # This is especially important when switching between different size parameters
    cache_cleared = False
    try:
        if os.path.exists(cache_slice):
            os.remove(cache_slice)
            cache_cleared = True
    except Exception as e:
        print(f"Warning: Could not remove {cache_slice}: {e}")
    
    try:
        if os.path.exists(cache_volume):
            os.remove(cache_volume)
            cache_cleared = True
    except Exception as e:
        print(f"Warning: Could not remove {cache_volume}: {e}")
    
    # Verify cache is actually gone (safety check)
    if os.path.exists(cache_slice) or os.path.exists(cache_volume):
        print(f"WARNING: Cache files still exist after removal attempt!")
        print(f"  This may cause errors. Trying to force remove...")
        try:
            if os.path.exists(cache_slice):
                os.unlink(cache_slice)  # Force remove
            if os.path.exists(cache_volume):
                os.unlink(cache_volume)  # Force remove
        except Exception as e:
            print(f"  Could not force remove: {e}")
    
    if cache_cleared:
        print(f"✓ Cleared coordinate grid cache (will be recreated for size={size})")
    
    # Pre-create cache file to avoid race conditions in parallel execution
    # Multiple workers trying to create the cache simultaneously causes Bus errors
    # We'll create it here before optimization starts
    if not os.path.exists(cache_slice):
        print(f"Pre-creating coordinate grid cache for size={size} to avoid race conditions...")
        try:
            # Create the cache file the same way g_s_t.py does
            lin_g_s = np.linspace(0.5, size - 0.5, size, dtype=np.float32)
            g_s = np.meshgrid(lin_g_s, lin_g_s)
            g_s = np.stack((g_s[0], g_s[1], np.zeros_like(g_s[0]), np.ones_like(g_s[0])))
            g_s = g_s.reshape(4, -1)
            g_s_mmap = np.memmap(
                cache_slice,
                dtype="float32",
                mode="w+",
                shape=(4, int(size * size)),
            )
            np.copyto(g_s_mmap, g_s)
            del g_s_mmap  # Close the memmap
            print(f"✓ Pre-created coordinate grid cache for size={size}")
        except Exception as e:
            print(f"Warning: Could not pre-create cache file: {e}")
            print(f"  Workers will create it individually (may cause race conditions)")
    
    # Check if slice needs resampling
    # NH-RS2V expects square slices, so we need to ensure the slice is square
    if slice_data.shape[0] != slice_data.shape[1]:
        # Slice is not square - need to make it square
        if slice_data.shape[0] != size or slice_data.shape[1] != size:
            print(f"Resampling slice from {slice_data.shape} to ({size}, {size}) to make it square...")
            zoom_factors = (size / slice_data.shape[0], size / slice_data.shape[1])
            slice_data = zoom(slice_data, zoom_factors, order=1, mode='nearest')
            print(f"  Resampled slice shape: {slice_data.shape}")
    elif slice_data.shape[0] != size or slice_data.shape[1] != size:
        # Slice is square but size doesn't match - only resample if size was explicitly set
        if size != max(slice_data.shape[0], slice_data.shape[1]):
            print(f"Note: Slice is {slice_data.shape[0]}x{slice_data.shape[1]}, but size={size}")
            print(f"      Resampling to match requested size...")
            zoom_factors = (size / slice_data.shape[0], size / slice_data.shape[1])
            slice_data = zoom(slice_data, zoom_factors, order=1, mode='nearest')
            print(f"  Resampled slice shape: {slice_data.shape}")
        else:
            # Size matches, no resampling needed
            print(f"Slice dimensions ({slice_data.shape[0]}x{slice_data.shape[1]}) match size parameter - no resampling needed.")
    
    # Note: solve_for_RT will normalize the data internally, so pass raw data
    # Define bounds for optimization
    # Bounds: [qx, qy, qz, qw, tx, ty, tz]
    # Quaternion: Identity rotation is (0, 0, 0, 1) for (qx, qy, qz, qw)
    # Translation bounds: can be constrained to search near middle slice
    volume_size = np.array(volume_data.shape)
    voxel_size = 0.4  # mm (from your data)
    
    # Rotation bounds: Quaternion representation
    # Identity rotation is (qx=0, qy=0, qz=0, qw=1)
    # To constrain near identity, use small bounds around 0 for qx, qy, qz
    # and bounds around 1 for qw (or keep qw in [-1, 1] and let normalization handle it)
    # rotation_constraint_factor: 0.1 = very tight (small rotations), 1.0 = full rotation space
    if rotation_constraint_factor < 1.0:
        # Constrain quaternion components to be small (near identity)
        # For small rotations, qx, qy, qz are small, qw is close to 1
        # We'll constrain qx, qy, qz to [-rotation_constraint_factor, rotation_constraint_factor]
        # and qw to [1-rotation_constraint_factor, 1] or [-1, 1] if constraint is loose
        max_quat_component = rotation_constraint_factor
        quat_w_min = max(-1.0, 1.0 - rotation_constraint_factor)
        quat_w_max = 1.0
    else:
        # Full rotation space
        max_quat_component = 1.0
        quat_w_min = -1.0
        quat_w_max = 1.0
    
    # Translation bounds: T is in NH-RS2V's internal coordinate system
    # To constrain near middle slice, use smaller bounds centered around 0
    # translation_constraint_factor: 0.1 = very tight (10% of volume), 1.0 = full volume search
    max_translation_constrained = np.max(volume_size) * voxel_size * translation_constraint_factor
    
    bounds = np.array([
        [-max_quat_component, max_quat_component],  # qx (constrained near 0 for identity)
        [-max_quat_component, max_quat_component],  # qy (constrained near 0 for identity)
        [-max_quat_component, max_quat_component],  # qz (constrained near 0 for identity)
        [quat_w_min, quat_w_max],  # qw (constrained near 1 for identity)
        [-max_translation_constrained, max_translation_constrained],  # tx
        [-max_translation_constrained, max_translation_constrained],  # ty
        [-max_translation_constrained, max_translation_constrained],  # tz
    ], dtype=np.float32)
    
    if rotation_constraint_factor < 1.0:
        print(f"Rotation bounds: qx,qy,qz ∈ ±{max_quat_component:.3f}, qw ∈ [{quat_w_min:.3f}, {quat_w_max:.3f}]")
        print(f"  This constrains rotation to be near identity (small angles).")
    else:
        print(f"Rotation bounds: Full rotation space (qx,qy,qz,qw ∈ [-1, 1])")
    
    print(f"Translation bounds: ±{max_translation_constrained:.2f} mm (constrained to {translation_constraint_factor*100:.0f}% of volume size)")
    print(f"  This constrains the search to be near the middle slice of the volume.")
    
    # Ground truth scaling coefficients (2-element array for lateral scaling)
    # Range: (0.5, 1.5). Use [1.0, 1.0] for no scaling (identity scaling)
    # This is used internally by the optimization, not for evaluation
    S_gt = np.array([1.0, 1.0], dtype=np.float32)
    
    # Run optimization
    print("Starting optimization...")
    print("Note: This may take several minutes depending on population size...")
    result = solve_for_RT(
        opt_name=method,
        loss_name=loss,
        bounds=bounds,
        in_slice=slice_data,  # Pass raw data, normalization happens inside
        in_volume=volume_data,  # Pass raw data, normalization happens inside
        S_gt=S_gt,
        size=size,
        population_size=population,
        parallel=Parallel(n_jobs=-1, verbose=1),
        backend=backend,
    )
    
    return result


def run_loftr_s2v(slice_data, volume_data, robust_estimator="GCRANSAC", 
                  use_local_optimization=True, use_cuda=True):
    """
    Run LoFTR-S2V deep learning method.
    
    Parameters:
    -----------
    slice_data : np.ndarray
        2D slice image
    volume_data : np.ndarray
        3D volume image
    robust_estimator : str
        Robust estimation method: 'TLS', 'RANSAC', 'GCRANSAC', 'MAGSAC'
    use_local_optimization : bool
        Whether to refine with local optimization
    use_cuda : bool
        Whether to use GPU
    """
    if not NH_RS2V_AVAILABLE:
        raise ImportError("NH-RS2V modules not available")
    
    print(f"\nRunning LoFTR-S2V method")
    print(f"Robust estimator: {robust_estimator}")
    print(f"Local optimization: {use_local_optimization}")
    print(f"Use CUDA: {use_cuda}")
    
    # This is more complex and requires the full data module setup
    # For now, we'll provide a placeholder
    print("LoFTR-S2V requires more setup. Please use the example scripts from")
    print("NH-RS2V-baselines repository as a template.")
    print("See: examples/nh_rs2v_loftr-s2v.py")
    
    raise NotImplementedError("LoFTR-S2V integration requires full dataset setup")


def main():
    parser = argparse.ArgumentParser(
        description="Run NH-RS2V registration on preprocessed MRI data"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="scipy_SLSQP",
        choices=["scipy_SLSQP", "scipy_LBFGSB", "nlopt_SBPLX", "nlopt_COBYLA", "random", "loftr"],
        help="Registration method to use"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="MAE",
        choices=["MAE", "MSE", "ZNCC"],
        help="Loss function for optimization methods"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=2048,
        help="Population size for optimization methods"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "cupy"],
        help="Backend for computation"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="slice_2d.npy",
        help="Path to preprocessed slice numpy array"
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="volume_3d.npy",
        help="Path to preprocessed volume numpy array"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="registration_result.json",
        help="Output file for registration result"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Size for slice sampling (default: auto-detect from slice dimensions, avoids downsampling)"
    )
    parser.add_argument(
        "--translation-constraint",
        type=float,
        default=0.3,
        help="Translation constraint factor (0.0-1.0): smaller values constrain search near middle slice. Default: 0.3 (30%% of volume size)"
    )
    parser.add_argument(
        "--rotation-constraint",
        type=float,
        default=1.0,
        help="Rotation constraint factor (0.0-1.0): smaller values constrain rotation near identity. Default: 1.0 (full rotation space). Use 0.1-0.3 for small rotations."
    )
    
    args = parser.parse_args()
    
    # CRITICAL: Clear cache at the very start to avoid any size mismatches
    # This must happen before loading data to ensure clean state
    import os
    cache_slice = "/tmp/g_s_slice"
    cache_volume = "/tmp/g_s_volume"
    try:
        if os.path.exists(cache_slice):
            os.remove(cache_slice)
        if os.path.exists(cache_volume):
            os.remove(cache_volume)
    except Exception:
        pass  # Ignore errors, will be handled in run_optimization_method
    
    # Load data
    slice_data, volume_data = load_preprocessed_data(args.slice, args.volume)
    
    # Auto-detect size if not specified
    if args.size is None or args.size <= 0:
        args.size = max(slice_data.shape[0], slice_data.shape[1])
        print(f"\nAuto-detected slice size: {args.size} (from slice dimensions {slice_data.shape})")
        print(f"This avoids downsampling and preserves full resolution.\n")
    
    # Run registration
    if args.method == "loftr":
        result = run_loftr_s2v(slice_data, volume_data)
    else:
        result = run_optimization_method(
            slice_data, volume_data,
            method=args.method,
            loss=args.loss,
            population=args.population,
            backend=args.backend,
            size=args.size,
            translation_constraint_factor=args.translation_constraint,
            rotation_constraint_factor=args.rotation_constraint
        )
    
    # Save result
    print(f"\nSaving result to {args.output}...")
    
    # Handle result format: solve_for_RT returns (R, T) tuple
    # R is 3x3 rotation matrix, T is 3x1 translation vector
    if isinstance(result, tuple) and len(result) == 2:
        R, T = result
        result_serializable = {
            "rotation_matrix": R.tolist() if isinstance(R, np.ndarray) else R,
            "translation_vector": T.tolist() if isinstance(T, np.ndarray) else T,
            "method": args.method,
            "loss": args.loss,
            "population_size": args.population,
        }
        
        # Also save as separate files for easier use
        np.save("registration_R.npy", R)
        np.save("registration_T.npy", T)
        print(f"  - Rotation matrix saved to registration_R.npy")
        print(f"  - Translation vector saved to registration_T.npy")
    else:
        # Fallback for unexpected format
        result_serializable = {
            "result": str(result),
            "raw_result": result.tolist() if isinstance(result, np.ndarray) else result,
        }
    
    with open(args.output, 'w') as f:
        json.dump(result_serializable, f, indent=2)
    
    print(f"✓ Registration complete! Result saved to {args.output}")
    print(f"\nRegistration parameters:")
    if isinstance(result, tuple) and len(result) == 2:
        R, T = result
        print(f"  Rotation matrix R (3x3):")
        print(f"    {R}")
        print(f"  Translation vector T (3x1):")
        print(f"    {T}")


if __name__ == "__main__":
    main()

