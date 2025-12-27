#!/usr/bin/env python3
"""
Fix for cupy import issue - makes cupy optional when not available.
This allows the optimization methods to work with numpy backend.
"""

import sys
from pathlib import Path

# Add NH-RS2V path
NH_RS2V_PATH = Path("/scratch/users/Proulx-S/tools/NH-RS2V-baselines")
if str(NH_RS2V_PATH) not in sys.path:
    sys.path.insert(0, str(NH_RS2V_PATH))

# Try to import cupy, if it fails, create a mock
try:
    import cupy as cp
    print("✓ CuPy is available (GPU acceleration possible)")
except ImportError:
    print("⚠ CuPy not available - creating mock for numpy backend")
    # Create a mock cupy module
    class MockCupy:
        """Mock cupy module for when cupy is not installed"""
        @staticmethod
        def get_default_memory_pool():
            return MockMemoryPool()
        
        @staticmethod
        def get_default_pinned_memory_pool():
            return MockMemoryPool()
        
        class MockMemoryPool:
            def used_bytes(self):
                return 0
            def total_bytes(self):
                return 0
            def n_free_blocks(self):
                return 0
            def free_all_blocks(self):
                pass
    
    # Inject mock into sys.modules
    import types
    cp = MockCupy()
    sys.modules['cupy'] = cp
    
    # Also need to patch the solve module
    import importlib.util
    solve_path = NH_RS2V_PATH / "nh_rs2v_baseline" / "optim" / "solve.py"
    if solve_path.exists():
        # Read the file and replace cupy import
        with open(solve_path, 'r') as f:
            content = f.read()
        
        # Replace the import with a try/except
        if "import cupy as cp" in content and "try:" not in content.split("import cupy")[0][-50:]:
            new_content = content.replace(
                "import cupy as cp",
                "try:\n    import cupy as cp\nexcept ImportError:\n    cp = None  # CuPy not available, use numpy backend"
            )
            # Write back (create backup first)
            backup_path = solve_path.with_suffix('.py.backup')
            if not backup_path.exists():
                with open(backup_path, 'w') as f:
                    f.write(content)
                print(f"  Created backup: {backup_path}")
            
            with open(solve_path, 'w') as f:
                f.write(new_content)
            print(f"  ✓ Patched {solve_path} to make cupy optional")
        else:
            print("  ℹ File already patched or structure changed")

if __name__ == "__main__":
    print("Fixing cupy import issue...")
    print("This allows optimization methods to work with numpy backend.")
    print("")

