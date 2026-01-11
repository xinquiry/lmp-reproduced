"""
Potential file management utilities.

Provides functions to locate and resolve paths to interatomic potential files
used in LAMMPS simulations.
"""

from pathlib import Path
import os
from typing import Optional


def get_potential_dir() -> Path:
    """
    Get the directory containing potential files.

    Checks in order:
    1. LMP_POTENTIAL_DIR environment variable
    2. Default: data/pot relative to package root

    Returns:
        Path to the potential files directory
    """
    if env_dir := os.environ.get("LMP_POTENTIAL_DIR"):
        return Path(env_dir)

    # Default: data/pot relative to package root
    # Path structure: src/lmp_reproduced/core/potentials.py
    # Need to go up 4 levels to reach project root
    package_root = Path(__file__).parent.parent.parent.parent
    return package_root / "data" / "pot"


def resolve_potential_path(
    filename: str,
    pot_dir: Optional[Path] = None,
    check_exists: bool = True
) -> Path:
    """
    Resolve a potential filename to its full path.

    Args:
        filename: Name of the potential file (e.g., "AlCu.eam.alloy")
        pot_dir: Optional custom potential directory. If None, uses get_potential_dir()
        check_exists: If True, raise FileNotFoundError if file doesn't exist

    Returns:
        Full path to the potential file

    Raises:
        FileNotFoundError: If check_exists=True and the file doesn't exist
    """
    pot_dir = pot_dir or get_potential_dir()
    path = pot_dir / filename

    if check_exists and not path.exists():
        raise FileNotFoundError(f"Potential file not found: {path}")

    return path


def list_available_potentials(pot_dir: Optional[Path] = None) -> list[Path]:
    """
    List all available potential files in the potential directory.

    Args:
        pot_dir: Optional custom potential directory

    Returns:
        List of paths to all potential files
    """
    pot_dir = pot_dir or get_potential_dir()

    if not pot_dir.exists():
        return []

    # Common potential file extensions
    extensions = {
        ".eam", ".alloy", ".fs", ".meam",
        ".tersoff", ".airebo", ".sw", ".set"
    }

    potentials = []
    for path in pot_dir.iterdir():
        if path.is_file():
            # Check if any extension matches
            if any(ext in path.name for ext in extensions):
                potentials.append(path)

    return sorted(potentials)


def get_potential_info() -> dict[str, dict]:
    """
    Get information about available potential files.

    Returns:
        Dictionary mapping filename to info dict with keys:
        - path: Full path to file
        - type: Inferred potential type (eam/alloy, meam, tersoff, etc.)
        - size: File size in bytes
    """
    potentials = list_available_potentials()

    info = {}
    for path in potentials:
        pot_type = _infer_potential_type(path.name)
        info[path.name] = {
            "path": path,
            "type": pot_type,
            "size": path.stat().st_size,
        }

    return info


def _infer_potential_type(filename: str) -> str:
    """Infer potential type from filename."""
    name_lower = filename.lower()

    if ".eam.alloy" in name_lower:
        return "eam/alloy"
    elif ".eam.fs" in name_lower:
        return "eam/fs"
    elif ".eam" in name_lower:
        return "eam"
    elif ".meam" in name_lower:
        return "meam"
    elif ".tersoff" in name_lower:
        return "tersoff"
    elif ".airebo" in name_lower:
        return "airebo"
    elif ".sw" in name_lower:
        return "sw"
    elif ".set" in name_lower:
        return "adp"  # Angular-dependent potential
    else:
        return "unknown"
