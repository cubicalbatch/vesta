"""Verify docker-entrypoint.sh behavior and ownership repair.

Ensures that the models directory and its subdirectories/symlinks are created
before `chown -R` runs across $DATA_DIR, so that the unprivileged runtime user
owns $DATA_DIR and the created models hierarchy.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ENTRYPOINT_PATH = Path(__file__).resolve().parent.parent / "docker-entrypoint.sh"


def test_entrypoint_chowns_models_after_creation(tmp_path: Path) -> None:
    """docker-entrypoint.sh must create baked model symlinks and run chown after."""
    data_dir = tmp_path / "data"
    baked_dir = tmp_path / "baked"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Create baked models structure: org/repo/model.onnx
    repo_baked = baked_dir / "minishlab" / "potion-retrieval-32M"
    repo_baked.mkdir(parents=True)
    (repo_baked / "model.onnx").write_text("dummy-onnx")

    log_file = tmp_path / "calls.log"

    # Mock chown: records whether models dir exists at the time chown runs
    mock_chown = bin_dir / "chown"
    mock_chown.write_text(f"""#!/bin/sh
echo "CHOWN: $*" >> "{log_file}"
if [ -d "{data_dir}/models" ]; then
    echo "MODELS_DIR_EXISTS_DURING_CHOWN=1" >> "{log_file}"
else
    echo "MODELS_DIR_EXISTS_DURING_CHOWN=0" >> "{log_file}"
fi
""")
    mock_chown.chmod(mock_chown.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Mock gosu: records its invocation
    mock_gosu = bin_dir / "gosu"
    mock_gosu.write_text(f"""#!/bin/sh
echo "GOSU: $*" >> "{log_file}"
""")
    mock_gosu.chmod(mock_gosu.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "VESTA_APP_USER": "testuser",
        "VESTA_DATA_DIR": str(data_dir),
        "VESTA_BAKED_MODELS": str(baked_dir),
        "VESTA_MODELS_DIR": str(data_dir / "models"),
    }

    result = subprocess.run(
        ["/bin/sh", str(ENTRYPOINT_PATH), "uvicorn", "vesta.main:app"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    # Verify symlink was created
    target_link = data_dir / "models" / "minishlab" / "potion-retrieval-32M"
    assert target_link.is_symlink()
    assert (target_link / "model.onnx").read_text() == "dummy-onnx"

    # Verify log output
    log_content = log_file.read_text()
    assert f"CHOWN: -R testuser:testuser {data_dir}" in log_content
    assert "MODELS_DIR_EXISTS_DURING_CHOWN=1" in log_content
    assert "GOSU: testuser uvicorn vesta.main:app" in log_content


def test_entrypoint_preserves_existing_models(tmp_path: Path) -> None:
    """Existing user models should not be overwritten by baked model symlinks."""
    data_dir = tmp_path / "data"
    baked_dir = tmp_path / "baked"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    # Baked model
    repo_baked = baked_dir / "testorg" / "testrepo"
    repo_baked.mkdir(parents=True)
    (repo_baked / "baked.bin").write_text("baked")

    # Pre-existing user model in data/models
    user_repo = data_dir / "models" / "testorg" / "testrepo"
    user_repo.mkdir(parents=True)
    (user_repo / "user.bin").write_text("user")

    # Mock chown and gosu
    for cmd in ("chown", "gosu"):
        f = bin_dir / cmd
        f.write_text("#!/bin/sh\nexit 0\n")
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "VESTA_APP_USER": "testuser",
        "VESTA_DATA_DIR": str(data_dir),
        "VESTA_BAKED_MODELS": str(baked_dir),
        "VESTA_MODELS_DIR": str(data_dir / "models"),
    }

    subprocess.run(
        ["/bin/sh", str(ENTRYPOINT_PATH), "echo", "test"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    # user directory should remain intact and not replaced with symlink
    assert not user_repo.is_symlink()
    assert (user_repo / "user.bin").read_text() == "user"
    assert not (user_repo / "baked.bin").exists()
