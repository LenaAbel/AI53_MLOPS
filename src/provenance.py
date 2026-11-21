from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def training_data_identity(
        train_manifest: Path = Path("data/processed/manifests/train.csv"),
        validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    The manifest records:
      - Git commit SHA and whether the working tree is dirty;
      - identity of the prepared training + validation manifests;
      - Python, PyTorch, torchvision, DVC and MLflow versions;
      - OS/platform and CUDA runtime (when available);
      - compute device; random seed; image size; training parameters;
      - MLflow Run ID.
    """
    # 1. Git: commit SHA and clean/dirty working tree
    git_commit = "unavailable"
    git_dirty = None  # None means "unknown" when Git cannot answer

    try:
        commit_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if commit_result.returncode == 0:
            git_commit = commit_result.stdout.strip()

        # `git status --porcelain` prints one line per changed file.
        # The pipeline outputs (reports/ and dvc.lock) are excluded:
        # DVC deletes or rewrites them during `dvc repro`, so they would
        # always make the folder look dirty. Only code and settings count.
        status_command = [
            "git", "status", "--porcelain", "--",
            ".",
            ":(exclude)reports",
            ":(exclude)dvc.lock",
        ]
        status_result = subprocess.run(status_command, capture_output=True, text=True)
        if status_result.returncode == 0:
            changed_files = status_result.stdout.strip()
            git_dirty = changed_files != ""
    except FileNotFoundError:
        # Git is not installed: keep the "unavailable" values
        pass

    # 2. Data: identity of the training and validation manifests
    data_identity = training_data_identity()

    # 3. Environment: OS, Python and package versions
    environment = {
        "os": platform.system(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    for package_name in ["torch", "torchvision", "dvc", "mlflow"]:
        try:
            environment[package_name] = version(package_name)
        except PackageNotFoundError:
            environment[package_name] = "not installed"

    if torch.cuda.is_available():
        environment["cuda"] = torch.version.cuda
    else:
        environment["cuda"] = "unavailable"

    # 4. Settings used for training
    train_params = params["train"]
    augmentation_params = params["augmentation"]

    configuration = {
        "image_size": params["data"]["image_size"],
        "batch_size": train_params["batch_size"],
        "epochs": train_params["epochs"],
        "learning_rate": train_params["learning_rate"],
        "dropout": train_params["dropout"],
        "augmentation_enabled": augmentation_params["enabled"],
        "horizontal_flip": augmentation_params["horizontal_flip"],
        "rotation_degrees": augmentation_params["rotation_degrees"],
    }

    # 5. Execution: device, seed and MLflow Run ID
    execution = {
        "device": device,
        "random_state": train_params["random_state"],
        "mlflow_run_id": mlflow_run_id,
    }

    return {
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
        },
        "data": {
            "training_data_identity": data_identity,
        },
        "environment": environment,
        "configuration": configuration,
        "execution": execution,
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
