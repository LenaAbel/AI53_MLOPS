from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import Optimizer

from src.data import build_training_loaders
from src.model import BeanCNN
from src.provenance import write_run_manifest
from src.reproducibility import set_reproducibility


def resolve_device(requested: str) -> torch.device:
    """Resolve the execution device requested in params.yaml."""
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compute.device=cuda but CUDA is not available")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError("compute.device must be auto, cpu or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    if training and optimizer is None:
        raise ValueError("optimizer must be provided when training=True")

    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    # MLflow prints Unicode symbols; DVC may capture stdout using a legacy Windows encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # =========================================================
    # 1. LOAD CONFIGURATION
    # =========================================================
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))

    # =========================================================
    # 2. VERIFY REQUIRED PREPARED DATA
    # =========================================================
    train_manifest_path = Path("data/processed/manifests/train.csv")
    validation_manifest_path = Path("data/processed/manifests/validation.csv")

    missing_manifests = [
        str(path)
        for path in (train_manifest_path, validation_manifest_path)
        if not path.exists()
    ]
    if missing_manifests:
        raise FileNotFoundError(
            "Required prepared manifest(s) are missing: "
            + ", ".join(missing_manifests)
            + ". Run PREPARE first."
        )

    # =========================================================
    # 3. REPRODUCIBILITY
    # =========================================================
    seed = int(params["train"]["random_state"])
    set_reproducibility(seed)

    # =========================================================
    # 4. DEVICE + DATA + MODEL
    # =========================================================
    device = resolve_device(str(params["compute"]["device"]))
    train_loader, val_loader, class_names = build_training_loaders(
        Path("data/processed"), params
    )

    model = BeanCNN(
        num_classes=len(class_names),
        dropout=float(params["train"]["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(params["train"]["learning_rate"])
    )
    epochs = int(params["train"]["epochs"])
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    # =========================================================
    # 5. CONFIGURE MLFLOW
    # =========================================================
    tracking = params["tracking"]
    mlflow.set_tracking_uri(tracking["uri"])
    mlflow.set_experiment(tracking["experiment"])

    run_name = (
        f"cnn_aug-{params['augmentation']['enabled']}"
        f"_dropout-{params['train']['dropout']}"
        f"_lr-{params['train']['learning_rate']}"
    )

    # =========================================================
    # 6. ONE MLFLOW RUN = ONE TRAIN EXECUTION
    # =========================================================
    with mlflow.start_run(run_name=run_name) as run:
        # -----------------------------------------------------
        # CAPTURE PROVENANCE FIRST
        # -----------------------------------------------------
        # Written before TRAIN creates any file, so the Git status
        # describes the code exactly as it was when the Run started.
        run_manifest_path = Path("reports/run_manifest.json")
        run_manifest = write_run_manifest(
            params, str(device), run.info.run_id, run_manifest_path
        )
        training_identity = run_manifest["data"]["training_data_identity"]

        # -----------------------------------------------------
        # LOG RUN PARAMETERS
        # -----------------------------------------------------
        train_params = params["train"]
        augmentation_params = params["augmentation"]

        run_params = {
            "image_size": params["data"]["image_size"],
            "batch_size": train_params["batch_size"],
            "epochs": train_params["epochs"],
            "learning_rate": train_params["learning_rate"],
            "dropout": train_params["dropout"],
            "random_state": train_params["random_state"],
            "augmentation_enabled": augmentation_params["enabled"],
            "horizontal_flip": augmentation_params["horizontal_flip"],
            "rotation_degrees": augmentation_params["rotation_degrees"],
            "device": str(device),
            "training_data_identity": training_identity,
        }
        mlflow.log_params(run_params)

        # -----------------------------------------------------
        # LOG TRAINING DATASET INPUT
        # -----------------------------------------------------
        train_manifest_df = pd.read_csv(train_manifest_path)
        if "label" not in train_manifest_df.columns:
            raise ValueError("Training manifest must contain a 'label' column")

        training_dataset = mlflow.data.from_pandas(
            train_manifest_df,
            source=str(train_manifest_path),
            name="bean-training",
            targets="label",
        )
        mlflow.log_input(training_dataset, context="training")

        # =====================================================
        # 7. TRAIN + VALIDATE
        # =====================================================
        start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, None, device, False
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"], step=epoch)

        training_seconds = time.perf_counter() - start
        mlflow.log_metric("training_seconds", training_seconds)

        # =====================================================
        # 8. CREATE LOCAL OUTPUTS
        # =====================================================
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": list(class_names),
                "image_size": int(params["data"]["image_size"]),
                "dropout": float(params["train"]["dropout"]),
            },
            "models/model.pt",
        )

        summary = {
            "device": str(device),
            "epochs": epochs,
            "training_seconds": training_seconds,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_accuracy": history["val_accuracy"][-1],
            "final_val_macro_f1": history["val_macro_f1"][-1],
        }
        Path("reports/train_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        xs = range(1, epochs + 1)
        with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False,
                             "font.size": 10, "axes.titleweight": "bold"}):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
            axes[0].plot(xs, history["train_loss"], color="#102A43", marker="o", label="Train")
            axes[0].plot(xs, history["val_loss"], color="#159E72", marker="o", label="Validation")
            axes[0].set(title="Optimization", xlabel="Epoch", ylabel="Cross-entropy loss")
            axes[1].plot(xs, history["val_accuracy"], color="#102A43", marker="o", label="Accuracy")
            axes[1].plot(xs, history["val_macro_f1"], color="#159E72", marker="s", label="Macro-F1")
            axes[1].set(title="Validation evidence", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
            for ax in axes:
                ax.set_xticks(list(xs))
                ax.grid(axis="y", alpha=0.18)
                ax.legend(frameon=False)
            fig.suptitle("Candidate training | final epoch is the logged model", color="#102A43")
            fig.savefig("reports/training_curves.png", dpi=180)
            plt.close(fig)

        # =====================================================
        # 9. LOG ARTIFACTS
        # =====================================================
        # run_manifest.json already exists: it was written at the start of the Run
        report_paths = [
            Path("reports/training_curves.png"),
            Path("reports/train_summary.json"),
            run_manifest_path,
        ]
        for report_path in report_paths:
            mlflow.log_artifact(str(report_path))

        # =====================================================
        # 10. LOG FINAL PYTORCH MODEL
        # =====================================================
        # The logged model runs on CPU in evaluation mode
        cpu_model = model.to("cpu")
        cpu_model.eval()

        # Input example: one batch of normalized validation images (float32, NCHW)
        first_validation_batch = next(iter(val_loader))
        example_images = first_validation_batch[0].cpu()
        input_example = example_images.numpy()

        with torch.no_grad():
            example_logits = cpu_model(example_images)
        output_example = example_logits.numpy()

        signature = infer_signature(input_example, output_example)

        model_metadata = {
            "class_names": list(class_names),
            "image_size": int(params["data"]["image_size"]),
        }

        mlflow.pytorch.log_model(
            cpu_model,
            name="bean-cnn",
            signature=signature,
            input_example=input_example,
            metadata=model_metadata,
            serialization_format="pickle",
            code_paths=["src"],
        )

        print(f"Training complete. MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
