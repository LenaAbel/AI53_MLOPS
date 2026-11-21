# LAB REPORT - Léna Abel

## Milestone 1 - Make training repeatable

> In your report, explain what pyproject.toml, uv.lock and .venv/ are for. Use the environment from setup.

- **pyproject.toml** : This file is used to define the project and the dependencies required for the project. It defines different informations such as the project metadat (project name,authors, license, etc.) As well as the Python version and the dependencies required for the project. It is used by package managers like pip to install the dependencies and build the project.
- **uv.lock** : This file is used to lock the exact versions of all dependencies required for the project. It records the exact package versions, sources, and integrity hashes for both direct packages and sub-dependencies across operating systems. It is automatically generated and used by uv to reproduce the exact same environment on any machine.
- **venv/** : This directory is used to create a virtual environment for the project. It contains a self-contained Python environment with its own installation directories, which allows you to install packages without affecting the global Python installation. It is used to isolate the project dependencies and avoid conflicts with other projects.

> Does random_state=42 guarantee bit-for-bit identical results on every CPU/GPU/software
stack? Explain briefly.

- No, random_state=42 does not guarantee bit-for-bit identical results on every CPU/GPU/software stack. This parameter is used to initialize the random number generator within a specific library so results are reproducible on the same setup. However, different factors such as hardware differences (CPU vs. GPU), multi-threading execution order, and library or driver versions can still cause slight variations. It is mostly used to get consistent results when running the same code in the same environment.

## Milestone 2 - Extract and track the images

> In one sentence each, state what the Git remote stores, what the DVC remote stores,
and whether git push transfers the DVC-managed raw images.

- **Git remote**: The Git remote (the GitHub repository) stores the project's code.
- **DVC remote**: The DVC remote (../ai53-phase1-dvc-storage) stores the actual large data artifacts tracked by DVC. Specifically the raw bean-leaf images from data/raw/ (as well as DVC-cached processed datasets and local model checkpoints).
- **git push**: No, git push only transfers the small `data/raw.dvc` metadata pointer to GitHub, while a separate dvc push command is required to upload the actual DVC-managed raw images to the DVC remote.

## Milestone 4 - Design the DVC pipeline from scratch

> Should PREPARE depend on reports/data_quality.json?

- Yes, PREPARE should depend on `reports/data_quality.json`. This report is only written when every data check passes. If VALIDATE fails, the report is not created and DVC stops before PREPARE. This dependency also creates the link between VALIDATE and PREPARE in the DVC graph. As a result, PREPARE never runs on rejected data.

> Should PREPARE also depend directly on data/raw?

- Yes, PREPARE should also depend directly on `data/raw`. PREPARE reads the raw images itself to resize them and build the manifests. The quality report only stores the number of images per class. An image could be replaced without changing these counts. In that case, the report would stay the same and PREPARE would not run again without this dependency.

> Which changes in params.yaml should make TRAIN run again?

- TRAIN should run again when a setting that affects the model or the training changes. This includes `data.image_size`, all the `train` settings (batch_size, epochs, learning_rate, dropout, random_state) and all the `augmentation` settings (enabled, horizontal_flip, rotation_degrees). `compute.device` is also included because CPU and GPU can give slightly different results. `tracking.experiment` is included because it decides where the Run is saved in MLflow. Changes to `selection.run_id`, `evaluation.batch_size` or `tracking.uri` should not make TRAIN run again.

> Why should EVALUATE track selection.run_id?

- EVALUATE loads the selected model from MLflow, not from DVC. `selection.run_id` is the only link between DVC and this model. If the selected Run changes, the saved test results no longer match the chosen model. Tracking this setting makes DVC run EVALUATE again in that case. It also keeps a record in Git of which model was tested.

> Which file contains scores that DVC should track as metrics?

- `reports/test_metrics.json` contains the final test scores (test_accuracy and test_macro_f1). DVC should track it as metrics with `cache: false` so that it stays in Git. `reports/train_summary.json` is also tracked as metrics because it stores the validation scores of the last epoch. This makes it possible to compare Runs with `dvc metrics show` and `dvc metrics diff`.

## Milestone 5 - Record what each MlFlow Run used
> In one sentence per tool, explain what Git, DVC and MLflow each record about a
training Run.

- **Git**: Git records the exact code, the pipeline definition (`dvc.yaml`, `dvc.lock`) and the settings in `params.yaml` used for the Run, identified by the commit SHA saved in the run manifest.
- **DVC**: DVC records the exact version of the data and outputs used by the Run, by storing the hashes of the raw images, the prepared images and `models/model.pt` in `dvc.lock`.
- **MLflow**: MLflow records the Run itself, with its parameters, its metrics at each epoch, the training dataset, the report artifacts and the logged `bean-cnn` model.

## Milestone 6 - Predict which stages need to run again

> Replace each ? with RERUN if the stage should run again or REUSED if DVC can keep its outputs. In the EVALUATE column, use NEEDS UPDATE instead of RERUN if the saved test results would no longer match.

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`; the quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

> Explain your labels for at least two rows: one setting change and one file edit. For each stage in those rows, name the input that changes or explain why its inputs stay the same. Use your dvc.yaml for this explanation.

- **Change `train.learning_rate`**:
  - VALIDATE is reused. It has no params in `dvc.yaml` and its code and data do not change.
  - PREPARE is reused. It only tracks `data.image_size`, so the learning rate is not one of its inputs.
  - TRAIN runs again. `train.learning_rate` is listed in its params.
  - EVALUATE is reused. It does not track `train.learning_rate`. The selected Run ID stays the same, so the saved test results still match the chosen model.
- **Change `data.image_size`**:
  - VALIDATE is reused. It does not track any setting from `params.yaml`.
  - PREPARE runs again. `data.image_size` is listed in its params.
  - TRAIN runs again. `data.image_size` is in its params, and the prepared images and manifests also change.
  - EVALUATE needs an update. `data.image_size` and the prepared test images change. The saved test results no longer match the current data. The selected Run was also trained with the old image size, so EVALUATE would reject it until a new Run is selected.
- **Edit only `README.md`**:
  - All the stages are reused. `README.md` is not a dependency of any stage in `dvc.yaml`.

> Before training, choose the main validation metric and how you will break a tie in Milestone 7. Write down what you expect a lower learning rate to change.

- **Main metric**: The main validation metric is `val_macro_f1` at the last epoch. This is a three-class problem. Macro-F1 gives the same weight to each class. A model that ignores one class gets a low score, even if its accuracy stays high.
- **Tie-break rule**: If two Runs have the same `val_macro_f1`, the Run with the higher `val_accuracy` is chosen. If they are still equal, the Run with the lower `val_loss` is chosen.
- **Lower learning rate**: With a learning rate of 0.0003 instead of 0.001, the weights change more slowly at each step. Training only lasts 4 epochs. The model should therefore learn less in the same time. We expect a higher `val_loss` and slightly lower validation scores than Run A. Training time should stay about the same, because the number of epochs does not change.