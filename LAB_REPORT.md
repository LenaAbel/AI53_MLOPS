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

> Make a second table in your report with the same stage columns and one row per check. Include the relevant terminal lines for each check. Compare the results with your predictions and explain any differences.

- **Run A**: `3567da7a5b9b4b5fa98cb9eec553a5de` (baseline, `learning_rate` = 0.001)
- **Run B**: `c89d8410bea6481880c6d6c6848ab1fc` (`learning_rate` = 0.0003)

| Check | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Edit `src/validate_data.py` (comment only) | RERUN | REUSED | REUSED | NOT RUN |
| Run again without changing anything | REUSED | REUSED | REUSED | NOT RUN |
| Change `train.learning_rate` to 0.0003 | REUSED | REUSED | RERUN | NOT RUN |

**Edit `src/validate_data.py` (comment only)**

```
'data/raw.dvc' didn't change, skipping
Running stage 'validate':
> uv run --locked python -m src.validate_data
  "status": "PASS",
  "total_images": 1295
Updating lock file 'dvc.lock'
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
```

**Run again without changing anything**

```
'data/raw.dvc' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.
```

**Change `train.learning_rate` to 0.0003**

```
'data/raw.dvc' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Running stage 'train':
> uv run --locked python -m src.train
Epoch 04/4 | train_loss=0.9198 | val_loss=0.8927 | val_acc=0.6165 | val_macro_f1=0.6091
Training complete. MLflow Run ID: c89d8410bea6481880c6d6c6848ab1fc
```

- **Comparison with the predictions**: The results match the predictions for VALIDATE, PREPARE and TRAIN. EVALUATE is marked NOT RUN instead of REUSED. `dvc repro train` only runs TRAIN and the stages before it, so EVALUATE is never checked.
- **Edit `src/validate_data.py`**: VALIDATE ran again because its code is one of its dependencies. The new report had the same content as before. PREPARE depends on the report and not on the code, so DVC skipped it. TRAIN was also skipped because its inputs did not change.
- **Run again without changing anything**: This check was run after the comment was added. No dependency or setting had changed since the previous run. DVC skipped every stage.
- **Change `train.learning_rate`**: Only TRAIN ran again. `train.learning_rate` is listed in its params and not in the params of VALIDATE or PREPARE.
- **Lower learning rate results**: Run B reached a `val_macro_f1` of 0.6091 and a `val_accuracy` of 0.6165. Run A reached 0.6254 and 0.6391. The `val_loss` was higher for Run B (0.8927 against 0.8328). This supports the prediction. The model learned more slowly in the same 4 epochs. Training time was 130.7 s for Run B and 152.9 s for Run A. The gap is probably due to the load of the machine, since both Runs train for the same number of epochs.
- **Git dirty flag**: The run manifests of Run A and Run B both show `"dirty": true`. Run B was started from a clean commit. Before running a stage, DVC deletes its old outputs. Some of them are tracked by Git (`reports/` with `cache: false`). `git status` then saw them as deleted files. For Run A, VALIDATE and PREPARE had also created `dvc.lock` and new reports just before TRAIN. The flag was fixed in `src/provenance.py`. It now ignores `reports/` and `dvc.lock`, so it only describes the code and the settings.

## Milestone 7 - Compare three controlled runs

> Return the learning rate to Run A's value (0.001). Then choose exactly one change to train.dropout, augmentation.enabled or train.batch_size. Write down what you expect to happen and why.

- **Controlled change**: Run C uses the same settings as Run A, except `train.dropout`, which goes from 0.30 to 0.50. The learning rate is back to 0.001.
- **Prediction**: Dropout turns off a random part of the neurons at each training step. A higher dropout adds more regularization. It helps against overfitting, but it also makes learning slower. Training only lasts 4 epochs, and Run A does not overfit yet (its `val_loss` is lower than its `train_loss`). We therefore expect a slightly higher `train_loss` and validation scores close to or slightly lower than Run A. Training time should stay about the same, because dropout does not change the amount of computation.