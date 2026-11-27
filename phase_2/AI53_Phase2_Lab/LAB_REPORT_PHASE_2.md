# AI53 Phase 2 lab report

At each milestone, copy the observed results under **Checkpoint**, then answer the matching **Question** from the Lab Guide.

## Milestone 1 · Inspect the model handoff

### Checkpoint 1 · Evidence

- GitHub repository URL: https://github.com/LenaAbel/AI53_MLOPS (Phase 2 is in the `phase_2/AI53_Phase2_Lab` folder)
- Expected model version: `phase1-reference-v1`
- Three class labels: `angular_leaf_spot`, `bean_rust`, `healthy`
- First pushed commit SHA: `cb8f5aa00afae8111abf817652c05bdfbeb88dcb`

### Question 1 : What does the model version identify? What does the Git SHA you just recorded identify?
MLOps

- The model version identifies the trained model itself. `phase1-reference-v1` names the exact weights in `model/bean-cnn.onnx` and the way they were trained. The API returns this version, so each prediction can be traced back to the model that produced it. 
- The Git SHA (retrieved via `git rev-parse HEAD`) is the unique identifier (hash) of the current commit pointed to by HEAD. 
   - In this MLOps context, it identifies the exact snapshot of the source code (training scripts, preprocessing, API code, and configuration) at that specific moment. This ensures full traceability and reproducibility of the model and the associated environment. If the model is retrained or the code is modified, a new SHA will be generated, allowing developers to track changes and roll back if necessary.
## Milestone 2 · Serve predictions with FastAPI

### Checkpoint 2 · Evidence

- Final API test count after TODO 1: 7 passed
- Browser prediction JSON for `data/current/leaf-01.jpg` (`UNAVAILABLE` and the error shown if the page gives no prediction):

```json
{
  "label": "angular_leaf_spot",
  "confidence": 0.5016346573829651,
  "model_version": "phase1-reference-v1"
}
```

### Question 2 : What does /ready check that /health does not check? Why is the model version part of a prediction response?

- The `/health` endpoint checks whether the API service is running and responsive, but it does not verify if the model is loaded and ready to make predictions. 
- In contrast, the `/ready` endpoint checks both the service's responsiveness and whether the model is fully loaded and operational, ensuring that predictions can be made successfully.
- The model version is part of each prediction response because it links every answer to the model that produced it. Because several model versions can be deployed over time, the version allows for traceability and reproducibility. Without the version, a wrong or surprising prediction could not be traced back to its model. The client can also check that the API really uses the expected model, here `phase1-reference-v1`.

## Milestone 3 · Run tests and build in GitHub Actions

### Checkpoint 3 · Evidence

- Local branch: `main`
- Local commit SHA: `005f1efc9c78fd1b7bda04907bf621b4dca9871a`
- Local API test count: 7 passed
- Actions run source (`main`, `PULL REQUEST` or `UNAVAILABLE`): `main`
- Actions run URL: https://github.com/LenaAbel/AI53_MLOPS/actions/runs/46017239680
- `tests` job status: `PASS`
- `docker-build` job status: `PASS`
- Container smoke result or first failed CI step: `PASS` (the step "Check the running container" succeeded)

For the two job statuses, use `PASS`, `FAIL`, `PENDING`, `SKIPPED`, `CANCELLED` or `UNAVAILABLE`. Use `SKIPPED` when Actions shows that a job was skipped. Use `UNAVAILABLE` for missing CI evidence.

### Question 3 : What does needs: tests enforce? Why do you check the run on main after the pull request is merged?

- `needs: tests` makes the `docker-build` job wait for the `tests` job. The image is only built when the tests pass. If a test fails, `docker-build` is skipped. No image is built or started from code that is known to be broken.
- The pull request run tests a trial merge of `ci-checks` into `main`, made at that moment. The commit that really ends up on `main` after the merge can be different. `main` may also have changed in the meantime. The run on `main` tests the exact SHA that will be released. This is the SHA used for staging in Milestone 4, so its CI evidence must belong to that exact commit.

## Milestone 4 · Verify a staging release

### Checkpoint 4 · Evidence

- Mode (`RENDER` or `LOCAL`): `RENDER`
- Target URL: https://ai53-leaf-lena-abel.onrender.com/
- Tested commit SHA: `005f1efc9c78fd1b7bda04907bf621b4dca9871a` (shown as `005f1ef` in Render Deploys)
- Expected model version from Checkpoint 1: `phase1-reference-v1`
- Observed model version from `/ready` (`UNAVAILABLE` if it returned none): `phase1-reference-v1`
- Smoke check result (`PASS` or `FAIL`): `PASS`
- First error, if any: None
- Earlier Render attempt, if switching to local (URL if available and reason for switching): Not applicable, Render was used directly. The Render root directory was set to `phase_2/AI53_Phase2_Lab`, because Phase 2 is a sub-folder of the repository.

```
PASS GET /ready (model_version=phase1-reference-v1)
PASS GET /health
PASS POST /predict (label=angular_leaf_spot, model_version=phase1-reference-v1)
```

### Question 4 : Suppose /health returns 200 and /predict reports an unexpected model version. Would you approve this deployment? What would you check first?

- No, this deployment should not be approved. `/health` only shows that the server process answers. It says nothing about the model. An unexpected version means the API serves a model that was not checked. Its predictions cannot be trusted or traced back to the expected model.
- First, I would check which commit is really deployed. In Render, the deployed SHA must match the SHA from Checkpoint 3. A different commit may contain another model file. Then I would check the model inside that commit. The version comes from the metadata of `model/bean-cnn.onnx`, so the file itself may have been replaced. I would compare it with `model/README.md` and with the `/ready` response.

## Milestone 5 · Compare input distributions

### Checkpoint 5 · Evidence

Fill each row from its own `summary.json`. Use the file from the CURRENT report for the first row and the file from the SHIFTED report for the second. In both files, `current_mean` is the mean brightness of the batch you checked. Use the exact `score` (p-value) and `threshold` (cutoff). Enter `DRIFT` when `drift_detected` is true and `NO DRIFT` when it is false.

| Batch | Reference mean (`reference_mean`) | Batch mean (`current_mean`) | Method | Score | Threshold | Result (`DRIFT` or `NO DRIFT`) |
| --- | --- | --- | --- | --- | --- | --- |
| `CURRENT` | | | | | | |
| `SHIFTED` | | | | | | |

- Final full test count:

### Question 5 · Explanation

Answer:
