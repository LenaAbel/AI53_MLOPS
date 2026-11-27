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

- Local branch:
- Local commit SHA:
- Local API test count:
- Actions run source (`main`, `PULL REQUEST` or `UNAVAILABLE`):
- Actions run URL:
- `tests` job status:
- `docker-build` job status:
- Container smoke result or first failed CI step:

For the two job statuses, use `PASS`, `FAIL`, `PENDING`, `SKIPPED`, `CANCELLED` or `UNAVAILABLE`. Use `SKIPPED` when Actions shows that a job was skipped. Use `UNAVAILABLE` for missing CI evidence.

### Question 3 · Explanation

Answer:

## Milestone 4 · Verify a staging release

### Checkpoint 4 · Evidence

- Mode (`RENDER` or `LOCAL`):
- Target URL:
- Tested commit SHA:
- Expected model version from Checkpoint 1:
- Observed model version from `/ready` (`UNAVAILABLE` if it returned none):
- Smoke check result (`PASS` or `FAIL`):
- First error, if any:
- Earlier Render attempt, if switching to local (URL if available and reason for switching):

### Question 4 · Explanation

Answer:

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
