# DistillationCommandParser

Sequence-level distillation of Phi-3.5-mini-instruct (3.8B, teacher) into EleutherAI/pythia-70m (student) for a command parser that returns exactly one JSON object with the keys `intent`, `date`, `time`, `place`. The student is served by a FastAPI service on a local k3d Kubernetes cluster, with SeaweedFS as the object store.

Author: Puneet Kumar (roll no. 230101025), Cloud Computing (CS3081) mini project.

## Architecture

    commands_pool.jsonl (3,000 commands)
      -> teacher labelling (Ollama, Phi-3.5, temp 0, seed 42), 2 sharded Jobs
      -> clean + verify -> data/v1/train.jsonl (2,435 kept)
      -> train pythia-70m on the first 1,500 records -> models/student-v1, student-v2
      -> model published as a GitHub Release asset
      -> FastAPI service (parser-service:v1) downloads the model from SeaweedFS (S3)
      -> Kubernetes Deployment + Service + Ingress + HPA on k3d (cluster cloud-mini)

## Results (validation set, n = 100)

| Model | Validity | Intent | Date | Time | Place | Full frame |
|---|---|---|---|---|---|---|
| Teacher (Phi-3.5) | 91 | 72 | 72 | 80 | 82 | 44 |
| Student v1 | 100 | 77 | 77 | 84 | 86 | 39 |
| Student v2 | 100 | 74 | 82 | 85 | 86 | 45 |
| Untrained pythia-70m | 95 | 16 | 5 | 3 | 81 | 0 |

Scores are percentages. The marking gate is JSON validity >= 95% and full frame >= half of the teacher's (22); both students pass it, and v2 beats the teacher on full frame. Standard load test (CPU-only): p50 1200 ms, p95 3000 ms, throughput 2.71 req/s, 0 failures in 324 requests. Cost: Rs 0.81 per 1,000 requests (hypothetical rate card in `student_kit/data/rate_card.json`). Full details in `report/report.pdf`.

## Prerequisites

- Docker Desktop (running) and a Unix shell
- `k3d` and `kubectl`
- Python 3.12+ with a virtual environment (for training, tests and the report builder); `pip install -r train/requirements.txt` for training
- `gh` (GitHub CLI) only if you want to inspect CI runs from the terminal

## Quickstart

    git clone https://github.com/puneetkumar-2005/DistillationCommandParser.git
    cd DistillationCommandParser
    make up

`make up` creates or starts the k3d cluster, builds and imports the service image, applies SeaweedFS, seeds the v1 model from the GitHub Release into S3, applies the app manifests and waits for the Deployment to be Available. It took 2m23s on the author's laptop. Check it works:

    curl -s -X POST http://localhost:8080/parse \
      -H 'Content-Type: application/json' \
      -d '{"text": "set an alarm for tomorrow at seven thirty am"}'
    # {"intent":"alarm_set","date":"tomorrow","time":"seven thirty am","place":null}

Tear the cluster down with:

    make down

Other useful commands:

    make train            # train the student on all 1,500 records (training/config.yaml)
    make train SUBSET=50  # train on the first 50 records only, to verify training runs
    python3 -m pytest tests/ -v   # 8 unit tests
    ruff check .          # lint (student_kit/ is provided code and is excluded)

## Model

The trained student is not in git. Download it from the GitHub Release:

https://github.com/puneetkumar-2005/DistillationCommandParser/releases/tag/student-v1

Asset `student-v1.zip` (216.33 MiB), sha256 `6c0452f4c276c1d61e61d9c41602abf634ba965fd6755385380383d1b6d2bf9a`. The checksum is recorded in `results/task2/model.sha256`, and the service and CI both fetch the model from this Release (or from S3, where the same files live under `models/v1/`).

## CI

GitHub Actions (`.github/workflows/ci.yml`), in order: lint (ruff) -> unit tests -> secret scan (gitleaks, nothing found) -> download the model from the Release -> CPU torch evaluation -> `score.py --min-validity 95` quality gate.

- Green run: https://github.com/puneetkumar-2005/DistillationCommandParser/actions/runs/42458414578 (2m3s)
- Green run: https://github.com/puneetkumar-2005/DistillationCommandParser/actions/runs/42460797290 (1m59s)
- Deliberately failed run: https://github.com/puneetkumar-2005/DistillationCommandParser/actions/runs/42460805060 (PR #1, `--max-new-tokens 1` truncates the model output, so the run fails at "Evaluate and quality gate" with all earlier steps green; the PR was closed unmerged.)

## Evidence

- `results/task1/` -- labelling logs, clean counts, checksum check
- `results/task2/` -- training logs, eval logs, predictions and scores (teacher, student v1/v2, untrained baseline)
- `results/task3/` -- 50-request run and peak memory
- `results/task4/` -- load tests, rolling update, recovery, cold start, HPA and per-pod CPU/memory
- `results/task6/` -- `make up` run notes
- `report/report.pdf` -- the full report (max 8 pages)
- `network/` -- Task 5 subnets and NACL

## Notes on measurement

All service, load and latency measurements are CPU-only: the service image uses CPU-only torch, pods are limited to 1 CPU and 1 GiB, and a k3d cluster cannot see the host GPU. The laptop GPU (RTX 3050) was used only for Task 1 teacher labelling and Task 2 training, because CPU labelling was measured at roughly 13-14 hours. Tasks 1 and 2 have no latency measurement in the marking.

## AI use

See `AI_USE.md`. In short: an AI assistant (Muse) drafted code, configs, docs and tests and explained each step; the author ran every command on their own laptop, verified every result on their own machine before committing, and fixed the failures the runs exposed.
