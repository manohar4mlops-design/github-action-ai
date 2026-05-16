# Complete Project Explanation: Deploying a Machine Learning Model with GitHub Actions and Hugging Face Spaces

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Overall Architecture](#2-overall-architecture)
3. [Folder Structure Explained](#3-folder-structure-explained)
4. [The Machine Learning Model](#4-the-machine-learning-model)
5. [app/Dockerfile — Building the Docker Image](#5-appdockerfile--building-the-docker-image)
6. [app/requirements.txt — Python Dependencies](#6-apprequirementstxt--python-dependencies)
7. [app/main.py — The API Server](#7-appmainpy--the-api-server)
8. [hf-space/Dockerfile — The Deployment Pointer](#8-hf-spacedockerfile--the-deployment-pointer)
9. [tests/test_app.py — Automated Tests](#9-teststest_apppy--automated-tests)
10. [.github/workflows/deploy.yml — The CI/CD Pipeline](#10-githubworkflowsdeployyml--the-cicd-pipeline)
11. [Hugging Face Spaces — Where the Model Runs](#11-hugging-face-spaces--where-the-model-runs)
12. [Docker Hub — Where the Image Lives](#12-docker-hub--where-the-image-lives)
13. [GPU or CPU? What Hardware Is Used?](#13-gpu-or-cpu-what-hardware-is-used)
14. [Model Versioning — v1 vs v2](#14-model-versioning--v1-vs-v2)
15. [GitHub Secrets — How Credentials Are Managed](#15-github-secrets--how-credentials-are-managed)
16. [End-to-End Flow: What Happens When You Push Code](#16-end-to-end-flow-what-happens-when-you-push-code)

---

## 1. What This Project Does

This project demonstrates a complete **MLOps CI/CD pipeline** using GitHub Actions to automatically deploy a machine learning model to Hugging Face Spaces.

**In simple terms:**
- A sentiment analysis model (DistilBERT) is packaged inside a Docker image
- That Docker image is stored on Docker Hub
- Every time a developer pushes code to GitHub, a pipeline automatically:
  - Runs tests to validate the code
  - Verifies the Docker image exists
  - Deploys the model to Hugging Face Spaces
  - Confirms the deployment is live

**The key MLOps principle demonstrated:** No one manually logs into a server, copies files, or restarts services. Everything is automated through the pipeline.

---

## 2. Overall Architecture

```
Developer pushes code
        │
        ▼
   GitHub Repository
        │
        ▼
  GitHub Actions (CI/CD)
   ┌────┴────┐
   │         │
   ▼         ▼
validate  check-image ──── Docker Hub
   │         │           (image stored here)
   └────┬────┘
        │ both pass
        ▼
      deploy ──────────── Hugging Face Spaces
        │                 (model runs here)
        ▼
   health-check ──────── Live API endpoint
```

**Three external services are involved:**
| Service | Role |
|---|---|
| **GitHub** | Stores code and runs the pipeline |
| **Docker Hub** | Stores the pre-built Docker image |
| **Hugging Face Spaces** | Runs the model as a live API |

---

## 3. Folder Structure Explained

```
github-action-model/
│
├── .github/
│   └── workflows/
│       └── deploy.yml        ← The CI/CD pipeline definition
│
├── app/                      ← Application code (used to build Docker image)
│   ├── main.py               ← FastAPI server with DistilBERT model
│   ├── requirements.txt      ← Python package dependencies
│   ├── Dockerfile            ← Instructions to build the Docker image
│   └── __init__.py           ← Makes app/ importable as a Python package
│
├── hf-space/
│   └── Dockerfile            ← One-line file pointing to Docker Hub image
│
├── tests/
│   └── test_app.py           ← Automated tests run by the pipeline
│
├── docs/
│   └── PROJECT_EXPLAINED.md  ← This file
│
└── pytest.ini                ← Test configuration
```

**Important distinction:**
- `app/` contains everything needed to **build** the Docker image. This is done **once manually** on a developer's machine. The CI/CD pipeline never builds the image.
- `hf-space/` contains what the pipeline **actually deploys** — a single Dockerfile that tells Hugging Face which image to pull from Docker Hub.

---

## 4. The Machine Learning Model

**Model name:** `distilbert-base-uncased-finetuned-sst-2-english`

**What is DistilBERT?**
DistilBERT is a smaller, faster version of BERT (Bidirectional Encoder Representations from Transformers), a language model created by Google. "Distilled" means it was trained to mimic a larger model (BERT) while being 40% smaller and 60% faster, with only a 3% drop in accuracy.

**What task does it perform?**
Sentiment Analysis — given a sentence of text, it predicts whether the sentiment is POSITIVE or NEGATIVE, along with a confidence score (0.0 to 1.0).

**Example:**
```
Input:  "I love this course!"
Output: {"label": "POSITIVE", "score": 0.9999}
```

**Where does the model come from?**
The model is NOT stored inside the Docker image. It lives on Hugging Face Hub at:
`https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english`

When the container starts on Hugging Face Spaces, the `transformers` library automatically downloads the model weights (~250MB) from Hugging Face Hub into a local cache inside the container. This happens only on first startup.

**Who trained this model?**
Hugging Face. It was fine-tuned on the SST-2 (Stanford Sentiment Treebank) dataset — a collection of movie reviews labeled as positive or negative.

---

## 5. app/Dockerfile — Building the Docker Image

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Line by line explanation:**

| Line | What it does |
|---|---|
| `FROM python:3.11-slim` | Starts from an official Python 3.11 base image. The `slim` variant is smaller — it excludes documentation and unnecessary packages |
| `WORKDIR /app` | Sets the working directory inside the container to `/app`. All subsequent commands run from here |
| `COPY requirements.txt .` | Copies `requirements.txt` from your machine into the container |
| `RUN pip install --no-cache-dir -r requirements.txt` | Installs all Python packages. `--no-cache-dir` keeps the image smaller by not caching downloaded packages |
| `COPY main.py .` | Copies the FastAPI application code into the container |
| `EXPOSE 7860` | Documents that the container listens on port 7860. Hugging Face Spaces requires this specific port |
| `CMD [...]` | The command that runs when the container starts. Starts the uvicorn web server serving the FastAPI app on port 7860 |

**Why is this done manually and not in the pipeline?**
Building this image takes several minutes because PyTorch alone is ~779MB. If the pipeline built the image on every push, each deployment would take 10+ minutes. Instead, the image is built once, stored on Docker Hub, and the pipeline just references it — making deployments fast.

---

## 6. app/requirements.txt — Python Dependencies

```
fastapi==0.111.0
uvicorn==0.29.0
transformers==4.41.0
torch==2.3.0
pydantic==2.7.0
```

| Package | Purpose |
|---|---|
| `fastapi` | The web framework for building the REST API |
| `uvicorn` | The ASGI web server that runs the FastAPI application |
| `transformers` | Hugging Face library that provides DistilBERT and the `pipeline()` function |
| `torch` | PyTorch — the deep learning framework DistilBERT runs on |
| `pydantic` | Data validation library used by FastAPI to validate request and response shapes |

**Why are exact versions pinned?**
Pinning versions (e.g., `torch==2.3.0` instead of just `torch`) ensures the Docker image always installs the exact same packages. Without pinning, a rebuild six months later might install a newer version of torch that breaks the model.

---

## 7. app/main.py — The API Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

MODEL_VERSION = "v2"

app = FastAPI(title="DistilBERT Sentiment API")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    result = classifier(request.text)[0]
    return PredictResponse(
        label=result["label"],
        score=round(result["score"], 4),
        model_version=MODEL_VERSION
    )
```

**Section by section:**

**`MODEL_VERSION = "v2"`**
A constant that identifies which version of the application is running. This is returned in every API response so anyone calling the API can confirm which version is deployed.

**`app = FastAPI(...)`**
Creates the FastAPI application instance. FastAPI automatically generates interactive API documentation at `/docs`.

**`classifier = pipeline(...)`**
This line runs when the container starts. It downloads the DistilBERT model from Hugging Face Hub and loads it into memory. The `pipeline()` function is a high-level abstraction from the `transformers` library — it handles tokenization (converting text to numbers), model inference, and converting the output back to human-readable labels.

**`class PredictRequest(BaseModel)`**
Defines what the API expects as input. FastAPI uses this to automatically validate incoming requests. If someone sends a request without the `text` field, FastAPI returns a 422 error automatically.

**`class PredictResponse(BaseModel)`**
Defines the exact shape of the API response. FastAPI uses this to validate the output and generate documentation.

**`/health` endpoint**
A lightweight endpoint used by the pipeline to confirm the service is running. Returns the current version so it is easy to verify which version is deployed.

**`/predict` endpoint**
The main endpoint. Accepts a JSON body with a `text` field, runs it through DistilBERT, and returns the sentiment label, confidence score, and model version. The empty text check (`if not request.text.strip()`) prevents the model from receiving blank input, which would cause an error.

---

## 8. hf-space/Dockerfile — The Deployment Pointer

```dockerfile
FROM 03sarath/distilbert-sentiment:v1

EXPOSE 7860
```

This is the most important file in the deployment process — and it is intentionally minimal.

**What it does:**
- `FROM 03sarath/distilbert-sentiment:v1` — tells Hugging Face Spaces to pull this specific image from Docker Hub and run it
- `EXPOSE 7860` — tells Hugging Face the container listens on port 7860

**Why is this a separate Dockerfile from `app/Dockerfile`?**
These two Dockerfiles serve completely different purposes:

| | `app/Dockerfile` | `hf-space/Dockerfile` |
|---|---|---|
| **Purpose** | Build the full image | Point to an already-built image |
| **Size** | Defines a ~8.67GB image | 2 lines |
| **When used** | Once, manually by developer | Every push, by the pipeline |
| **Contains** | Python, packages, code | Nothing — just a reference |

**How version upgrades work:**
To upgrade from v1 to v2, a developer changes one word in this file:
```dockerfile
FROM 03sarath/distilbert-sentiment:v2
```
Commits and pushes. The pipeline detects the change, reads the new tag, verifies it exists on Docker Hub, and deploys it to Hugging Face Spaces. The entire upgrade is automated from that single line change.

---

## 9. tests/test_app.py — Automated Tests

```python
import sys
from unittest.mock import MagicMock

_mock_clf = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.9998}])
_mock_transformers = MagicMock()
_mock_transformers.pipeline.return_value = _mock_clf
sys.modules["transformers"] = _mock_transformers
sys.modules["torch"] = MagicMock()

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_label_and_score():
    response = client.post("/predict", json={"text": "I love this product!"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "score" in body

def test_predict_empty_text_returns_400():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
```

**Why mock `transformers` and `torch`?**
Installing PyTorch in the pipeline just to run tests would take 5+ minutes and add complexity. The tests are checking the **API logic** (routing, validation, response shape) — not the model itself. So we replace `transformers` and `torch` with fake (mock) objects using Python's `unittest.mock`. The `sys.modules` trick intercepts the import before Python tries to load the real libraries.

**The three tests:**

| Test | What it verifies |
|---|---|
| `test_health` | The `/health` endpoint returns HTTP 200 and a `status: ok` field |
| `test_predict_returns_label_and_score` | The `/predict` endpoint returns a response containing `label` and `score` fields |
| `test_predict_empty_text_returns_400` | Sending empty text returns HTTP 400 (bad request), not a server crash |

**Why does `test_predict_empty_text_returns_400` matter?**
In production, users will send unexpected inputs. This test confirms the validation logic works — the API rejects bad input gracefully rather than passing it to the model and crashing.

---

## 10. .github/workflows/deploy.yml — The CI/CD Pipeline

This file defines the entire automated pipeline. GitHub reads this file and executes it every time code is pushed to the `main` branch.

### Trigger

```yaml
on:
  push:
    branches:
      - main
```

The pipeline only runs on pushes to `main`. Pushes to other branches (feature branches, etc.) are ignored.

### Global Variable

```yaml
env:
  DOCKER_REPO: 03sarath/distilbert-sentiment
```

The Docker Hub repository name, available to all jobs. Defined once to avoid repetition.

---

### Job 1: validate

```yaml
validate:
  name: Validate Code
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - run: pip install pytest fastapi httpx pydantic
    - run: pytest tests/ -v
```

**What it does:**
GitHub spins up a fresh Ubuntu virtual machine, downloads the repo code, installs only the lightweight test dependencies (no torch, no transformers — because they are mocked), and runs the 3 tests. If any test fails, the entire pipeline stops and deployment never happens.

**MLOps significance:** This is the quality gate. Bad code cannot reach production.

---

### Job 2: check-image

```yaml
check-image:
  name: Verify Docker Hub Image
  runs-on: ubuntu-latest
  outputs:
    image_tag: ${{ steps.read-tag.outputs.tag }}
  steps:
    - uses: actions/checkout@v4
    - name: Read image tag from hf-space/Dockerfile
      id: read-tag
      run: |
        TAG=$(grep '^FROM' hf-space/Dockerfile | awk -F: '{print $2}')
        echo "tag=$TAG" >> $GITHUB_OUTPUT
    - name: Check image exists on Docker Hub
      run: docker manifest inspect ${{ env.DOCKER_REPO }}:${{ steps.read-tag.outputs.tag }}
```

**What it does:**
1. Reads the `hf-space/Dockerfile` and extracts the image tag (e.g., `v1` or `v2`) using `grep` and `awk`
2. Calls Docker Hub's API via `docker manifest inspect` to confirm that exact image tag exists
3. If the image does not exist on Docker Hub, the pipeline fails here — protecting against deploying a broken reference

**Why this matters:**
If a developer changes `hf-space/Dockerfile` to reference `v3` but forgets to build and push the `v3` image to Docker Hub, this job catches it before deployment even starts.

Jobs 1 and 2 run **in parallel** (neither has a `needs` dependency on the other), saving time.

---

### Job 3: deploy

```yaml
deploy:
  name: Deploy to Hugging Face Spaces
  runs-on: ubuntu-latest
  needs: [validate, check-image]
  steps:
    - uses: actions/checkout@v4
    - name: Upload to Hugging Face Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
        HF_USERNAME: ${{ secrets.HF_USERNAME }}
        HF_SPACE_NAME: ${{ secrets.HF_SPACE_NAME }}
      run: |
        pip install huggingface_hub -q
        python -c "
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path='hf-space/',
            repo_id='$HF_USERNAME/$HF_SPACE_NAME',
            repo_type='space',
            token='$HF_TOKEN'
        )
        print('Deployed successfully to HF Space')
        "
```

**What it does:**
Only runs if both Job 1 and Job 2 passed (`needs: [validate, check-image]`). Installs the `huggingface_hub` Python library and uses its `HfApi.upload_folder()` method to push the `hf-space/` folder to the Hugging Face Space repository. Hugging Face detects the new Dockerfile, pulls the referenced image from Docker Hub, and starts running it.

**`${{ secrets.HF_TOKEN }}`** — reads the Hugging Face API token from GitHub Secrets (never stored in code).

---

### Job 4: health-check

```yaml
health-check:
  name: Post-Deploy Health Check
  runs-on: ubuntu-latest
  needs: deploy
  steps:
    - name: Wait for Space to rebuild
      run: sleep 90
    - name: Call health endpoint and confirm version
      env:
        HF_USERNAME: ${{ secrets.HF_USERNAME }}
        HF_SPACE_NAME: ${{ secrets.HF_SPACE_NAME }}
      run: |
        curl --fail \
             --retry 5 \
             --retry-delay 30 \
             --retry-connrefused \
             https://$HF_USERNAME-$HF_SPACE_NAME.hf.space/health
```

**What it does:**
Waits 90 seconds for Hugging Face Spaces to pull the image and start the container, then calls the `/health` endpoint. `--retry 5 --retry-delay 30` means it tries up to 5 times, waiting 30 seconds between each attempt (up to 2.5 additional minutes). If the endpoint never responds, the pipeline fails — alerting the team that the deployment succeeded but the service is not healthy.

**The full job dependency chain:**
```
validate ──┐
           ├──► deploy ──► health-check
check-image┘
```

---

## 11. Hugging Face Spaces — Where the Model Runs

**What is Hugging Face Spaces?**
Hugging Face Spaces is a free hosting platform designed specifically for machine learning applications. It supports running apps built with Gradio, Streamlit, or Docker containers.

**What type of Space is this project using?**
This project uses a **Docker Space**. When you create a Docker Space, Hugging Face gives you a git repository. You push a Dockerfile to that repository. Hugging Face builds and runs whatever that Dockerfile defines.

**How is the Space structured?**
```
HF Space repository (git)
└── Dockerfile   ← This is what we push via the pipeline
                    FROM 03sarath/distilbert-sentiment:v1
                    EXPOSE 7860
```

When HF Spaces sees a new commit to this repository, it:
1. Reads the Dockerfile
2. Pulls `03sarath/distilbert-sentiment:v1` from Docker Hub
3. Runs the container
4. Exposes it at `https://03sarath-distilbert-sentiment.hf.space`

**Is this running as a container?**
Yes. Hugging Face Spaces runs Docker containers internally. Your entire application (FastAPI server + DistilBERT model) runs inside a container on Hugging Face's infrastructure.

**Live URL format:**
```
https://{hf-username}-{space-name}.hf.space
```
For this project: `https://03sarath-distilbert-sentiment.hf.space`

**Available endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns service status and version |
| `/predict` | POST | Accepts text, returns sentiment |
| `/docs` | GET | Auto-generated API documentation |

---

## 12. Docker Hub — Where the Image Lives

**What is Docker Hub?**
Docker Hub is a public registry for storing and sharing Docker images. It is to Docker images what GitHub is to code.

**Image naming convention:**
```
03sarath/distilbert-sentiment:v1
│         │                   │
│         │                   └── Tag (version)
│         └── Repository name
└── Docker Hub username
```

**Tags used in this project:**
| Tag | Contents | When used |
|---|---|---|
| `latest` | Same as v1, initial build | During development |
| `v1` | Original version, health returns `{"status": "ok"}` | First deployment |
| `v2` | Updated version, health returns `{"status": "ok", "version": "v2"}` + predict includes `model_version` | Second deployment |

**How was the image built?**
Manually, once, on a developer's machine:
```bash
docker build -t 03sarath/distilbert-sentiment:v1 ./app
docker push 03sarath/distilbert-sentiment:v1
```

The pipeline never builds the image. It only verifies the image exists and references it in the deployment.

---

## 13. GPU or CPU? What Hardware Is Used?

**Short answer: CPU only, no GPU.**

**On Hugging Face Spaces (Free tier):**
- Runs on CPU
- 2 vCPUs, 16GB RAM
- No GPU

**Does DistilBERT need a GPU?**
No. DistilBERT is small enough to run on CPU for inference (making predictions). It is slower than GPU inference but perfectly adequate for demonstration and low-to-medium traffic use cases.

**How long does a prediction take on CPU?**
Approximately 50–200ms per request — fast enough for a REST API.

**If you needed GPU:**
Hugging Face Spaces offers paid GPU tiers (T4, A10G) that can be configured in Space settings. The code would not need to change — PyTorch automatically uses CUDA if a GPU is available.

**What about the Docker image — does it include CUDA?**
The image was built with `python:3.11-slim` as the base, which has no CUDA support. The PyTorch version installed (`torch==2.3.0`) includes CUDA libraries within the package itself (they are bundled), but they are only activated when a compatible NVIDIA GPU is present. On CPU-only machines, PyTorch silently falls back to CPU execution.

---

## 14. Model Versioning — v1 vs v2

This project demonstrates how to manage model versions through the CI/CD pipeline without any manual server intervention.

**What changed between v1 and v2:**

| | v1 | v2 |
|---|---|---|
| `MODEL_VERSION` constant | Not present | `"v2"` |
| `/health` response | `{"status": "ok"}` | `{"status": "ok", "version": "v2"}` |
| `/predict` response | `{"label": "...", "score": ...}` | `{"label": "...", "score": ..., "model_version": "v2"}` |
| `PredictResponse` schema | `label`, `score` | `label`, `score`, `model_version` |

**How to deploy v2:**
Change one line in `hf-space/Dockerfile`:
```dockerfile
FROM 03sarath/distilbert-sentiment:v2
```
Commit and push. The pipeline handles the rest.

**How to roll back to v1:**
Change the line back:
```dockerfile
FROM 03sarath/distilbert-sentiment:v1
```
Commit and push. Pipeline deploys v1 again.

**Why is this powerful?**
Every version is an immutable Docker image stored permanently on Docker Hub. You can deploy any version at any time just by changing a tag. No rebuilding, no data loss, no downtime beyond the container restart time.

---

## 15. GitHub Secrets — How Credentials Are Managed

Three secrets are stored in GitHub (repo → Settings → Secrets → Actions):

| Secret | Value | Used in |
|---|---|---|
| `HF_TOKEN` | Hugging Face write-access token | deploy job — authenticates the upload |
| `HF_USERNAME` | Hugging Face username | deploy job, health-check job |
| `HF_SPACE_NAME` | Name of the HF Space | deploy job, health-check job |

**Why not put these directly in the workflow file?**
If credentials are in the code, anyone with access to the repository can see them. GitHub Secrets are encrypted and are only injected into the pipeline at runtime — they never appear in logs or code. In the workflow logs, they appear as `***`.

**How they are accessed in the workflow:**
```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

---

## 16. End-to-End Flow: What Happens When You Push Code

Here is the complete sequence of events from `git push` to a live model:

```
1. Developer changes hf-space/Dockerfile (e.g., v1 → v2)
2. git commit + git push origin main
3. GitHub detects push to main branch
4. GitHub Actions starts the workflow

5. JOB 1 (validate) — runs in parallel with Job 2
   - Spins up Ubuntu VM
   - Downloads repo code
   - Installs pytest, fastapi, httpx, pydantic
   - Runs 3 tests against mocked app
   - All pass → Job 1 complete

6. JOB 2 (check-image) — runs in parallel with Job 1
   - Spins up Ubuntu VM
   - Downloads repo code
   - Reads hf-space/Dockerfile → extracts tag "v2"
   - Calls Docker Hub API → confirms 03sarath/distilbert-sentiment:v2 exists
   - Job 2 complete

7. JOB 3 (deploy) — starts only after Job 1 AND Job 2 pass
   - Spins up Ubuntu VM
   - Downloads repo code
   - Installs huggingface_hub library
   - Calls HfApi.upload_folder() with HF credentials from secrets
   - Uploads hf-space/Dockerfile to HF Space git repo
   - HF Spaces detects the new commit
   - HF Spaces pulls 03sarath/distilbert-sentiment:v2 from Docker Hub
   - HF Spaces starts the container
   - Job 3 complete

8. JOB 4 (health-check) — starts after Job 3
   - Waits 90 seconds for container to start
   - Calls GET https://03sarath-distilbert-sentiment.hf.space/health
   - Retries up to 5 times if not ready
   - Gets 200 OK → pipeline succeeds
   - If never responds → pipeline fails, team is alerted

9. Model is live at:
   https://03sarath-distilbert-sentiment.hf.space/predict
```

**Total time from push to live:** approximately 4–5 minutes.

---

*This documentation covers the complete project as built for the MLOps Specialization Course. The same CI/CD pattern applies to any machine learning model, not just DistilBERT — the pipeline structure remains identical whether you are deploying a computer vision model, a recommender system, or a large language model.*
