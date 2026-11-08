# Cloud Computing (CS3081) mini project
#   make train            train the student on all 1,500 records (training/config.yaml)
#   make train SUBSET=50  train on the first 50 records only, to verify training runs
SUBSET ?= 0

.PHONY: train

train:
	python3 train/train.py --config training/config.yaml --subset $(SUBSET)

# Task 6: reproducible up/down (recipes use > instead of Tab, so pasting is safe)
.RECIPEPREFIX = >
.ONESHELL:
SHELL := /bin/bash

CLUSTER := cloud-mini
RELEASE_URL := https://github.com/puneetkumar-2005/DistillationCommandParser/releases/download/student-v1/student-v1.zip

.PHONY: up down

up:
> set -e
> if k3d cluster list | grep -q "^$(CLUSTER) "; then k3d cluster start $(CLUSTER) || true; else k3d cluster create --config cluster/k3d.yaml; fi
> docker build -t parser-service:v1 service/
> k3d image import parser-service:v1 -c $(CLUSTER)
> kubectl apply -f cluster/seaweedfs.yaml
> kubectl wait --for=condition=available deployment -n data --all --timeout=180s
> curl -L -o /tmp/student-v1.zip $(RELEASE_URL)
> python3 -m zipfile -e /tmp/student-v1.zip /tmp/student-v1
> kubectl port-forward -n data svc/seaweedfs-s3 8333:8333 >/tmp/port-forward.log 2>&1 &
> PF_PID=$$!
> trap 'kill $$PF_PID 2>/dev/null || true' EXIT
> sleep 5
> export S3_ENDPOINT=http://localhost:8333
> python3 scripts/s3.py mb datasets models || true
> for f in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json train_config.json; do python3 scripts/s3.py upload models /tmp/student-v1/student-v1/$$f v1/$$f; done
> python3 scripts/s3.py upload models /tmp/student-v1.zip v1/student-v1.zip
> sha256sum /tmp/student-v1.zip | cut -d' ' -f1 > /tmp/student-v1/model.sha256
> python3 scripts/s3.py upload models /tmp/student-v1/model.sha256 v1/model.sha256
> python3 scripts/s3.py ls models v1/
> kill $$PF_PID
> kubectl apply -f app/namespace.yaml
> sleep 2
> kubectl apply -f app/
> kubectl wait --for=condition=available deployment/parser -n app --timeout=300s
> curl -s -X POST http://localhost:8080/parse -H 'Content-Type: application/json' -d '{"text": "set an alarm for tomorrow at seven thirty am"}'
> echo
> echo "stack is up: parser Ready, the curl above is the proof"

down:
> k3d cluster delete $(CLUSTER)
> rm -f /tmp/student-v1.zip
> rm -rf /tmp/student-v1
