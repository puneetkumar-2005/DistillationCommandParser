# Cloud Computing (CS3081) mini project
#   make train            train the student on all 1,500 records (training/config.yaml)
#   make train SUBSET=50  train on the first 50 records only, to verify training runs
SUBSET ?= 0

.PHONY: train

train:
	python3 train/train.py --config training/config.yaml --subset $(SUBSET)
