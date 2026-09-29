#!/bin/bash

set -e

echo "1. Preprocessing training data..."
.venv/Scripts/python.exe scripts/preprocess.py \
  --input "data/titanic/train (1).csv" \
  --output "data/titanic/train_processed.csv"

echo "2. Preprocessing test data..."
.venv/Scripts/python.exe scripts/preprocess.py \
  --input "data/titanic/test (1).csv" \
  --output "data/titanic/test_processed.csv"

echo "3. Creating training features..."
.venv/Scripts/python.exe scripts/featurize.py \
  --input "data/titanic/train_processed.csv" \
  --output "data/titanic/train_features.csv"

echo "4. Creating test features..."
.venv/Scripts/python.exe scripts/featurize.py \
  --input "data/titanic/test_processed.csv" \
  --output "data/titanic/test_features.csv"

echo "5. Training model..."
.venv/Scripts/python.exe scripts/train.py \
  --input "data/titanic/train_features.csv" \
  --output "models/model.pkl" \
  --eval-output "data/titanic/eval_features.csv"

echo "6. Evaluating model..."
.venv/Scripts/python.exe scripts/evaluate.py \
  --input "data/titanic/eval_features.csv" \
  --model "models/model.pkl" \
  --output "metrics/evaluation.json"

echo "7. Making predictions..."
.venv/Scripts/python.exe scripts/predict.py \
  --input "data/titanic/test_features.csv" \
  --model "models/model.pkl" \
  --output "predictions/predictions.csv"

echo "Pipeline completed successfully!"