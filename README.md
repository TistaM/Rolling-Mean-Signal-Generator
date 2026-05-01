#  MLOps Engineering - Rolling Mean Signal Generator

# Overview

This project implements a minimal MLOps-style batch pipeline in Python.

The goal is to simulate a simple trading signal workflow with a focus on:

Reproducibility (deterministic runs using config + seed)

Observability (structured logs and machine-readable metrics)

Deployment readiness (Dockerized, one-command execution)

The pipeline reads market data (OHLCV), computes a rolling mean, generates a signal, and outputs metrics and logs.

# Project Structure

mlops-task/
├── run.py
├── config.yaml
├── data.csv
├── requirements.txt
├── Dockerfile
├── README.md
├── metrics.json
├── metrics_error.json   # optional (error case example)
└── run.log

# Input Files

# config.yaml
    
    yaml
seed: 42
window: 5
version: "v1"

# data.csv

Contains OHLCV data (10,000 rows)

Only the close column is used for computation

# Processing Pipeline

The pipeline performs the following steps:

1. Config Loading & Validation

   Validates required fields: seed, window, version
   Sets deterministic seed

2. Dataset Loading & Validation

Checks:

    Missing file
    Invalid CSV format
    Empty dataset
    Missing close column

3. Rolling Mean Calculation

   Computes rolling mean on close using configured window
   First window-1 rows produce NaN (handled consistently)

4. Signal Generation

   signal = 1 if close > rolling_mean else 0

5. Metrics Calculation

   rows_processed
   signal_rate
   latency_ms

# How to Run Locally

# Step 1: Install dependencies

pip install -r requirements.txt

# Step 2: Run the pipeline

python run.py --input data.csv --config config.

yaml --output metrics.json --log-file run.log

# Docker Execution

# Build image

docker build -t mlops-task .

# Run container

docker run --rm mlops-task

# Output

# Success Output (metrics.json)

json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 34,
  "seed": 42,
  "status": "success"
}

# Error Output (metrics_error.json)

json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input file not found: wrong.csv"
}

Note: The metrics file is always written in both success and error cases.

# Logging (run.log)

The pipeline logs:

Job start timestamp
Config validation details
Dataset loading info
Processing steps
Metrics summary
Job completion status

# Key Features

Deterministic runs using seed
Robust input validation
Clear logging for observability
Machine-readable metrics output
Dockerized for reproducibility and portability

# Notes

No hardcoded paths are used
Works both locally and inside Docker
Designed to reflect real-world MLOps batch workflows

# Author

Tista Mukherjee

Submitted as part of the ML/MLOps Engineering Internship – Technical Assessment
