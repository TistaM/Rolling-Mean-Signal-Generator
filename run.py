import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def write_metrics(output_path: str, metrics: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure. Config must be a YAML dictionary.")

    required_fields = ["seed", "window", "version"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer.")

    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValueError("Config field 'window' must be a positive integer.")

    if not isinstance(config["version"], str) or not config["version"]:
        raise ValueError("Config field 'version' must be a non-empty string.")

    np.random.seed(config["seed"])

    logging.info(
        "Config loaded and validated: seed=%s, window=%s, version=%s",
        config["seed"],
        config["window"],
        config["version"],
    )

    return config


def load_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or has no columns.")
    except pd.errors.ParserError as error:
        raise ValueError(f"Invalid CSV format: {error}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "close" not in df.columns:
        raise ValueError("Missing required column: close")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if df["close"].isna().all():
        raise ValueError("Column 'close' contains no valid numeric values.")

    logging.info("Dataset loaded successfully. Rows loaded: %s", len(df))

    return df


def process_data(df: pd.DataFrame, window: int) -> pd.DataFrame:
    logging.info("Computing rolling mean using window=%s", window)

    df = df.copy()

    # First window-1 rows will have NaN rolling_mean.
    # For signal generation, NaN comparison returns False, so signal becomes 0.
    df["rolling_mean"] = df["close"].rolling(window=window).mean()

    logging.info("Generating binary signal from close vs rolling_mean")

    df["signal"] = np.where(df["close"] > df["rolling_mean"], 1, 0)

    return df


def run_job(input_path: str, config_path: str, output_path: str) -> dict:
    start_time = time.perf_counter()

    config = load_config(config_path)
    df = load_dataset(input_path)

    processed_df = process_data(df, config["window"])

    rows_processed = int(len(processed_df))
    signal_rate = float(processed_df["signal"].mean())

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    metrics = {
        "version": config["version"],
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": config["seed"],
        "status": "success",
    }

    logging.info("Metrics summary: %s", metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal MLOps-style batch job for rolling mean signal generation."
    )

    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON file.")
    parser.add_argument("--log-file", required=True, help="Path to output log file.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    setup_logging(args.log_file)

    logging.info("Job started")

    try:
        metrics = run_job(
            input_path=args.input,
            config_path=args.config,
            output_path=args.output,
        )

        write_metrics(args.output, metrics)

        logging.info("Metrics written to %s", args.output)
        logging.info("Job ended with status: success")

        print(json.dumps(metrics, indent=2))

        return 0

    except Exception as error:
        logging.exception("Job failed due to error")

        error_metrics = {
            "version": "unknown",
            "status": "error",
            "error_message": str(error),
        }

        try:
            if Path(args.config).exists():
                with open(args.config, "r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                if isinstance(config, dict) and "version" in config:
                    error_metrics["version"] = config["version"]
        except Exception:
            pass

        write_metrics(args.output, error_metrics)

        logging.info("Error metrics written to %s", args.output)
        logging.info("Job ended with status: error")

        print(json.dumps(error_metrics, indent=2))

        return 1


if __name__ == "__main__":
    sys.exit(main())