import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
LATEST_RUN_FILENAME = "_latest_run.txt"


def _as_dir_str(path: Path) -> str:
    return str(path) + os.sep


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_bucket_dir(paths_cfg, bucket):
    if bucket == "latest_experiment":
        return Path(paths_cfg["latest_experiment_dir"])
    if bucket == "baseline":
        return Path(paths_cfg["baseline_dir"])
    raise ValueError(f"Unsupported output bucket: {bucket}")


def _resolve_run_id(run_id_value, bucket_root: Path):
    if run_id_value == "auto":
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if run_id_value == "latest":
        latest_file = bucket_root / LATEST_RUN_FILENAME
        if not latest_file.exists():
            raise FileNotFoundError(
                f"Cannot resolve run_id='latest' because pointer file does not exist: {latest_file}"
            )
        return latest_file.read_text(encoding="utf-8").strip()
    return run_id_value


def _build_train_configs(raw):
    paths_cfg = raw["paths"]
    dataset_cfg = raw["dataset"]
    split_cfg = dataset_cfg["split"]
    train_cfg = raw["train"]
    model_cfg = raw["model"]

    model_name = train_cfg["model_name"]
    output_bucket = train_cfg["output_bucket"]
    bucket_root = _resolve_bucket_dir(paths_cfg, output_bucket)
    run_id = _resolve_run_id(train_cfg.get("run_id", "auto"), bucket_root)
    base_dir = bucket_root / run_id

    return SimpleNamespace(
        model_name=model_name,
        output_bucket=output_bucket,
        run_id=run_id,
        bucket_root_dir=_as_dir_str(bucket_root),
        base_dir=_as_dir_str(base_dir),
        model_dir=_as_dir_str(base_dir),
        log_dir=_as_dir_str(base_dir),
        data_dir=paths_cfg["data_dir"],
        pretrained_dir=paths_cfg["pretrained_dir"],
        results_dir=paths_cfg["results_dir"],
        analysis_dir=paths_cfg["analysis_dir"],
        latest_experiment_dir=paths_cfg["latest_experiment_dir"],
        baseline_dir=paths_cfg["baseline_dir"],
        latest_run_file=str(bucket_root / LATEST_RUN_FILENAME),
        log_filename=str(base_dir / f"{model_name}_log.txt"),
        model_filename=str(base_dir / f"{model_name}.pt"),
        run_metadata_filename=str(base_dir / "run_metadata.json"),
        main_db=dataset_cfg["main_db"],
        total_users=split_cfg["train_end"],
        num_training_subjects=split_cfg["train_start"],
        num_validation_subjects=split_cfg["val_end"],
        sequence_length=model_cfg["sequence_length"],
        batch_size_train=train_cfg["batch_size_train"],
        batch_size_val=train_cfg["batch_size_val"],
        dimensionality=model_cfg["dimensionality"],
        output_dim=model_cfg["output_dim"],
        batches_per_epoch=train_cfg["batches_per_epoch"],
        val_batches_per_epoch=train_cfg["val_batches_per_epoch"],
        epochs=train_cfg["epochs"],
        decimals=train_cfg["decimals"],
        num_workers=train_cfg["num_workers"],
        lr=train_cfg["lr"],
        betas=tuple(train_cfg["betas"]),
        K=model_cfg["K"],
        hlayers=model_cfg["hlayers"],
        hlayers_rec=model_cfg["hlayers_rec"],
        hlayers_pos=model_cfg["hlayers_pos"],
        hheads=model_cfg["hheads"],
        vlayers=model_cfg["vlayers"],
        vheads=model_cfg["vheads"],
    )


def _build_test_configs(raw, train_configs):
    test_cfg = raw["test"]
    split_cfg = raw["dataset"]["split"]
    paths_cfg = raw["paths"]

    output_bucket = test_cfg["output_bucket"]
    bucket_root = _resolve_bucket_dir(paths_cfg, output_bucket)
    run_id = _resolve_run_id(test_cfg.get("run_id", "latest"), bucket_root)
    results_dir = bucket_root / run_id
    checkpoint_path = test_cfg.get("checkpoint_path", "")
    model_filename = checkpoint_path if checkpoint_path else str(results_dir / f"{train_configs.model_name}.pt")

    return SimpleNamespace(
        db=test_cfg["db"],
        output_bucket=output_bucket,
        run_id=run_id,
        bucket_root_dir=_as_dir_str(bucket_root),
        db_filename=train_configs.main_db,
        model_name=train_configs.model_name,
        results_dir=_as_dir_str(results_dir),
        model_filename=model_filename,
        analysis_dir=_as_dir_str(Path(paths_cfg["analysis_dir"]) / run_id),
        latest_run_file=str(bucket_root / LATEST_RUN_FILENAME),
        run_metadata_filename=str(results_dir / "evaluation_metadata.json"),
        num_test_subjects=split_cfg["test_end"] - split_cfg["test_start"],
        num_validation_subjects=split_cfg["val_end"],
        total_num_sessions=test_cfg["total_num_sessions"],
        enrolment_samples=test_cfg["enrolment_samples"],
        test_samples=test_cfg["test_samples"],
        impostor_test_samples=test_cfg["impostor_test_samples"],
    )


raw_configs = _load_config()
configs = _build_train_configs(raw_configs)
test_configs = _build_test_configs(raw_configs, configs)
