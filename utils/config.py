from pathlib import Path
from types import SimpleNamespace

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_bucket_dir(paths_cfg, bucket):
    if bucket == "latest_experiment":
        return paths_cfg["latest_experiment_dir"]
    if bucket == "baseline":
        return paths_cfg["baseline_dir"]
    raise ValueError(f"Unsupported output bucket: {bucket}")


def _build_train_configs(raw):
    paths_cfg = raw["paths"]
    dataset_cfg = raw["dataset"]
    split_cfg = dataset_cfg["split"]
    train_cfg = raw["train"]
    model_cfg = raw["model"]

    model_name = train_cfg["model_name"]
    output_bucket = train_cfg["output_bucket"]
    base_dir = _resolve_bucket_dir(paths_cfg, output_bucket)

    return SimpleNamespace(
        model_name=model_name,
        output_bucket=output_bucket,
        base_dir=base_dir,
        model_dir=base_dir,
        log_dir=base_dir,
        data_dir=paths_cfg["data_dir"],
        pretrained_dir=paths_cfg["pretrained_dir"],
        results_dir=paths_cfg["results_dir"],
        analysis_dir=paths_cfg["analysis_dir"],
        latest_experiment_dir=paths_cfg["latest_experiment_dir"],
        baseline_dir=paths_cfg["baseline_dir"],
        log_filename=f"{base_dir}{model_name}_log.txt",
        model_filename=f"{base_dir}{model_name}.pt",
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
    results_dir = _resolve_bucket_dir(paths_cfg, output_bucket)
    checkpoint_path = test_cfg.get("checkpoint_path", "")
    if checkpoint_path:
        model_filename = checkpoint_path
    else:
        model_filename = train_configs.model_filename

    return SimpleNamespace(
        db=test_cfg["db"],
        output_bucket=output_bucket,
        db_filename=train_configs.main_db,
        model_name=train_configs.model_name,
        results_dir=results_dir,
        model_filename=model_filename,
        analysis_dir=paths_cfg["analysis_dir"],
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
