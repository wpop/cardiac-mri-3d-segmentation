from pathlib import Path

import pytest

from cardiac_segmentation.config import (
    SinglePatientOverfitConfig,
    SinglePatientOverfitConfigLoader,
)


def test_load_single_patient_overfit_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "overfit.yaml"
    checkpoint_path = tmp_path / "artifacts" / "single_patient.pt"
    config_path.write_text(
        "\n".join(
            (
                "single_patient_overfit:",
                "  patient_id: patient001",
                "  epoch_count: 4",
                "  batch_size: 2",
                "  num_workers: 0",
                "  pin_memory: false",
                "  random_seed: 42",
                "  base_channels: 2",
                "  learning_rate: 0.001",
                "  weight_decay: 0.00001",
                "  device: cpu",
                "  checkpoint_path: artifacts/single_patient.pt",
            )
        ),
        encoding="utf-8",
    )

    config = SinglePatientOverfitConfigLoader(
        project_root=tmp_path,
    ).load(Path("overfit.yaml"))

    assert config.patient_id == "patient001"
    assert config.epoch_count == 4
    assert config.batch_size == 2
    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.random_seed == 42
    assert config.base_channels == 2
    assert config.learning_rate == pytest.approx(0.001)
    assert config.weight_decay == pytest.approx(0.00001)
    assert config.device == "cpu"
    assert config.checkpoint_path == checkpoint_path.resolve(strict=False)


def test_single_patient_overfit_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "overfit.yaml"
    config_path.write_text(
        "\n".join(
            (
                "single_patient_overfit:",
                "  patient_id: patient001",
                "  epoch_count: 1",
                "  batch_size: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "  random_seed: 42",
                "  base_channels: 2",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  device: cpu",
                "  checkpoint_path: artifacts/single_patient.pt",
                "  scheduler: cosine",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys: scheduler"):
        SinglePatientOverfitConfigLoader(
            project_root=tmp_path,
        ).load(config_path)


def test_single_patient_overfit_loader_rejects_missing_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "overfit.yaml"
    config_path.write_text(
        "\n".join(
            (
                "single_patient_overfit:",
                "  patient_id: patient001",
                "  epoch_count: 1",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required keys"):
        SinglePatientOverfitConfigLoader(
            project_root=tmp_path,
        ).load(config_path)


def test_single_patient_overfit_config_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Patient identifier must not be empty"):
        SinglePatientOverfitConfig(
            patient_id=" ",
            epoch_count=1,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            random_seed=42,
            base_channels=2,
            learning_rate=0.001,
            weight_decay=0.0,
            device="cpu",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )

    with pytest.raises(ValueError, match="Epoch count must be a positive integer"):
        SinglePatientOverfitConfig(
            patient_id="patient001",
            epoch_count=True,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            random_seed=42,
            base_channels=2,
            learning_rate=0.001,
            weight_decay=0.0,
            device="cpu",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )

    with pytest.raises(ValueError, match="Learning rate must be finite"):
        SinglePatientOverfitConfig(
            patient_id="patient001",
            epoch_count=1,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            random_seed=42,
            base_channels=2,
            learning_rate=0.0,
            weight_decay=0.0,
            device="cpu",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )

    with pytest.raises(ValueError, match="Device must be exactly one of"):
        SinglePatientOverfitConfig(
            patient_id="patient001",
            epoch_count=1,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            random_seed=42,
            base_channels=2,
            learning_rate=0.001,
            weight_decay=0.0,
            device="mps",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )
