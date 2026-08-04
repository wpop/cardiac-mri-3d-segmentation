from pathlib import Path
from typing import ClassVar

from cardiac_segmentation.data.acdc_patient_info import AcdcPatientInfo


class AcdcInfoParser:
    """Parse and validate one real ACDC patient Info.cfg file."""

    _EXPECTED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "ED",
            "ES",
            "Group",
            "Height",
            "NbFrame",
            "Weight",
        }
    )

    def parse(self, info_path: Path) -> AcdcPatientInfo:
        """Read one Info.cfg file and return validated patient metadata."""
        resolved_path = info_path.expanduser().resolve(strict=False)

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"ACDC patient information file does not exist: {resolved_path}"
            )

        fields = self._read_fields(resolved_path)
        self._validate_keys(fields, resolved_path)

        return AcdcPatientInfo(
            patient_id=resolved_path.parent.name,
            ed_frame=self._parse_integer(fields, "ED", resolved_path),
            es_frame=self._parse_integer(fields, "ES", resolved_path),
            clinical_group=fields["Group"].strip(),
            height_cm=self._parse_float(fields, "Height", resolved_path),
            frame_count=self._parse_integer(fields, "NbFrame", resolved_path),
            weight_kg=self._parse_float(fields, "Weight", resolved_path),
        )

    def _read_fields(self, info_path: Path) -> dict[str, str]:
        """Read colon-separated fields while rejecting malformed or duplicate entries."""
        try:
            lines = info_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise OSError(
                f"Failed to read ACDC patient information file: {info_path}"
            ) from error

        fields: dict[str, str] = {}

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if ":" not in line:
                raise ValueError(
                    f"Malformed line {line_number} in {info_path}: {raw_line}"
                )

            key, value = line.split(":", maxsplit=1)
            normalized_key = key.strip()
            normalized_value = value.strip()

            if not normalized_key:
                raise ValueError(
                    f"Empty key at line {line_number} in {info_path}."
                )

            if not normalized_value:
                raise ValueError(
                    f"Empty value for key '{normalized_key}' in {info_path}."
                )

            if normalized_key in fields:
                raise ValueError(
                    f"Duplicate key '{normalized_key}' in {info_path}."
                )

            fields[normalized_key] = normalized_value

        return fields

    def _validate_keys(
        self,
        fields: dict[str, str],
        info_path: Path,
    ) -> None:
        """Ensure that Info.cfg contains exactly the verified ACDC metadata fields."""
        actual_keys = set(fields)
        missing_keys = self._EXPECTED_KEYS - actual_keys
        unknown_keys = actual_keys - self._EXPECTED_KEYS

        if missing_keys:
            formatted_keys = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"Missing required keys in {info_path}: {formatted_keys}"
            )

        if unknown_keys:
            formatted_keys = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"Unknown keys in {info_path}: {formatted_keys}"
            )

    def _parse_integer(
        self,
        fields: dict[str, str],
        key: str,
        info_path: Path,
    ) -> int:
        """Convert one required metadata field to an integer."""
        try:
            return int(fields[key])
        except ValueError as error:
            raise ValueError(
                f"Key '{key}' must contain an integer in {info_path}."
            ) from error

    def _parse_float(
        self,
        fields: dict[str, str],
        key: str,
        info_path: Path,
    ) -> float:
        """Convert one required metadata field to a floating-point value."""
        try:
            return float(fields[key])
        except ValueError as error:
            raise ValueError(
                f"Key '{key}' must contain a number in {info_path}."
            ) from error
