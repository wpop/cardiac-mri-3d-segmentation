from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    """Store output locations used by dataset inspection."""

    output_dir: Path
    report_filename: str
    summary_filename: str
    visualizations_dirname: str

    def __post_init__(self) -> None:
        """Validate output paths and generated artifact names."""
        if not str(self.output_dir).strip():
            raise ValueError("Inspection output directory must not be empty.")

        if not self.report_filename.endswith(".json"):
            raise ValueError("Inspection report filename must use the .json extension.")

        if not self.summary_filename.endswith(".csv"):
            raise ValueError("Inspection summary filename must use the .csv extension.")

        if not self.visualizations_dirname.strip():
            raise ValueError("Visualizations directory name must not be empty.")

        if Path(self.visualizations_dirname).is_absolute():
            raise ValueError("Visualizations directory name must be relative.")

    @property
    def report_path(self) -> Path:
        """Return the generated JSON report path."""
        return self.output_dir / self.report_filename

    @property
    def summary_path(self) -> Path:
        """Return the generated CSV summary path."""
        return self.output_dir / self.summary_filename

    @property
    def visualizations_dir(self) -> Path:
        """Return the generated visualization directory."""
        return self.output_dir / self.visualizations_dirname
