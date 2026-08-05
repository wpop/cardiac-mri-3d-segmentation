from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.data.acdc_dataset_inspection_csv_writer import (
    AcdcDatasetInspectionCsvWriter,
)
from cardiac_segmentation.data.acdc_dataset_inspection_json_writer import (
    AcdcDatasetInspectionJsonWriter,
)
from cardiac_segmentation.data.acdc_dataset_inspection_report import (
    AcdcDatasetInspectionReport,
)
from cardiac_segmentation.data.acdc_dataset_inspector import (
    AcdcDatasetInspector,
)


class AcdcDatasetInspectionRunner:
    """Run complete ACDC inspection and persist its output artifacts."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the runner with validated application configuration."""
        self._config = config
        self._inspector = AcdcDatasetInspector(config)
        self._json_writer = AcdcDatasetInspectionJsonWriter()
        self._csv_writer = AcdcDatasetInspectionCsvWriter()

    def run(self) -> AcdcDatasetInspectionReport:
        """Inspect the dataset, write JSON and CSV, and return the report."""
        report = self._inspector.inspect()

        self._json_writer.write(
            report=report,
            output_path=self._config.inspection.report_path,
        )
        self._csv_writer.write(
            report=report,
            output_path=self._config.inspection.summary_path,
        )

        return report
