# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from olive.common.config_utils import ParamCategory
from olive.common.utils import hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, QairtPreparedModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtPreparation(Pass):
    """Prepare a HuggingFace model for QAIRT by running an external preparation script.

    This pass executes a Python script that performs quantization and other preparation
    steps to convert a HuggingFace model into a QAIRT-compatible format. The script
    receives configuration via a JSON file and produces a QairtPreparedModelHandler.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "script_path": PassConfigParam(
                type_=str,
                required=True,
                category=ParamCategory.PATH,
                description="Path to the Python script that performs QAIRT preparation. "
                "The script should accept a --config argument pointing to a JSON configuration file.",
            ),
            "script_config": PassConfigParam(
                type_=dict,
                required=False,
                default_value={},
                description="Configuration dictionary to pass to the preparation script. "
                "This will be merged with input_model_path and output_model_path in the JSON config file. "
                "Example: {'precision': 'int8', 'calibration_samples': 100, 'backend': 'HTP'}",
            )
        }

    def _run_for_config(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtPreparedModelHandler:
        """Execute the preparation script to convert HfModelHandler to QairtPreparedModelHandler.

        Args:
            model: Input HfModelHandler
            config: Pass configuration
            output_model_path: Path where the prepared model should be saved

        Returns:
            QairtPreparedModelHandler pointing to the prepared model

        Raises:
            ValueError: If input model is not HfModelHandler or script path is invalid
            RuntimeError: If script execution fails
        """
        # Validate input model type
        if not isinstance(model, HfModelHandler):
            raise ValueError(
                f"QairtPreparation requires HfModelHandler as input, got {type(model).__name__}"
            )

        # Resolve and validate script path
        script_path = Path(config.script_path).resolve()
        if not script_path.exists():
            raise ValueError(f"Preparation script not found at: {script_path}")
        if not script_path.suffix == ".py":
            raise ValueError(f"Script must be a Python file (.py), got: {script_path}")

        # Prepare configuration for the script
        script_config = {
            "input_model_path": str(Path(model.model_path).resolve()),
            "output_model_path": str(Path(output_model_path).resolve()),
        }
        
        # Merge user-provided config
        if config.script_config:
            logging.error("CONFIG SET ADDING CONFIGS")
            script_config.update(config.config)

        # Create temporary JSON config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="olive_qairt_prep_"
        ) as config_file:
            logging.error(script_config)
            json.dump(script_config, config_file, indent=2)
            config_file_path = config_file.name

        try:
            logger.info("Executing script %s", script_path)
            logger.debug("Script configuration: %s", script_config)

            # Execute the preparation script
            result = subprocess.run(
                ["python", str(script_path), "--config", config_file_path],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                check=False,
            )

            # Log script output
            if result.stdout:
                logger.debug("Script stdout:\n%s", result.stdout)
            if result.stderr:
                logger.debug("Script stderr:\n%s", result.stderr)

            # Check for errors
            if result.returncode != 0:
                error_msg = (
                    f"QAIRT preparation script failed with exit code {result.returncode}.\n"
                    f"Script: {script_path}\n"
                    f"Working directory: {script_path.parent}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )
                raise RuntimeError(error_msg)

            logger.info("QAIRT preparation script completed successfully")

        finally:
            # Clean up temporary config file
            try:
                Path(config_file_path).unlink()
            except Exception as e:
                logger.warning("Failed to delete temporary config file %s: %s", config_file_path, e)

        # Verify output exists
        output_path = Path(output_model_path)
        if not output_path.exists():
            raise RuntimeError(
                f"Script completed but output model not found at: {output_model_path}. "
                "The preparation script may not have created the expected output."
            )

        # Ensure config.json is present in output (copy from input if needed)
        source_config_path = Path(model.model_path) / "config.json"
        if source_config_path.exists():
            dest_config_path = output_path / "config.json" if output_path.is_dir() else output_path.parent / "config.json"
            if not dest_config_path.exists():
                logger.info("Copying config.json from input model to output")
                hardlink_copy_file(source_config_path, dest_config_path.parent, follow_symlinks=True)

        # TODO(team): Add validation of output model format to ensure it meets QAIRT requirements
        # For now, we trust the script produces valid output

        return QairtPreparedModelHandler(model_path=output_model_path)
