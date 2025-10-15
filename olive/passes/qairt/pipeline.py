# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import tempfile
from typing import Union

from olive.common.hf.wrapper import ModelWrapper
from olive.hardware import AcceleratorSpec
from olive.model import CompositeModelHandler, HfModelHandler, QairtModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.train_utils import load_hf_base_model

logger = logging.getLogger(__name__)


class QairtPipeline(Pass):
    """Create a QAIRT E2E Pipeline which peforms various QAIRT-specific processing on a model.

    Uses QAIRT E2E Pipeline Python API from the QAIRT SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "backend": PassConfigParam(
                type_=str,
                default_value="CPU",
                description="Target accelerator to prepare for. Accepted values are 'CPU' and 'HTP'.",
            ),
            "soc_details": PassConfigParam(type_=str, default_value=None, description=""),
        }

    def _run_for_config(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[CompositeModelHandler, QairtModelHandler]:
        if not isinstance(model, HfModelHandler):
            raise NotImplementedError("Handlers aside from HfModelHandler are unsupported.")

        # Attempt to import QAIRT, if found via qairt-dev this will attempt to install all necessary dependencies
        try:
            import qairt  # noqa: F401  # pylint: disable=unused-import
        except ImportError:
            raise ImportError("Failed 'import qairt'. Please ensure that qairt-dev has been installed successfully.")

        logger.info("Successfully imported QAIRT and installed all dependencies.")

        # QAIRT Pipeline API imports
        try:
            from pipeline import execute
            from pipeline.config import PipelineGenAIConfig
        except ImportError:
            raise ImportError("Failed to import QAIRT Pipeline API")

        logger.info("Successfully loaded QAIRT Pipeline API.")

        with tempfile.TemporaryDirectory(prefix="olive_tmp") as temp_dir:
            logger.info("Converting HF model to PyTorch model...")
            pytorch_model = load_hf_base_model(model)

            logger.debug("Saving model to temporary directory...")
            model_wrapper = ModelWrapper.from_model(pytorch_model)
            model_wrapper.save_model(temp_dir)

            pipeline_config = PipelineGenAIConfig(model_path=temp_dir, backend=config.backend)

            execute(pipeline_config, output_path=output_model_path, stages=["source_transformations", "genai_builder"])

        return QairtModelHandler(model_path=output_model_path)
