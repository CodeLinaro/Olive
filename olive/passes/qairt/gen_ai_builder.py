# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path

from olive.hardware import AcceleratorSpec
from olive.model import QairtContainerModelHandler, QairtPreparedModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtGenAIBuilder(Pass):
    """Create a QairtContainerModelHandler from a QairtPreparedModelHandler.

    Applies various QAIRT-specific optimizations depending on model architecture,
    converts them to DLC, and compiles a context binary compatible with the specified SoC.

    Uses QAIRT GenAIBuilder Python API from the QAIRT SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "backend": PassConfigParam(
                type_=str,
                default_value="CPU",
                description="Target accelerator to prepare the model for. Accepted values are 'CPU' and 'HTP'.",
            ),
            "soc_details": PassConfigParam(
                type_=str,
                default_value=None,
                description="Device specification to use for compilation. Can be specified"
                " as a spec string in the form 'chipset:value;dsp_arch:value;soc_model:value|...'."
                " This option will be ignored if any device custom configurations are set."
                # TODO - Add some examples
                "e.g. ",
            ),
            "cache_dir": PassConfigParam(type_=str, default_value=None, description=""),
            "log_level": PassConfigParam(type_=str, default_value=None, description=""),
        }

    def _run_for_config(
        self,
        model: QairtPreparedModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtContainerModelHandler:
        # Attempt to import QAIRT Python API - if not present, something is probably wrong with user setup
        try:
            from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
        except ImportError as e:
            # TODO - Should probably give better message here
            raise e

        # TODO - Should file ticket for someone to formally add this to GenAIBuilder API
        if config.log_level:
            os.environ["QAIRT_LOG_LEVEL"] = config.log_level

        if not config.cache_dir:
            logger.warning(
                "QAIRT GenAIBuilder cache directory not set. Using this will decrease future preparation time."
            )

        # TODO - unsure if we want to rely on underlying cache behavior or GenAIBuilder? we can use Olive's
        gen_ai_builder = GenAIBuilderFactory.create(
            pretrained_model_path=Path(model.model_path), backend_type=config.backend, cache_root=config.cache_dir
        )
        gen_ai_builder.set_targets([config.soc_details])
        gen_ai_container = gen_ai_builder.build()
        gen_ai_container.save(output_model_path, exist_ok=True)

        # TODO - Do we need any model attributes aside from this?
        return QairtContainerModelHandler(model_path=output_model_path)
