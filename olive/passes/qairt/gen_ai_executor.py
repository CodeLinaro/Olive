# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
from copy import deepcopy
from pathlib import Path

from olive.hardware import AcceleratorSpec
from olive.model import QairtContainerModelHandler, QairtPreparedModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtGenAIExecutor(Pass):
    """Create a QairtContainerModelHandler from a QairtPreparedModelHandler.

    Applies various QAIRT-specific optimizations depending on model architecture,
    converts them to DLC, and compiles a context binary compatible with the specified SoC.

    Uses QAIRT GenAIBuilder Python API from the QAIRT SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "prompt": PassConfigParam(
                type_=str, default_value=None, description="Prompt to pass to the model for evaluation."
            ),
            "prompt_file": PassConfigParam(
                type_=str,
                default_value=None,
                description="File containing a prompt to pass to the model for evaluation.",
            ),
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

        # TODO - May need to add QAIRT-specific model attributes - e.g. soc details? features used? etc?
        #       Maybe the names/paths of context binaries and what is weight shared?
        model_attributes = deepcopy(model.model_attributes)
        print(model_attributes)
        return QairtContainerModelHandler(model_path=output_model_path, model_attributes=model_attributes)
