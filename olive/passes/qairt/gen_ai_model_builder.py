# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from copy import deepcopy
from pathlib import Path

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler, QairtModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtGenAIModelBuilder(Pass):
    """Create a QAIRT GenAI Model applying various GenAI model transformations.

    Uses QAIRT GenAIModelBuilder Python API from the QAIRT SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "backend": PassConfigParam(
                type_=str,
                default_value="CPU",
                description="Target accelerator to prepare for. Accepted values are 'CPU' and 'HTP'.",
            ),
            # TODO - Should improve this API perhaps
            "soc_details": PassConfigParam(type_=str, default_value=None, description=""),
            # TODO - Could optionally add cache_dir, but is there a use case for this? Olive has caching between passes
            # TODO - Could add option to toggle weight sharing? What is use case not to do this?
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtModelHandler:
        # Attempt to import QAIRT Python API - if not present, something is probably wrong with user setup
        try:
            import qairt
        except ImportError as e:
            # TODO - Should probably give better message here
            raise e

        # Validate QAIRT BE type
        from qairt.api.configs.common import BackendType

        qairt_backend = config.backend
        if qairt_backend not in BackendType:
            raise ValueError(
                f"{qairt_backend} is not a valid QAIRT backend type. Valid backends: {[backend for backend in BackendType]}"
            )

        # TODO - Verify this is correct validation
        if qairt_backend != BackendType.HTP and config.soc_details is not None:
            raise ValueError("soc_details is not supported for backends other than HTP")

        # Import relevant GenAI APIs
        from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory

        # TODO - unsure if we want to rely on underlying cache behavior or GenAIBuilder? we can use Olive's
        gen_ai_builder = GenAIBuilderFactory.create(Path(model.model_path), config.backend)
        # TODO - Only one SoC detail is allowed right now but I think we should let GenAIBuilder fail w/ their validation for this
        gen_ai_builder.set_targets([config.soc_details])
        gen_ai_container = gen_ai_builder.build()
        gen_ai_container.save(output_model_path, exist_ok=True)

        # TODO - May need to add QAIRT-specific model attributes - e.g. soc details? features used? etc?
        #       Maybe the names/paths of context binaries and what is weight shared?
        model_attributes = deepcopy(model.model_attributes)
        return QairtModelHandler(model_path=output_model_path, model_attributes=model_attributes)
