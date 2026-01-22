# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import numbers
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, ClassVar, Union

import onnx.helper as helper
from onnx import checker, TensorProto, save

from olive.common.utils import hardlink_copy_dir, hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModelHandler, QairtModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtEncapsulation(Pass):
    """Encapsulates Qairt models with onnx context nodes."""

    qairt_to_onnx_dtype: ClassVar[dict] = {
        "f32": TensorProto.FLOAT,
        "float32": TensorProto.FLOAT,
        "f64": TensorProto.DOUBLE,
        "float64": TensorProto.DOUBLE,
        "f16": TensorProto.FLOAT16,
        "bf16": TensorProto.BFLOAT16,
        "i8": TensorProto.INT8,
        "int8_t": TensorProto.INT8,
        "i16": TensorProto.INT16,
        "int16_t": TensorProto.INT16,
        "i32": TensorProto.INT32,
        "int32_t": TensorProto.INT32,
        "i64": TensorProto.INT64,
        "int64_t": TensorProto.INT64,
        "u8": TensorProto.UINT8,
        "uint8_t": TensorProto.UINT8,
        "u16": TensorProto.UINT16,
        "uint16_t": TensorProto.UINT16,
        "u32": TensorProto.UINT32,
        "uint32_t": TensorProto.UINT32,
        "u64": TensorProto.UINT64,
        "uint64_t": TensorProto.UINT64,
        "bool": TensorProto.BOOL,
        "boolean": TensorProto.BOOL,
    }

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "log_level": PassConfigParam(
                type_=str,
                default_value=None,
                description="Log level to be used within underlying QAIRT components."
                "Valid values: DEBUG, INFO, WARN, ERROR.",
            ),
            "run_checker": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Runs the onnx checker on the model before it is encapsulated."
            ),
            "opset_imports": PassConfigParam(
                type_=list,
                default_value=[
                    ["com.microsoft", 1],
                ],
                required=False,
                description="Opset name and version to be added in the generated context model",
            ),
        }

    def _run_for_config(
        self,
        model: Union[QairtModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            from qairt.gen_ai_api.containers.llm_container import LLMContainer
            from qairt.modules.genie_execution.genie_config import ExportFormat
        except ImportError:
            raise ImportError("Please install olive-ai[qairt] to use QAIRT models.")
        

        container: LLMContainer = LLMContainer.load(model.model_path)

        # NEED TO EXTRACT METADATA FROM CONTAINER FOR ONNX WRAPPING SCRIPT
        # THIS IS SOMEWHAT COMPLICATED CAUSE IT DEPENDS ON THE NODE TYPE WE ARE USING FOR THIS MODEL, SHOULD BE ORT INPUTS

        # Input/Ouptut metadata
        container.inputs = [("dummy_input", TensorProto.INT32, [-1, -1, -1, -1])]
        container.outputs = [("dummy_output", TensorProto.INT32, [-1, -1, -1, -1])]
        
        input_info = {input[0]: (input[1], input[2]) for input in container.inputs}

        output_info = {output[0]: (output[1], output[2]) for output in container.outputs}

        # Input/Output tensor helpers
        inputs = []
        for (name, datatype, shape) in container.inputs:
            inputs.append(helper.make_tensor_value_info(name, datatype, shape))

        outputs = []
        for (name, datatype, shape) in container.outputs:
            outputs.append(helper.make_tensor_value_info(name, datatype, shape))

        # TODO - Should maybe separate this to a helper function so that different export formats can have their own functions
        # Export the container 
        container.export(output_model_path, export_format=ExportFormat.LM_EXECUTOR)  # Expect no binaries/libs but exported model

        context_node = helper.make_node(
            "EPContext",
            inputs=[name for name, _ in input_info.items()],
            outputs=[name for name, _ in output_info.items()],
            name="ContextNode",
            domain="com.microsoft",
        )

        context_node.attribute.extend([helper.make_attribute("ep_context_type", "zip")])
        context_node.attribute.extend([helper.make_attribute("ep_zip_context", "model.zip")])
        context_node.attribute.extend([helper.make_attribute("source", "QAIRTExport")])

        # Create the ONNX Graph
        graph_def = helper.make_graph(nodes=[context_node], name="EP_Context_Model", inputs=inputs, outputs=outputs)
        op_imports = [helper.make_opsetid(i[0], i[1]) for i in config.opset_imports]

        # Define the model with an Execution Provider (EP) Context
        model_def = helper.make_model(graph_def, opset_imports=op_imports)
        model_def.ir_version = 10

        if config.run_checker:
            checker.check_model(model_def)

        # Save the model
        # TODO - Need to derive a better name here from LLMContainer
        model_name = "model"
        context_model_output = f"{model_name}.onnx"
        context_model_output_dir = Path(output_model_path) / (context_model_output)

        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)

        save(model_def, context_model_output_dir)

        # NEED TO WRAP IN ONNX MODEL USING INFO + SCRIPT IN ONNX
        return ONNXModelHandler(model_path=output_model_path)
