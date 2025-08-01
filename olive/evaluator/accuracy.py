# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import abstractmethod
from inspect import isfunction, signature
from typing import Any, Callable, ClassVar, Union

import torch
import torchmetrics
from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam
from olive.data.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)

class AccuracyBase(AutoConfigClass):
    registry: ClassVar[dict[str, type["AccuracyBase"]]] = {}
    metric_cls_map: ClassVar[dict[str, Union[torchmetrics.Metric, Callable]]] = {
        "accuracy_score": torchmetrics.Accuracy,
        "f1_score": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall,
        "auroc": torchmetrics.AUROC,
        "perplexity": torchmetrics.text.perplexity.Perplexity,
        "cosine_similarity":torchmetrics.functional.cosine_similarity,
        "mse":torchmetrics.functional.mean_squared_error,
        "psnr": torchmetrics.image.PeakSignalNoiseRatio,
        "sqnr": torchmetrics.functional.signal_noise_ratio,
        "l2_norm":torchmetrics.functional.mean_squared_error
    }

    def __init__(self, config: Union[ConfigBase, dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.resolve_kwargs()

    def resolve_kwargs(self):
        config_dict = self.config.dict()
        kwargs = config_dict.pop("kwargs", {})
        config_dict.update(kwargs or {})
        self.config_dict = config_dict

    @classmethod
    def _metric_config_from_torch_metrics(cls):
        metric_module = cls.metric_cls_map[cls.name]
        params = signature(metric_module).parameters
        # if the metrics is calculated by torchmetrics.functional, we should filter the label data out
        ignore_idx = 0
        if isfunction(metric_module):
            ignore_idx = 2
            logger.debug("Will ignore the first two params of torchmetrics.functional.")
        metric_config = {}
        for param, info in params.items():
            if ignore_idx > 0:
                ignore_idx -= 1
                continue
            annotation = info.annotation if info.annotation != info.empty else None
            default_value, required = (info.default, False) if info.default != info.empty else (None, True)
            if info.kind in (info.VAR_KEYWORD, info.VAR_POSITIONAL):
                required = False
            metric_config[param] = ConfigParam(type_=annotation, required=required, default_value=default_value)
        if "task" in metric_config:
            metric_config["task"].default_value = "binary"
            metric_config["task"].required = False
        return metric_config

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return cls._metric_config_from_torch_metrics()

    # @staticmethod
    # def prepare_tensors(preds, target, dtypes=torch.int):
    #     dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes, dtypes]
    #     assert len(dtypes) == 2, "dtypes should be a list or tuple with two elements."
    #     preds = torch.tensor(preds, dtype=dtypes[0]) if not isinstance(preds, torch.Tensor) else preds.to(dtypes[0])
    #     target = torch.tensor(target, dtype=dtypes[1]) if not isinstance(target, torch.Tensor) else target.to(dtypes[1])
    #     return preds, target
    
    @staticmethod
    def prepare_tensors(preds, target, dtypes=torch.int):
        dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes, dtypes]
        assert len(dtypes) == 2, "dtypes should be a list or tuple with two elements."
        if isinstance(preds, dict):
            preds = {
                k: torch.tensor(v, dtype=dtypes[0]) if not isinstance(v, torch.Tensor)
                else v.to(dtypes[0])
                for k, v in preds.items()
            }
        else:
            preds = (
                torch.tensor(preds, dtype=dtypes[0])
                if not isinstance(preds, torch.Tensor)
                else preds.to(dtypes[0])
            )
        if isinstance(target, dict):
            target = {
                k: torch.tensor(v, dtype=dtypes[1]) if not isinstance(v, torch.Tensor)
                else v.to(dtypes[1])
                for k, v in target.items()
            }
        else:
            target = (
                torch.tensor(target, dtype=dtypes[1])
                if not isinstance(target, torch.Tensor)
                else target.to(dtypes[1])
            )
        return preds, target

    @abstractmethod
    def measure(self, model_output, target):
        raise NotImplementedError


class AccuracyScore(AccuracyBase):
    name: str = "accuracy_score"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        accuracy = torchmetrics.Accuracy(**self.config_dict)
        result = accuracy(preds_tensor, target_tensor)
        return result.item()


class F1Score(AccuracyBase):
    name: str = "f1_score"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        f1 = torchmetrics.F1Score(**self.config_dict)
        result = f1(preds_tensor, target_tensor)
        return result.item()


class Precision(AccuracyBase):
    name: str = "precision"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        precision = torchmetrics.Precision(**self.config_dict)
        result = precision(preds_tensor, target_tensor)
        return result.item()


class Recall(AccuracyBase):
    name: str = "recall"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        recall = torchmetrics.Recall(**self.config_dict)
        result = recall(preds_tensor, target_tensor)
        return result.item()


class AUROC(AccuracyBase):
    name: str = "auroc"

    def measure(self, model_output, target):
        logits_tensor, target_tensor = self.prepare_tensors(model_output.logits, target, [torch.float, torch.int32])
        if self.config_dict.get("task") == "binary" and len(logits_tensor.shape) > 1 and logits_tensor.shape[-1] == 2:
            logits_tensor = torch.softmax(logits_tensor, dim=-1)[:, 1]
        auroc = torchmetrics.AUROC(**self.config_dict)
        target_tensor = target_tensor.flatten()
        result = auroc(logits_tensor, target_tensor)
        return result.item()


class Perplexity(AccuracyBase):
    name: str = "perplexity"

    def measure(self, model_output, target):
        # update ignore_index if not set
        config = self.config_dict
        if config["ignore_index"] is None:
            config["ignore_index"] = IGNORE_INDEX

        # create perplexity metric
        perplexity = torchmetrics.text.perplexity.Perplexity(**config)

        # loop through samples
        # the logits are large matrix, so converting all to tensors at once is slow
        num_samples = len(model_output.preds)
        for i in range(num_samples):
            logits, targets = self.prepare_tensors(model_output.preds[i], target[i], dtypes=[torch.float, torch.long])
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            # shift targets to the right by one, and drop the last token of logits
            logits = logits[..., :-1, :]
            targets = targets[..., 1:]
            perplexity.update(logits, targets)
        result = perplexity.compute()
        return result.item()

class CosineSimilarity(AccuracyBase):
    name: str = "cosine_similarity"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(
            model_output.preds, target, dtypes=torch.float
        )

        if isinstance(preds_tensor, dict):
            return {
                key: self._compute_mean_similarity(preds_tensor[key], target_tensor[key])
                for key in preds_tensor
            }
        else:
            return self._compute_mean_similarity(preds_tensor, target_tensor)

    def _compute_mean_similarity(self, preds, targets):
        if preds.shape[0] != targets.shape[0]:
            raise ValueError(f"Batch size mismatch: preds {preds.shape}, targets {targets.shape}")
        # import pdb;pdb.set_trace()
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        similarities = torch.nn.functional.cosine_similarity(preds_flat, targets_flat, dim=1)
        return similarities.mean().item()

class MSE(AccuracyBase):
    name: str = "mse"
    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(
            model_output.preds, target, dtypes=torch.float
        )
        def flatten(tensor):
            # return tensor.view(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
            return tensor.view(-1).unsqueeze(0)
        def compute_mse(pred, tgt):
            pred = flatten(pred)
            tgt = flatten(tgt)
            if pred.shape != tgt.shape:
                raise ValueError(f"Shape mismatch: preds {pred.shape}, target {tgt.shape}")
            return torch.mean((pred - tgt) ** 2).item()
        if isinstance(preds_tensor, dict):
            result = {}
            for k in preds_tensor:
                result[k] = compute_mse(preds_tensor[k], target_tensor[k])
            return result
        else:
            return compute_mse(preds_tensor, target_tensor)

# class PSNR(AccuracyBase):
#     name: str = "psnr"

#     def measure(self, model_output, target):
#         preds_tensor, target_tensor = self.prepare_tensors(
#             model_output.preds, target, dtypes=torch.float
#         )

#         def flatten(tensor):
#             # return tensor.view(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
#             return tensor.view(-1).unsqueeze(0)

#         def compute_psnr(pred, tgt):
#             pred = flatten(pred)
#             tgt = flatten(tgt)
#             if pred.shape != tgt.shape:
#                 raise ValueError(f"Shape mismatch: preds {pred.shape}, target {tgt.shape}")
#             mse = torch.mean((pred - tgt) ** 2, dim=1)
#             epsilon = 1e-7
#             max_pixel = torch.max(tgt, dim=1).values
#             max_pixel = torch.clamp(max_pixel, min=epsilon)
#             psnr = 20 * torch.log10(max_pixel) - 10 * torch.log10(mse + epsilon)
#             return torch.mean(psnr).item()

#         if isinstance(preds_tensor, dict):
#             result = {}
#             for k in preds_tensor:
#                 result[k] = compute_psnr(preds_tensor[k], target_tensor[k])
#             return result
#         else:
#             return compute_psnr(preds_tensor, target_tensor)

class PSNR(AccuracyBase):
    name: str = "psnr"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(
            model_output.preds, target, dtypes=torch.float
        )

        if isinstance(preds_tensor, dict):
            return {
                key: self._compute_mean_psnr(preds_tensor[key], target_tensor[key])
                for key in preds_tensor
            }
        else:
            return self._compute_mean_psnr(preds_tensor, target_tensor)

    def _compute_mean_psnr(self, preds, targets):
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        mse = torch.mean((preds_flat - targets_flat) ** 2, dim=1)
        epsilon = 1e-7
        max_pixel = torch.max(targets_flat, dim=1).values.clamp(min=epsilon)
        psnr = 20 * torch.log10(max_pixel) - 10 * torch.log10(mse + epsilon)
        return psnr.mean().item()

# class SQNR(AccuracyBase):
#     name: str = "sqnr"

#     def measure(self, model_output, target):
#         preds_tensor, target_tensor = self.prepare_tensors(
#             model_output.preds, target, dtypes=torch.float
#         )

#         def flatten(tensor):
#             # return tensor.view(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
#             return tensor.view(-1).unsqueeze(0)

#         def compute_sqnr(pred, tgt):
#             pred = flatten(pred)
#             tgt = flatten(tgt)
#             if pred.shape != tgt.shape:
#                 raise ValueError(f"Shape mismatch: preds {pred.shape}, target {tgt.shape}")
#             epsilon = torch.finfo(torch.float32).eps
#             signal_power = torch.norm(tgt, dim=1).clamp(min=epsilon)
#             noise_power = torch.norm(tgt - pred, dim=1).clamp(min=epsilon)
#             sqnr = 10 * torch.log10(signal_power / noise_power)
#             return torch.mean(sqnr).item()

#         if isinstance(preds_tensor, dict):
#             result = {}
#             for k in preds_tensor:
#                 result[k] = compute_sqnr(preds_tensor[k], target_tensor[k])
#             return result
#         else:
#             return compute_sqnr(preds_tensor, target_tensor)

class SQNR(AccuracyBase):
    name: str = "sqnr"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(
            model_output.preds, target, dtypes=torch.float
        )

        if isinstance(preds_tensor, dict):
            return {
                key: self._compute_mean_sqnr(preds_tensor[key], target_tensor[key])
                for key in preds_tensor
            }
        else:
            return self._compute_mean_sqnr(preds_tensor, target_tensor)

    def _compute_mean_sqnr(self, preds, targets):
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        epsilon = torch.finfo(torch.float32).eps
        signal_power = torch.norm(targets_flat, dim=1).clamp(min=epsilon)
        noise_power = torch.norm(targets_flat - preds_flat, dim=1).clamp(min=epsilon)
        sqnr = 10 * torch.log10(signal_power / noise_power)
        return sqnr.mean().item()

# class L2Norm(AccuracyBase):
#     name: str = "l2_norm"

#     def measure(self, model_output, target):
#         preds_tensor, target_tensor = self.prepare_tensors(
#             model_output.preds, target, dtypes=torch.float
#         )

#         def flatten(tensor):
#             # return tensor.view(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
#             return tensor.view(-1).unsqueeze(0)

#         def compute_l2(pred, tgt):
#             pred = flatten(pred)
#             tgt = flatten(tgt)
#             if pred.shape != tgt.shape:
#                 raise ValueError(f"Shape mismatch: preds {pred.shape}, target {tgt.shape}")
#             # Compute L2 norm per sample, then average
#             l2 = torch.norm(pred - tgt, p=2, dim=1)
#             return torch.mean(l2).item()

#         if isinstance(preds_tensor, dict):
#             result = {}
#             for k in preds_tensor:
#                 result[k] = compute_l2(preds_tensor[k], target_tensor[k])
#             return result
#         else:
#             return compute_l2(preds_tensor, target_tensor)

class L2Norm(AccuracyBase):
    name: str = "l2_norm"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(
            model_output.preds, target, dtypes=torch.float
        )

        if isinstance(preds_tensor, dict):
            return {
                key: self._compute_mean_l2(preds_tensor[key], target_tensor[key])
                for key in preds_tensor
            }
        else:
            return self._compute_mean_l2(preds_tensor, target_tensor)

    def _compute_mean_l2(self, preds, targets):
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        l2 = torch.norm(preds_flat - targets_flat, p=2, dim=1)
        return l2.mean().item()