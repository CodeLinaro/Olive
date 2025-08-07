# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olive.evaluator.accuracy import (
    AUROC,
    MSE,
    PSNR,
    SQNR,
    AccuracyScore,
    CosineSimilarity,
    F1Score,
    L2Norm,
    Perplexity,
    Precision,
    Recall,
)
from olive.evaluator.olive_evaluator import OliveModelOutput


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
@pytest.mark.parametrize(
    "metric_config",
    [
        {"task": "binary"},
        {"task": "binary", "kwargs": {"test": 2}},
    ],
)
def test_evaluate_accuracyscore(mock_torchmetrics, mock_torch_tensor, metric_config):
    # setup
    acc = AccuracyScore(metric_config)
    assert "kwargs" in acc.config.dict()
    assert "kwargs" not in acc.config_dict
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Accuracy().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_f1score(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = F1Score()
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.F1Score().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_precision(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = Precision()
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Precision().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_recall(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = Recall()
    model_output = OliveModelOutput([1, 0, 1, 1], None)
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.Recall().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.preds, dtype=torch.int32)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_auc(mock_torchmetrics, mock_torch_tensor):
    # setup
    acc = AUROC()
    model_output = OliveModelOutput(None, [1, 0, 1, 1])
    targets = [1, 1, 1, 1]
    expected_res = 0.99
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.AUROC().return_value = mock_res

    # execute
    actual_res = acc.measure(model_output, targets)

    # assert
    mock_torch_tensor.assert_any_call(model_output.logits, dtype=torch.float)
    mock_torch_tensor.assert_any_call(targets, dtype=torch.int32)
    assert actual_res == expected_res


@patch("olive.evaluator.accuracy.torch.tensor")
@patch("olive.evaluator.accuracy.torchmetrics")
def test_evaluate_perplexity(mock_torchmetrics, mock_torch_tensor):
    # setup
    Perplexity()
    batch = 2
    seqlen = 3
    vocab_size = 10
    model_output = OliveModelOutput(np.random.rand(batch, seqlen, vocab_size).tolist(), None)
    targets = np.random.randint(0, vocab_size, (batch, seqlen)).tolist()
    expected_res = 20.0
    mock_res = MagicMock()
    mock_res.item.return_value = expected_res
    mock_torchmetrics.text.perplexity.Perplexity().compute.return_value = mock_res

    # execute
    actual_res = Perplexity().measure(model_output, targets)

    # assert
    for i in range(batch):
        mock_torch_tensor.assert_any_call(model_output.preds[i], dtype=torch.float)
        mock_torch_tensor.assert_any_call(targets[i], dtype=torch.long)
    assert actual_res == expected_res


# Save original torch functions
original_tensor = torch.tensor
original_log10 = torch.log10
original_norm = torch.norm


@patch("olive.evaluator.accuracy.torch.nn.functional.cosine_similarity")
@patch("olive.evaluator.accuracy.torch.tensor")
def test_cosine_similarity(mock_tensor, mock_cosine_similarity):
    metric = CosineSimilarity()
    preds = [[1, 0], [0, 1]]
    targets = [[1, 0], [0, 1]]

    mock_tensor.side_effect = lambda x, dtype=None: original_tensor(x, dtype=torch.float)
    mock_cosine_similarity.return_value = torch.tensor([1.0, 1.0])

    result = metric.measure(OliveModelOutput(preds, None), targets)
    assert result["avg"] == 1.0
    assert result["min"] == 1.0
    assert result["max"] == 1.0


@patch("olive.evaluator.accuracy.torch.mean")
@patch("olive.evaluator.accuracy.torch.tensor")
def test_mse(mock_tensor, mock_mean):
    metric = MSE()
    preds = [[1, 2], [3, 4]]
    targets = [[1, 2], [3, 4]]

    mock_tensor.side_effect = lambda x, dtype=None: original_tensor(x, dtype=torch.float)
    mock_mean.return_value = torch.tensor([0.0, 0.0])

    result = metric.measure(OliveModelOutput(preds, None), targets)
    assert result["avg"] == 0.0
    assert result["min"] == 0.0
    assert result["max"] == 0.0


@patch("olive.evaluator.accuracy.torch.log10")
@patch("olive.evaluator.accuracy.torch.mean")
@patch("olive.evaluator.accuracy.torch.tensor")
def test_psnr(mock_tensor, mock_mean, mock_log10):
    metric = PSNR()
    preds = [[1, 2], [3, 4]]
    targets = [[1, 2], [3, 4]]

    mock_tensor.side_effect = lambda x, dtype=None: original_tensor(x, dtype=torch.float)
    mock_mean.return_value = torch.tensor([0.0, 0.0])
    mock_log10.side_effect = lambda x: original_log10(x)

    result = metric.measure(OliveModelOutput(preds, None), targets)
    assert isinstance(result["avg"], float)


@patch("olive.evaluator.accuracy.torch.norm")
@patch("olive.evaluator.accuracy.torch.log10")
@patch("olive.evaluator.accuracy.torch.tensor")
def test_sqnr(mock_tensor, mock_log10, mock_norm):
    metric = SQNR()
    preds = [[1, 2], [3, 4]]
    targets = [[1, 2], [3, 4]]

    mock_tensor.side_effect = lambda x, dtype=None: original_tensor(x, dtype=torch.float)
    mock_norm.side_effect = lambda x, dim=None: original_norm(x, dim=dim)
    mock_log10.side_effect = lambda x: original_log10(x)

    result = metric.measure(OliveModelOutput(preds, None), targets)
    assert isinstance(result["avg"], float)


@patch("olive.evaluator.accuracy.torch.norm")
@patch("olive.evaluator.accuracy.torch.tensor")
def test_l2_norm(mock_tensor, mock_norm):
    metric = L2Norm()
    preds = [[1, 2], [3, 4]]
    targets = [[1, 2], [3, 4]]

    mock_tensor.side_effect = lambda x, dtype=None: original_tensor(x, dtype=torch.float)
    mock_norm.side_effect = lambda x, p=2, dim=1: original_norm(x, p=p, dim=dim)

    result = metric.measure(OliveModelOutput(preds, None), targets)
    assert isinstance(result["avg"], float)
