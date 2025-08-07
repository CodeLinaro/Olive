# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.evaluator.olive_evaluator import OliveEvaluatorConfig


class TestEvaluation:
    def test_metrics_config(self):
        metrics_config = [
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                    {
                        "name": "auroc",
                        "priority": -1,
                        "goal": {"type": "max-degradation", "value": 0.01},
                    },
                    {"name": "f1_score"},
                    {"name": "precision"},
                    {"name": "recall"},
                    {"name": "perplexity"},
                ],
            },
            {
                "name": "hf_accuracy",
                "type": "accuracy",
                "backend": "huggingface_metrics",
                "sub_types": [
                    {"name": "precision", "priority": -1, "goal": {"type": "max-degradation", "value": 0.01}},
                    {
                        "name": "recall",
                        "priority": -1,
                        "metric_config": {
                            "load_params": {"process_id": 0},
                            "compute_params": {"suffix": True},
                            "result_key": "recall",
                        },
                    },
                ],
            },
            {
                "name": "latency",
                "type": "latency",
                "sub_types": [
                    {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 20}},
                    {"name": "max"},
                    {"name": "min"},
                ],
            },
            {
                "name": "test",
                "type": "custom",
                "sub_types": [
                    {
                        "name": "test",
                        "priority": 3,
                        "higher_is_better": True,
                        "goal": {"type": "max-degradation", "value": 0.01},
                    }
                ],
            },
        ]

        metrics = OliveEvaluatorConfig(metrics=metrics_config).metrics
        for metric in metrics:
            assert metric.user_config, "user_config should not be None anytime"
            assert metric.name in ["accuracy", "hf_accuracy", "latency", "test"]

    def test_cosine_similarity(self):
        config = {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [{"name": "cosine_similarity", "priority": -1, "higher_is_better": True}],
        }
        metrics = OliveEvaluatorConfig(metrics=[config]).metrics
        assert metrics[0].sub_types[0].name == "cosine_similarity"

    def test_mse(self):
        config = {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [{"name": "mse", "priority": 1, "higher_is_better": False}],
        }
        metrics = OliveEvaluatorConfig(metrics=[config]).metrics
        assert metrics[0].sub_types[0].name == "mse"

    def test_psnr(self):
        config = {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [{"name": "psnr", "priority": -1, "higher_is_better": True}],
        }
        metrics = OliveEvaluatorConfig(metrics=[config]).metrics
        assert metrics[0].sub_types[0].name == "psnr"

    def test_sqnr(self):
        config = {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [{"name": "sqnr", "priority": -1, "higher_is_better": True}],
        }
        metrics = OliveEvaluatorConfig(metrics=[config]).metrics
        assert metrics[0].sub_types[0].name == "sqnr"

    def test_l2_norm(self):
        config = {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [{"name": "l2_norm", "priority": -1, "higher_is_better": False}],
        }
        metrics = OliveEvaluatorConfig(metrics=[config]).metrics
        assert metrics[0].sub_types[0].name == "l2_norm"
