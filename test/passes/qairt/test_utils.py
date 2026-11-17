# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import importlib.metadata
from unittest.mock import patch

import pytest

from olive.passes.qairt.utils import QairtLogLevel, qairt_dev_version_at_least


@pytest.mark.parametrize("value", ["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"])
def test_qairt_log_level_string_equality(value):
    """QairtLogLevel members compare equal to their string values."""
    assert QairtLogLevel(value) == value


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        (QairtLogLevel.ERROR, "error"),
        (QairtLogLevel.WARNING, "warn"),
        (QairtLogLevel.INFO, "info"),
        (QairtLogLevel.DEBUG, "verbose"),
        (QairtLogLevel.TRACE, "verbose"),
    ],
)
def test_to_genie_log_level(log_level, expected):
    """Each QairtLogLevel maps to the correct Genie log level string."""
    assert log_level.to_genie_log_level() == expected


def test_qairt_log_level_invalid_value():
    """Invalid values are rejected by the enum."""
    with pytest.raises(ValueError, match="'WARN' is not a valid QairtLogLevel"):
        QairtLogLevel("WARN")


@pytest.mark.parametrize(
    ("installed", "threshold", "expected"),
    [
        ("0.15.0", "0.15.0.dev0", True),
        ("0.15.0.dev0", "0.15.0.dev0", True),
        ("0.15.1", "0.15.0.dev0", True),
        ("0.8.0", "0.8.0", True),
        ("0.14.1", "0.15.0.dev0", False),
        ("0.14.0", "0.15.0.dev0", False),
        ("0.14.0.dev0", "0.15.0.dev0", False),
        ("0.7.0", "0.8.0", False),
    ],
)
def test_qairt_dev_version_at_least(installed, threshold, expected):
    with patch("olive.passes.qairt.utils.importlib.metadata.version", return_value=installed):
        assert qairt_dev_version_at_least(threshold) is expected


def test_qairt_dev_version_at_least_not_installed():
    """Returns False when qairt-dev is not installed."""
    with patch(
        "olive.passes.qairt.utils.importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        assert qairt_dev_version_at_least("0.15.0.dev0") is False
