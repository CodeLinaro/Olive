# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import importlib.metadata

from packaging.version import Version

from olive.common.utils import StrEnumBase


def qairt_dev_version_at_least(version: str) -> bool:
    """Return True if the installed qairt-dev package version is >= version."""
    try:
        return Version(importlib.metadata.version("qairt-dev")) >= Version(version)
    except importlib.metadata.PackageNotFoundError:
        return False


class QairtLogLevel(StrEnumBase):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"

    def to_genie_log_level(self) -> str:
        """Map to the closest Genie log level string."""
        return {
            QairtLogLevel.ERROR: "error",
            QairtLogLevel.WARNING: "warn",
            QairtLogLevel.INFO: "info",
            QairtLogLevel.DEBUG: "verbose",
            QairtLogLevel.TRACE: "verbose",
        }[self]
