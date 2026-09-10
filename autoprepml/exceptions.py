"""Stable exception types exposed by AutoPrepML."""


class AutoPrepMLError(Exception):
    """Base class for errors raised by the AutoPrepML library."""


class ConfigurationError(AutoPrepMLError, ValueError):
    """Configuration is missing, invalid, or incompatible."""


class NotFittedError(AutoPrepMLError):
    """A fitted artifact or transformation was required but is unavailable."""


class ContractError(AutoPrepMLError, ValueError):
    """Input data does not satisfy a declared data contract."""


class ValidationError(AutoPrepMLError, ValueError):
    """A validation operation could not be completed."""


class ArtifactError(AutoPrepMLError):
    """A serialized preprocessing artifact is invalid or incompatible."""


class StorageError(AutoPrepMLError, OSError):
    """A storage operation failed."""


class IntegrationError(AutoPrepMLError):
    """An optional integration is unavailable or returned an invalid result."""


__all__ = [
    "AutoPrepMLError",
    "ConfigurationError",
    "NotFittedError",
    "ContractError",
    "ValidationError",
    "ArtifactError",
    "StorageError",
    "IntegrationError",
]
