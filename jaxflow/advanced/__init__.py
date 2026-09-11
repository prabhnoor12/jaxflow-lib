"""
JaxFlow Advanced - Premium Enterprise Features
=========================================

This module contains premium, enterprise-grade features for JaxFlow.
These features require a valid license key to activate.

Copyright (c) 2024 JaxFlow. All rights reserved.
"""

__version__ = "0.1.0"

from .licensing import (
    LicenseManager,
    LicenseTier,
    require_license
)
from .core import (
    AdvancedLoaderPro,
    DistributedTrainer,
    AutoScalingDataPipeline,
    FaultTolerantLoader
)
from .cloud import (
    CloudManager,
    CloudStorage,
    S3Storage,
    GCSStorage,
    AzureBlobStorage,
    CloudCredentials
)
from .monitoring import (
    SystemMonitor,
    TrainingMonitor,
    JAXProfiler,
    Metric
)
from .experiment import (
    ExperimentTracker,
    Experiment,
    HyperparameterOptimizer
)
from .security import (
    SecurityManager,
    AuditLogger,
    DataEncryptor,
    DataAnonymizer,
    ModelWatermarker,
    AuditLogEntry
)

__all__ = [
    # Licensing
    'LicenseManager',
    'LicenseTier',
    'require_license',
    # Core
    'AdvancedLoaderPro',
    'DistributedTrainer',
    'AutoScalingDataPipeline',
    'FaultTolerantLoader',
    # Cloud
    'CloudManager',
    'CloudStorage',
    'S3Storage',
    'GCSStorage',
    'AzureBlobStorage',
    'CloudCredentials',
    # Monitoring
    'SystemMonitor',
    'TrainingMonitor',
    'JAXProfiler',
    'Metric',
    # Experiment
    'ExperimentTracker',
    'Experiment',
    'HyperparameterOptimizer',
    # Security
    'SecurityManager',
    'AuditLogger',
    'DataEncryptor',
    'DataAnonymizer',
    'ModelWatermarker',
    'AuditLogEntry',
]
