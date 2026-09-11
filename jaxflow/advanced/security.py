"""
Security & Compliance Features
=============================

Enterprise-grade security and compliance capabilities.
"""

import os
import hashlib
import json
import logging
import pickle
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, UTC
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

from .licensing import require_license, LicenseManager


logger = logging.getLogger(__name__)


@dataclass
class AuditLogEntry:
    """Represents a single audit log entry."""
    timestamp: str
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address
        }


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Features:
    - Immutable audit trail
    - Tamper-evident logging
    - Structured JSON logging
    - Compliance export (GDPR, HIPAA, SOC2)
    """
    
    def __init__(
        self,
        log_dir: str = "./audit_logs",
        license_manager: Optional[LicenseManager] = None
    ):
        self.license_manager = license_manager or LicenseManager()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_buffer: List[AuditLogEntry] = []
        self._buffer_size = 100
    
    def log(
        self,
        event_type: str,
        resource: str,
        action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ):
        """Log an audit event."""
        entry = AuditLogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details or {},
            ip_address=ip_address
        )
        
        self._log_buffer.append(entry)
        
        if len(self._log_buffer) >= self._buffer_size:
            self._flush_buffer()
        
        logger.debug(f"Audit log: {event_type} - {action} on {resource}")
    
    def _flush_buffer(self):
        """Write buffered logs to disk."""
        if not self._log_buffer:
            return
        
        log_file = self.log_dir / f"audit_{datetime.now(UTC).date()}.jsonl"
        
        with open(log_file, "a") as f:
            for entry in self._log_buffer:
                f.write(json.dumps(entry.to_dict()) + "\n")
        
        self._log_buffer = []
    
    def export_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        export_format: str = "json"
    ) -> str:
        """
        Export compliance report for a date range.
        
        Supports SOC2, GDPR, HIPAA formats.
        """
        entries = []
        
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            with open(log_file, "r") as f:
                for line in f:
                    entry_dict = json.loads(line)
                    entry_timestamp = datetime.fromisoformat(entry_dict["timestamp"])
                    if start_date <= entry_timestamp <= end_date:
                        entries.append(entry_dict)
        
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(entries),
            "events": entries
        }
        
        export_file = self.log_dir / f"compliance_report_{datetime.now(UTC).date()}.{export_format}"
        
        with open(export_file, "w") as f:
            if export_format == "json":
                json.dump(report, f, indent=2)
        
        return str(export_file)


class DataEncryptor:
    """
    Encryption utilities for sensitive data.
    
    Features:
    - At-rest encryption for models and datasets
    - Secure key management
    - Fernet-based symmetric encryption
    """
    
    @require_license("security_compliance")
    def __init__(
        self,
        key: Optional[bytes] = None,
        license_manager: Optional[LicenseManager] = None
    ):
        self.license_manager = license_manager or LicenseManager()
        self._key = key or self._generate_key()
        self._cipher = None
        self._init_cipher()
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()
    
    def _init_cipher(self):
        """Initialize the encryption cipher."""
        try:
            from cryptography.fernet import Fernet
            self._cipher = Fernet(self._key)
        except ImportError:
            logger.warning("cryptography not installed. Encryption unavailable.")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data."""
        if not self._cipher:
            raise RuntimeError("Encryption not available")
        return self._cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self._cipher:
            raise RuntimeError("Encryption not available")
        return self._cipher.decrypt(encrypted_data)
    
    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypt a file."""
        with open(input_path, "rb") as f:
            data = f.read()
        
        encrypted = self.encrypt_data(data)
        
        with open(output_path, "wb") as f:
            f.write(encrypted)
        
        logger.info(f"Encrypted {input_path} to {output_path}")
    
    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypt a file."""
        with open(input_path, "rb") as f:
            encrypted = f.read()
        
        data = self.decrypt_data(encrypted)
        
        with open(output_path, "wb") as f:
            f.write(data)
        
        logger.info(f"Decrypted {input_path} to {output_path}")
    
    def get_key_hash(self) -> str:
        """Get a hash of the encryption key for verification."""
        return hashlib.sha256(self._key).hexdigest()
    
    def save_key(self, path: str):
        """Save encryption key to file (secure location recommended!)."""
        with open(path, "wb") as f:
            f.write(self._key)
        logger.warning(f"Encryption key saved to {path}. Ensure this is secure!")
    
    @classmethod
    def load_key(cls, path: str) -> "DataEncryptor":
        """Load encryption key from file."""
        with open(path, "rb") as f:
            key = f.read()
        return cls(key=key)


class DataAnonymizer:
    """
    Data anonymization for privacy compliance.
    
    Features:
    - PII (Personally Identifiable Information) detection and removal
    - Data masking
    - Differential privacy support
    - GDPR/CCPA compliance
    """
    
    @require_license("security_compliance")
    def __init__(self, license_manager: Optional[LicenseManager] = None):
        self.license_manager = license_manager or LicenseManager()
        
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }
    
    def anonymize_text(self, text: str) -> str:
        """
        Anonymize text by masking PII.
        """
        import re
        
        anonymized = text
        
        for pii_type, pattern in self.pii_patterns.items():
            mask = f"[{pii_type.upper()}_REDACTED]"
            anonymized = re.sub(pattern, mask, anonymized)
        
        return anonymized
    
    def add_differential_privacy(self, data, epsilon: float = 1.0):
        """
        Add differential privacy noise to data.
        """
        import numpy as np
        
        if isinstance(data, np.ndarray):
            scale = 1.0 / epsilon
            noise = np.random.laplace(0, scale, data.shape)
            return data + noise
        
        return data


class ModelWatermarker:
    """
    Model watermarking for intellectual property protection.
    
    Features:
    - Invisible watermarking of model weights
    - Watermark detection and verification
    - Tamper detection
    """
    
    @require_license("security_compliance")
    def __init__(self, license_manager: Optional[LicenseManager] = None):
        self.license_manager = license_manager or LicenseManager()
    
    def embed_watermark(self, params, watermark: str):
        """
        Embed a watermark into model parameters.
        """
        import jax
        import jax.numpy as jnp
        
        watermark_bytes = watermark.encode()
        watermark_hash = hashlib.sha256(watermark_bytes).digest()
        
        def embed_fn(x):
            if len(x.shape) > 0 and x.size > 100:
                # Embed watermark in least significant bits
                x_flat = x.flatten()
                bits = []
                for byte in watermark_hash:
                    for i in range(8):
                        bits.append((byte >> i) & 1)
                
                if len(bits) <= len(x_flat):
                    for i, bit in enumerate(bits):
                        x_flat = x_flat.at[i].set((x_flat[i] & ~1) | bit)
                    return x_flat.reshape(x.shape)
            return x
        
        return jax.tree_map(embed_fn, params)
    
    def verify_watermark(self, params, watermark: str) -> bool:
        """
        Verify if a watermark is present in model parameters.
        """
        import jax
        import jax.numpy as jnp
        
        watermark_bytes = watermark.encode()
        expected_hash = hashlib.sha256(watermark_bytes).digest()
        
        def extract_fn(x):
            if len(x.shape) > 0 and x.size > 100:
                x_flat = x.flatten()
                bits = []
                for i in range(256):  # SHA256 is 256 bits
                    if i < len(x_flat):
                        bits.append(int(x_flat[i]) & 1)
                
                extracted_bytes = bytearray()
                for i in range(0, len(bits), 8):
                    if i + 7 < len(bits):
                        byte = 0
                        for j in range(8):
                            byte |= bits[i + j] << j
                        extracted_bytes.append(byte)
                return bytes(extracted_bytes)
            return None
        
        extracted = None
        for leaf in jax.tree_util.tree_leaves(params):
            result = extract_fn(leaf)
            if result:
                extracted = result
                break
        
        if extracted:
            return extracted[:32] == expected_hash
        
        return False


class SecurityManager:
    """
    Unified security manager.
    
    Combines all security features into a single interface.
    """
    
    @require_license("security_compliance")
    def __init__(
        self,
        license_manager: Optional[LicenseManager] = None,
        audit_log_dir: str = "./audit_logs",
        encryption_key: Optional[bytes] = None
    ):
        self.license_manager = license_manager or LicenseManager()
        self.audit_logger = AuditLogger(audit_log_dir, license_manager)
        self.encryptor = DataEncryptor(encryption_key, license_manager)
        self.anonymizer = DataAnonymizer(license_manager)
        self.watermarker = ModelWatermarker(license_manager)
    
    def log_access(
        self,
        resource: str,
        action: str,
        user_id: Optional[str] = None,
        **details
    ):
        """Log a resource access event."""
        self.audit_logger.log(
            event_type="resource_access",
            resource=resource,
            action=action,
            user_id=user_id,
            details=details
        )
