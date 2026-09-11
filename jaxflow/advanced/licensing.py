"""
License Management System for JaxFlow Advanced
=============================================

Handles license validation, tier management, and feature access control.
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum


class LicenseTier(Enum):
    """Premium license tiers."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class LicenseStatus(Enum):
    """License validation status."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    OFFLINE_VALID = "offline_valid"


class LicenseManager:
    """
    Manages premium license validation and feature access.
    
    Features:
    - Online license validation
    - Offline mode support
    - Tier-based feature access
    - Usage tracking
    """
    
    # Feature mapping by tier
    FEATURE_TIERS = {
        "advanced_loader": [LicenseTier.BASIC, LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE],
        "distributed_training": [LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE],
        "cloud_integration": [LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE],
        "monitoring": [LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE],
        "experiment_tracking": [LicenseTier.PROFESSIONAL, LicenseTier.ENTERPRISE],
        "security_compliance": [LicenseTier.ENTERPRISE],
        "team_collaboration": [LicenseTier.ENTERPRISE],
        "sla_support": [LicenseTier.ENTERPRISE],
    }
    
    def __init__(self, license_key: Optional[str] = None):
        self.license_key = license_key or os.getenv("JAXFLOW_LICENSE_KEY")
        self.license_cache: Optional[Dict[str, Any]] = None
        self.last_validation: Optional[datetime] = None
        self.license_server_url = "https://api.jaxflow.com/v1/validate"
        self.offline_grace_period = timedelta(days=7)
    
    def validate(self, force_online: bool = False) -> LicenseStatus:
        """Validate the current license."""
        if not self.license_key:
            return LicenseStatus.INVALID
        
        # Check cache first (24-hour validity)
        if not force_online and self._is_cached_valid():
            return LicenseStatus.VALID
        
        # Try online validation
        try:
            status = self._validate_online()
            if status == LicenseStatus.VALID:
                return status
        except Exception:
            pass
        
        # Fall back to offline validation
        return self._validate_offline()
    
    def _is_cached_valid(self) -> bool:
        """Check if cached license is still valid."""
        if not self.license_cache or not self.last_validation:
            return False
        return datetime.now() - self.last_validation < timedelta(hours=24)
    
    def _validate_online(self) -> LicenseStatus:
        """Validate license against the server."""
        payload = {
            "license_key": self.license_key,
            "machine_id": self._get_machine_id(),
            "version": __version__,
        }
        
        response = requests.post(
            self.license_server_url,
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            self._cache_license(data)
            self.last_validation = datetime.now()
            return LicenseStatus.VALID
        
        return LicenseStatus.INVALID
    
    def _validate_offline(self) -> LicenseStatus:
        """Validate in offline mode using cache."""
        if not self.license_cache:
            return LicenseStatus.INVALID
        
        expiry = datetime.fromisoformat(self.license_cache.get("expires_at", ""))
        if datetime.now() > expiry:
            return LicenseStatus.EXPIRED
        
        # Check offline grace period from last online validation
        if self.last_validation:
            if datetime.now() - self.last_validation > self.offline_grace_period:
                return LicenseStatus.EXPIRED
        
        return LicenseStatus.OFFLINE_VALID
    
    def _get_machine_id(self) -> str:
        """Generate a unique machine identifier."""
        import platform
        import uuid
        machine_info = f"{platform.node()}-{uuid.getnode()}"
        return hashlib.sha256(machine_info.encode()).hexdigest()[:16]
    
    def _cache_license(self, data: Dict[str, Any]):
        """Cache license data locally."""
        self.license_cache = data
        cache_dir = os.path.expanduser("~/.jaxflow")
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(os.path.join(cache_dir, "license_cache.json"), "w") as f:
            json.dump(data, f)
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if current license includes the feature."""
        status = self.validate()
        if status in [LicenseStatus.INVALID, LicenseStatus.EXPIRED]:
            return False
        
        tier = LicenseTier(self.license_cache.get("tier", "basic"))
        return tier in self.FEATURE_TIERS.get(feature_name, [])
    
    def get_tier(self) -> Optional[LicenseTier]:
        """Get the current license tier."""
        if self.license_cache:
            return LicenseTier(self.license_cache.get("tier", "basic"))
        return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for analytics."""
        return {
            "tier": self.get_tier().value if self.get_tier() else None,
            "validated_at": self.last_validation.isoformat() if self.last_validation else None,
            "features_used": [],
        }


def require_license(feature: str):
    """Decorator to enforce license requirements for premium features."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get license manager from args or kwargs
            license_manager = kwargs.get('license_manager')
            if not license_manager and len(args) > 0:
                # Check if first arg has license_manager attribute
                obj = args[0]
                license_manager = getattr(obj, 'license_manager', None)
            
            if not license_manager:
                # Fall back to global license manager
                license_manager = LicenseManager()
            
            if not license_manager.has_feature(feature):
                raise PermissionError(
                    f"Feature '{feature}' requires a JaxFlow Advanced license. "
                    f"Visit https://jaxflow.com/pricing to upgrade."
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
