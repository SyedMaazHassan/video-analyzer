"""
Configuration Module
====================
System configuration and settings.
"""

try:
    from .system_config import ConfigurationManager, get_config_manager, get_system_config
    __all__ = ['ConfigurationManager', 'get_config_manager', 'get_system_config']
except ImportError:
    # If yaml not available, provide minimal interface
    __all__ = []