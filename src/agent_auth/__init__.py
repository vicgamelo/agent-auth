"""OAuth 2.1 authorization for AI agents and MCP servers."""

from .guard import Principal, ProtectedResource, ServiceClient, TokenVerifier, get_principal

__version__ = "0.1.0"

__all__ = ["Principal", "ProtectedResource", "ServiceClient", "TokenVerifier", "__version__", "get_principal"]
