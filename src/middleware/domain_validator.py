"""
Domain Validator for DRX Deep Research Platform.

Provides URL and domain validation with support for wildcard patterns.
Used by the PolicyFirewall to enforce domain restrictions from agent manifests.

Key Features:
- Extract domain from URLs (handles various URL formats)
- Wildcard pattern matching (e.g., "*.gov", "*.edu")
- Allowed/blocked domain list validation
- Support for subdomain matching

Part of WP-M6: Metadata Firewall Middleware implementation.

Usage:
    from src.middleware.domain_validator import DomainValidator

    validator = DomainValidator()

    # Check if domain is allowed
    is_allowed = await validator.is_domain_allowed(
        agent_id="searcher_v1",
        domain="example.gov",
    )

    # Validate a full URL
    allowed, reason = await validator.validate_url(
        agent_id="searcher_v1",
        url="https://data.example.gov/api/v1/data",
    )
"""

from __future__ import annotations

import fnmatch
import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from src.metadata.manifest import AgentManifest

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class DomainValidationError(Exception):
    """Base exception for domain validation errors."""

    pass


# =============================================================================
# Constants
# =============================================================================

# Common TLDs for validation
VALID_TLDS = frozenset([
    "com", "org", "net", "edu", "gov", "mil", "int",
    "io", "co", "ai", "app", "dev", "cloud", "tech",
    "uk", "de", "fr", "jp", "cn", "au", "ca", "br",
    "eu", "us", "info", "biz", "name", "pro", "museum",
])

# Regex for validating domain format
DOMAIN_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,}$"
)

# Regex for IP addresses (to handle IP-based URLs)
IP_REGEX = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


# =============================================================================
# DomainValidator Class
# =============================================================================


class DomainValidator:
    """
    Validates domains and URLs against agent-specific allow/block lists.

    This class provides domain validation functionality for the PolicyFirewall
    middleware. It supports wildcard patterns in domain lists and handles
    various URL formats.

    Wildcard Pattern Support:
        - "*.gov" matches any .gov domain (e.g., "data.gov", "api.census.gov")
        - "*.example.com" matches subdomains (e.g., "api.example.com")
        - "example.*" matches any TLD (e.g., "example.com", "example.org")

    Example:
        >>> validator = DomainValidator()
        >>> await validator.is_domain_allowed("searcher_v1", "data.gov")
        True
        >>> await validator.validate_url("searcher_v1", "https://blocked.com/page")
        (False, "Domain 'blocked.com' is in blocked list")

    Attributes:
        _manifest_cache: Cache for agent manifest lookups
        _pattern_cache: Cache for compiled wildcard patterns
    """

    def __init__(
        self,
        manifest_loader: Any = None,
    ) -> None:
        """
        Initialize the domain validator.

        Args:
            manifest_loader: Optional callable to load agent manifests.
                             If None, uses default registry lookup.
        """
        self._manifest_loader = manifest_loader
        self._manifest_cache: dict[str, "AgentManifest"] = {}
        self._pattern_cache: dict[str, re.Pattern[str]] = {}

    async def _get_manifest(self, agent_id: str) -> "AgentManifest | None":
        """
        Get agent manifest with caching.

        Args:
            agent_id: The agent identifier (e.g., "searcher_v1")

        Returns:
            AgentManifest if found, None otherwise
        """
        if agent_id in self._manifest_cache:
            return self._manifest_cache[agent_id]

        if self._manifest_loader:
            try:
                manifest = await self._manifest_loader(agent_id)
                self._manifest_cache[agent_id] = manifest
                return manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest for {agent_id}: {e}")
                return None

        return None

    def clear_cache(self) -> None:
        """Clear the manifest and pattern caches."""
        self._manifest_cache.clear()
        self._pattern_cache.clear()

    def extract_domain(self, url: str) -> str:
        """
        Extract the domain from a URL.

        Handles various URL formats including:
        - Full URLs (https://example.com/path)
        - URLs without scheme (example.com/path)
        - URLs with ports (example.com:8080)
        - IP addresses (192.168.1.1)

        Args:
            url: The URL to extract domain from

        Returns:
            Extracted domain string (lowercase)

        Raises:
            DomainValidationError: If URL is invalid or domain cannot be extracted

        Example:
            >>> validator.extract_domain("https://api.example.com:8080/v1/data")
            'api.example.com'
        """
        if not url:
            raise DomainValidationError("URL cannot be empty")

        # Normalize the URL
        url = url.strip()

        # Add scheme if missing (urlparse requires it for proper parsing)
        if not url.startswith(("http://", "https://", "ftp://")):
            url = f"https://{url}"

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise DomainValidationError(f"Invalid URL format: {e}")

        # Extract netloc (hostname:port)
        netloc = parsed.netloc

        if not netloc:
            raise DomainValidationError(f"Could not extract domain from URL: {url}")

        # Remove port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]

        # Remove any userinfo (user:pass@)
        if "@" in netloc:
            netloc = netloc.split("@")[-1]

        domain = netloc.lower()

        # Validate domain format
        if not self._is_valid_domain_format(domain):
            raise DomainValidationError(f"Invalid domain format: {domain}")

        return domain

    def _is_valid_domain_format(self, domain: str) -> bool:
        """
        Check if a domain has valid format.

        Args:
            domain: Domain string to validate

        Returns:
            True if valid format, False otherwise
        """
        if not domain:
            return False

        # Allow IP addresses
        if IP_REGEX.match(domain):
            return True

        # Allow localhost
        if domain == "localhost":
            return True

        # Validate domain format
        if DOMAIN_REGEX.match(domain):
            return True

        # Check for single-label domains (might be internal)
        if domain.isalnum() or domain.replace("-", "").isalnum():
            return True

        return False

    @lru_cache(maxsize=256)
    def _compile_wildcard_pattern(self, pattern: str) -> re.Pattern[str]:
        """
        Compile a wildcard domain pattern to regex.

        Supports wildcards:
        - "*" matches any sequence of characters
        - "?" matches a single character

        Args:
            pattern: Wildcard pattern (e.g., "*.gov", "api.*.example.com")

        Returns:
            Compiled regex pattern
        """
        # Escape special regex characters except * and ?
        escaped = re.escape(pattern)

        # Convert wildcard patterns
        # \* -> .* (match any characters)
        # \? -> . (match single character)
        regex_pattern = escaped.replace(r"\*", r"[^.]*").replace(r"\?", r".")

        # If pattern starts with *., allow matching subdomains
        if pattern.startswith("*."):
            regex_pattern = r"(?:[a-zA-Z0-9-]+\.)?" + regex_pattern[6:]

        # Anchor the pattern
        regex_pattern = f"^{regex_pattern}$"

        return re.compile(regex_pattern, re.IGNORECASE)

    def _matches_pattern(self, domain: str, pattern: str) -> bool:
        """
        Check if domain matches a wildcard pattern.

        Args:
            domain: Domain to check
            pattern: Wildcard pattern

        Returns:
            True if domain matches pattern

        Example:
            >>> validator._matches_pattern("data.gov", "*.gov")
            True
            >>> validator._matches_pattern("api.example.com", "*.example.com")
            True
        """
        # Exact match (most common case)
        if domain.lower() == pattern.lower():
            return True

        # Check if pattern contains wildcards
        if "*" not in pattern and "?" not in pattern:
            return False

        # Use fnmatch for simple cases
        if fnmatch.fnmatch(domain.lower(), pattern.lower()):
            return True

        # Use regex for complex patterns
        try:
            compiled = self._compile_wildcard_pattern(pattern.lower())
            return bool(compiled.match(domain.lower()))
        except re.error:
            logger.warning(f"Invalid wildcard pattern: {pattern}")
            return False

    def _is_subdomain_of(self, domain: str, parent_domain: str) -> bool:
        """
        Check if domain is a subdomain of parent_domain.

        Args:
            domain: Potential subdomain
            parent_domain: Parent domain to check against

        Returns:
            True if domain is subdomain of parent_domain

        Example:
            >>> validator._is_subdomain_of("api.example.com", "example.com")
            True
        """
        domain = domain.lower()
        parent_domain = parent_domain.lower()

        if domain == parent_domain:
            return True

        return domain.endswith(f".{parent_domain}")

    async def is_domain_allowed(
        self,
        agent_id: str,
        domain: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> bool:
        """
        Check if a domain is allowed for an agent.

        Domain validation logic:
        1. If domain is in blocked_domains (or matches blocked pattern), return False
        2. If allowed_domains is empty, return True (all allowed by default)
        3. If domain is in allowed_domains (or matches allowed pattern), return True
        4. Otherwise, return False

        Args:
            agent_id: The agent identifier
            domain: Domain to validate
            allowed_domains: Optional override for allowed domains list
            blocked_domains: Optional override for blocked domains list

        Returns:
            True if domain is allowed, False otherwise

        Example:
            >>> await validator.is_domain_allowed("searcher_v1", "example.gov")
            True
            >>> await validator.is_domain_allowed(
            ...     "searcher_v1",
            ...     "malicious.com",
            ...     blocked_domains=["*.com"]
            ... )
            False
        """
        domain = domain.lower()

        # Load domain lists from manifest if not provided
        if allowed_domains is None or blocked_domains is None:
            manifest = await self._get_manifest(agent_id)
            if manifest:
                if allowed_domains is None:
                    allowed_domains = list(manifest.allowed_domains)
                if blocked_domains is None:
                    blocked_domains = list(manifest.blocked_domains)
            else:
                if allowed_domains is None:
                    allowed_domains = []
                if blocked_domains is None:
                    blocked_domains = []

        # Check blocked domains first (takes precedence)
        for blocked_pattern in blocked_domains:
            if self._matches_pattern(domain, blocked_pattern):
                logger.debug(
                    f"Domain {domain} blocked for agent {agent_id} "
                    f"(matched pattern: {blocked_pattern})"
                )
                return False

            # Check subdomain matching for blocked
            if self._is_subdomain_of(domain, blocked_pattern):
                logger.debug(
                    f"Domain {domain} blocked as subdomain of {blocked_pattern}"
                )
                return False

        # If no allowed domains specified, all non-blocked domains are allowed
        if not allowed_domains:
            return True

        # Check allowed domains
        for allowed_pattern in allowed_domains:
            if self._matches_pattern(domain, allowed_pattern):
                logger.debug(
                    f"Domain {domain} allowed for agent {agent_id} "
                    f"(matched pattern: {allowed_pattern})"
                )
                return True

            # Check subdomain matching for allowed
            if self._is_subdomain_of(domain, allowed_pattern):
                logger.debug(
                    f"Domain {domain} allowed as subdomain of {allowed_pattern}"
                )
                return True

        # Domain not in allowed list
        logger.debug(
            f"Domain {domain} not in allowed list for agent {agent_id}"
        )
        return False

    async def validate_url(
        self,
        agent_id: str,
        url: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate a URL against agent's domain restrictions.

        Extracts the domain from the URL and checks it against the
        agent's allowed and blocked domain lists.

        Args:
            agent_id: The agent identifier
            url: Full URL to validate
            allowed_domains: Optional override for allowed domains
            blocked_domains: Optional override for blocked domains

        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
            - (True, None) if URL is allowed
            - (False, "reason") if URL is not allowed

        Example:
            >>> allowed, reason = await validator.validate_url(
            ...     "searcher_v1",
            ...     "https://api.blocked-site.com/data"
            ... )
            >>> if not allowed:
            ...     print(f"Blocked: {reason}")
        """
        try:
            domain = self.extract_domain(url)
        except DomainValidationError as e:
            return False, f"Invalid URL: {e}"

        is_allowed = await self.is_domain_allowed(
            agent_id=agent_id,
            domain=domain,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

        if is_allowed:
            return True, None

        # Determine reason for blocking
        blocked_domains = blocked_domains or []
        manifest = await self._get_manifest(agent_id)
        if manifest:
            blocked_domains = list(manifest.blocked_domains)

        for blocked_pattern in blocked_domains:
            if self._matches_pattern(domain, blocked_pattern):
                return False, f"Domain '{domain}' is in blocked list (matches '{blocked_pattern}')"
            if self._is_subdomain_of(domain, blocked_pattern):
                return False, f"Domain '{domain}' is subdomain of blocked domain '{blocked_pattern}'"

        return False, f"Domain '{domain}' is not in allowed domains list"

    async def validate_urls_batch(
        self,
        agent_id: str,
        urls: list[str],
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[tuple[str, bool, str | None]]:
        """
        Validate multiple URLs in batch.

        Args:
            agent_id: The agent identifier
            urls: List of URLs to validate
            allowed_domains: Optional override for allowed domains
            blocked_domains: Optional override for blocked domains

        Returns:
            List of (url, is_allowed, reason) tuples

        Example:
            >>> results = await validator.validate_urls_batch(
            ...     "searcher_v1",
            ...     ["https://good.gov", "https://bad.com"]
            ... )
            >>> for url, allowed, reason in results:
            ...     print(f"{url}: {'OK' if allowed else reason}")
        """
        results: list[tuple[str, bool, str | None]] = []

        for url in urls:
            is_allowed, reason = await self.validate_url(
                agent_id=agent_id,
                url=url,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )
            results.append((url, is_allowed, reason))

        return results

    def get_domain_info(self, url: str) -> dict[str, Any]:
        """
        Extract detailed information from a URL's domain.

        Args:
            url: URL to analyze

        Returns:
            Dictionary with domain information

        Example:
            >>> validator.get_domain_info("https://api.example.com:8080/path")
            {
                'domain': 'api.example.com',
                'subdomain': 'api',
                'root_domain': 'example.com',
                'tld': 'com',
                'is_ip': False,
                'port': 8080,
            }
        """
        try:
            # Parse URL
            url_stripped = url.strip()
            if not url_stripped.startswith(("http://", "https://", "ftp://")):
                url_stripped = f"https://{url_stripped}"

            parsed = urlparse(url_stripped)
            netloc = parsed.netloc

            # Extract port
            port: int | None = None
            if ":" in netloc:
                parts = netloc.split(":")
                netloc = parts[0]
                try:
                    port = int(parts[1])
                except ValueError:
                    pass

            domain = netloc.lower()

            # Check if IP address
            is_ip = bool(IP_REGEX.match(domain))

            if is_ip:
                return {
                    "domain": domain,
                    "subdomain": None,
                    "root_domain": domain,
                    "tld": None,
                    "is_ip": True,
                    "port": port,
                }

            # Split domain into parts
            parts = domain.split(".")

            if len(parts) < 2:
                return {
                    "domain": domain,
                    "subdomain": None,
                    "root_domain": domain,
                    "tld": None,
                    "is_ip": False,
                    "port": port,
                }

            tld = parts[-1]
            root_domain = ".".join(parts[-2:])

            subdomain = ".".join(parts[:-2]) if len(parts) > 2 else None

            return {
                "domain": domain,
                "subdomain": subdomain,
                "root_domain": root_domain,
                "tld": tld,
                "is_ip": False,
                "port": port,
            }

        except Exception as e:
            logger.warning(f"Failed to extract domain info from {url}: {e}")
            return {
                "domain": None,
                "subdomain": None,
                "root_domain": None,
                "tld": None,
                "is_ip": False,
                "port": None,
                "error": str(e),
            }


# =============================================================================
# Utility Functions
# =============================================================================


def is_internal_domain(domain: str) -> bool:
    """
    Check if domain is an internal/private domain.

    Args:
        domain: Domain to check

    Returns:
        True if domain is internal/private
    """
    internal_patterns = [
        "localhost",
        "*.local",
        "*.internal",
        "*.lan",
        "*.localdomain",
        "127.0.0.1",
        "0.0.0.0",
    ]

    domain = domain.lower()

    for pattern in internal_patterns:
        if pattern.startswith("*"):
            if domain.endswith(pattern[1:]):
                return True
        elif domain == pattern:
            return True

    # Check for private IP ranges
    if IP_REGEX.match(domain):
        parts = [int(p) for p in domain.split(".")]
        # 10.x.x.x
        if parts[0] == 10:
            return True
        # 172.16.x.x - 172.31.x.x
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return True
        # 192.168.x.x
        if parts[0] == 192 and parts[1] == 168:
            return True

    return False


def normalize_domain(domain: str) -> str:
    """
    Normalize a domain for consistent comparison.

    Args:
        domain: Domain to normalize

    Returns:
        Normalized domain string
    """
    # Lowercase
    domain = domain.lower().strip()

    # Remove trailing dot
    if domain.endswith("."):
        domain = domain[:-1]

    # Remove www. prefix for consistency
    if domain.startswith("www."):
        domain = domain[4:]

    return domain


# =============================================================================
# Type Exports
# =============================================================================

__all__ = [
    # Main class
    "DomainValidator",
    # Exceptions
    "DomainValidationError",
    # Utilities
    "is_internal_domain",
    "normalize_domain",
    # Constants
    "VALID_TLDS",
]
