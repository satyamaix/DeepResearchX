"""
URL Validation Utilities for SSRF Prevention.

Blocks requests to internal networks, localhost, and private IPs.
This module provides security controls to prevent Server-Side Request Forgery
(SSRF) attacks by validating URLs before they are fetched.

SSRF Attack Vectors Blocked:
- Localhost/loopback addresses (127.0.0.1, ::1, localhost)
- Private IP ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
- Link-local addresses (169.254.x.x)
- Internal hostnames (.local, .internal)
- Non-HTTP/HTTPS schemes (file://, gopher://, etc.)
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SSRFError(Exception):
    """Raised when URL fails SSRF validation."""

    pass


def validate_url(url: str, allowed_schemes: set[str] | None = None) -> str:
    """
    Validate URL is safe for server-side fetching.

    Performs comprehensive validation to prevent SSRF attacks by checking:
    1. URL scheme is allowed (default: http, https)
    2. Hostname is not empty
    3. Hostname is not localhost or loopback
    4. Hostname does not resolve to private/internal IPs
    5. Hostname is not an internal domain suffix

    Args:
        url: URL to validate
        allowed_schemes: Set of allowed schemes (default: {'http', 'https'})

    Returns:
        Validated URL (unchanged if valid)

    Raises:
        SSRFError: If URL is potentially dangerous

    Examples:
        >>> validate_url("https://example.com/page")
        'https://example.com/page'

        >>> validate_url("http://localhost/admin")
        Traceback (most recent call last):
            ...
        SSRFError: Localhost URLs not allowed

        >>> validate_url("file:///etc/passwd")
        Traceback (most recent call last):
            ...
        SSRFError: Scheme 'file' not allowed
    """
    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SSRFError(f"Invalid URL format: {e}")

    # Check scheme
    if parsed.scheme not in allowed_schemes:
        raise SSRFError(f"Scheme '{parsed.scheme}' not allowed")

    # Check for empty host
    if not parsed.hostname:
        raise SSRFError("No hostname in URL")

    hostname = parsed.hostname.lower()

    # Block localhost variants
    localhost_names = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}
    if hostname in localhost_names:
        raise SSRFError("Localhost URLs not allowed")

    # Block internal hostname suffixes
    internal_suffixes = (".local", ".internal", ".localhost", ".localdomain")
    if any(hostname.endswith(suffix) for suffix in internal_suffixes):
        raise SSRFError("Internal hostnames not allowed")

    # Block metadata service endpoints (cloud providers)
    metadata_hosts = {
        "169.254.169.254",  # AWS/GCP/Azure metadata
        "metadata.google.internal",  # GCP
        "metadata.internal",  # Generic cloud metadata
    }
    if hostname in metadata_hosts:
        raise SSRFError("Cloud metadata service URLs not allowed")

    # Check if hostname is an IP address directly
    try:
        ip_obj = ipaddress.ip_address(hostname)
        _check_ip_safety(ip_obj)
        return url  # IP was safe
    except ValueError:
        # Not an IP address, proceed with DNS resolution
        pass

    # Resolve hostname and check for private IPs
    try:
        # Get all IP addresses for hostname
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _type, _proto, _canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            _check_ip_safety(ip_obj)

    except socket.gaierror as e:
        # DNS resolution failed - log warning but allow
        # The actual request will fail with a more specific error
        logger.warning(f"DNS resolution failed for {hostname}: {e}")

    return url


def _check_ip_safety(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """
    Check if an IP address is safe for SSRF.

    Args:
        ip_obj: IP address object to check

    Raises:
        SSRFError: If IP is private, loopback, link-local, or reserved
    """
    if ip_obj.is_private:
        raise SSRFError(f"Private IP address not allowed: {ip_obj}")
    if ip_obj.is_loopback:
        raise SSRFError(f"Loopback address not allowed: {ip_obj}")
    if ip_obj.is_link_local:
        raise SSRFError(f"Link-local address not allowed: {ip_obj}")
    if ip_obj.is_reserved:
        raise SSRFError(f"Reserved address not allowed: {ip_obj}")
    if ip_obj.is_multicast:
        raise SSRFError(f"Multicast address not allowed: {ip_obj}")

    # Check for IPv4-mapped IPv6 addresses that might bypass checks
    if isinstance(ip_obj, ipaddress.IPv6Address):
        # Check IPv4-mapped addresses (::ffff:x.x.x.x)
        if ip_obj.ipv4_mapped:
            _check_ip_safety(ip_obj.ipv4_mapped)


def is_url_safe(url: str, allowed_schemes: set[str] | None = None) -> bool:
    """
    Check if URL is safe for server-side fetching without raising exception.

    This is a convenience function that wraps validate_url() for cases where
    a boolean check is preferred over exception handling.

    Args:
        url: URL to validate
        allowed_schemes: Set of allowed schemes (default: {'http', 'https'})

    Returns:
        True if URL is safe, False otherwise

    Examples:
        >>> is_url_safe("https://example.com")
        True
        >>> is_url_safe("http://localhost/admin")
        False
        >>> is_url_safe("file:///etc/passwd")
        False
    """
    try:
        validate_url(url, allowed_schemes)
        return True
    except SSRFError:
        return False


__all__ = [
    "SSRFError",
    "validate_url",
    "is_url_safe",
]
