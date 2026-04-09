"""
Tests for PII scrubbing utilities.

Notes
-----
Covers every pattern documented in ``pii_scrubber.py``:
  email, IPv4, IPv6, UUID, JWT/Bearer token, sensitive query params.

Also covers: ``scrub_dict()``, edge cases (empty string, already-clean text),
and ordering guarantees (token before UUID).
"""
from __future__ import annotations

from slo_engine.utils.pii_scrubber import scrub, scrub_dict


class TestEmailScrubbing:
    """
    Tests for email address pattern scrubbing.

    Notes
    -----
    Verifies that standard email formats (simple, plus-tagged, dotted local
    part, multiple occurrences) are replaced with ``[email]``. Non-email
    strings containing ``@``-like patterns must not be mangled.
    """

    def test_simple_email_replaced(self):
        assert scrub("contact reviewer@example.com today") == "contact [email] today"

    def test_email_with_plus_tag(self):
        assert scrub("user+tag@sub.domain.org") == "[email]"

    def test_email_with_dots_in_local(self):
        assert scrub("first.last@company.io") == "[email]"

    def test_multiple_emails_in_one_string(self):
        result = scrub("from a@x.com to b@y.net")
        assert "[email]" in result
        assert "a@x.com" not in result
        assert "b@y.net" not in result

    def test_not_an_email_left_alone(self):
        assert scrub("version 2.0 release") == "version 2.0 release"


class TestIPv4Scrubbing:
    """
    Tests for IPv4 address pattern scrubbing.

    Notes
    -----
    Valid dotted-decimal IPv4 addresses (0-255 per octet) must be replaced
    with ``[ipv4]``. Out-of-range octets (e.g. 999) must not be matched.
    """

    def test_standard_ipv4(self):
        assert scrub("client 192.168.1.100 connected") == "client [ipv4] connected"

    def test_loopback_scrubbed(self):
        assert scrub("from 127.0.0.1") == "from [ipv4]"

    def test_edge_address_255(self):
        assert scrub("addr=255.255.255.0") == "addr=[ipv4]"

    def test_out_of_range_octet_not_matched(self):
        result = scrub("999.999.999.999")
        assert "[ipv4]" not in result

    def test_multiple_ipv4(self):
        result = scrub("src=10.0.0.1 dst=10.0.0.2")
        assert result.count("[ipv4]") == 2


class TestIPv6Scrubbing:
    """
    Tests for IPv6 address pattern scrubbing.

    Notes
    -----
    Full colon-delimited IPv6 addresses and the loopback ``::1`` must be
    replaced with ``[ipv6]``. IPv6 strings must not be incorrectly matched
    as email addresses.
    """

    def test_full_ipv6(self):
        result = scrub("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert "[ipv6]" in result

    def test_loopback_double_colon(self):
        result = scrub("peer ::1 joined")
        assert "[ipv6]" in result

    def test_ipv6_not_mangled_as_email(self):
        text = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        result = scrub(text)
        assert "[email]" not in result


class TestUUIDScrubbing:
    """
    Tests for UUID v1-v5 hyphenated string scrubbing.

    Notes
    -----
    Standard hyphenated UUIDs (upper and lower case) must be replaced with
    ``[uuid]``. Short hex strings that are not UUIDs (e.g. CSS colour codes)
    must not be affected.
    """

    def test_standard_uuid_v4(self):
        result = scrub("id=550e8400-e29b-41d4-a716-446655440000")
        assert "[uuid]" in result
        assert "550e8400" not in result

    def test_uppercase_uuid(self):
        result = scrub("550E8400-E29B-41D4-A716-446655440000")
        assert "[uuid]" in result

    def test_uuid_in_path(self):
        result = scrub("/api/v1/services/550e8400-e29b-41d4-a716-446655440000/slos")
        assert "[uuid]" in result
        assert "550e8400" not in result

    def test_non_uuid_hex_left_alone(self):
        result = scrub("color #a1b2c3")
        assert "[uuid]" not in result


class TestTokenScrubbing:
    """
    Tests for Bearer and JWT token scrubbing.

    Notes
    -----
    JWT tokens (three base64url segments separated by ``.``) must be replaced
    with ``[token]``. The token substitution must happen before UUID matching
    to prevent JWT base64url content from being partially matched as UUIDs.
    """

    _JWT = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )

    def test_jwt_replaced(self):
        result = scrub(f"Authorization: Bearer {self._JWT}")
        assert "[token]" in result
        assert "eyJhbGci" not in result

    def test_bare_jwt_replaced(self):
        result = scrub(self._JWT)
        assert "[token]" in result

    def test_token_replaced_before_uuid_check(self):
        """
        Verify that JWT tokens are scrubbed before UUID pattern matching.

        Notes
        -----
        JWT tokens contain base64url segments that could partially match the
        UUID pattern. Scrubbing tokens first prevents false UUID replacements
        within JWT payloads.
        """
        result = scrub(self._JWT)
        assert "[uuid]" not in result


class TestSensitiveParamScrubbing:
    """
    Tests for sensitive query parameter value scrubbing.

    Notes
    -----
    Parameter names ``password``, ``token``, ``api_key``, ``apikey``,
    ``secret``, and ``email`` are treated as sensitive. Their values are
    replaced with ``[redacted]``. Non-sensitive parameters must be unchanged.
    """

    def test_password_param(self):
        result = scrub("/login?password=s3cr3t&user=alice")
        assert "s3cr3t" not in result
        assert "password=[redacted]" in result
        assert "user=alice" in result

    def test_token_param(self):
        result = scrub("/api?token=abc123xyz")
        assert "abc123xyz" not in result
        assert "token=[redacted]" in result

    def test_api_key_param(self):
        result = scrub("/v1/data?api_key=MY_SECRET_KEY")
        assert "MY_SECRET_KEY" not in result
        assert "=[redacted]" in result

    def test_apikey_no_separator(self):
        result = scrub("/v1/data?apikey=MY_SECRET_KEY")
        assert "MY_SECRET_KEY" not in result

    def test_secret_param(self):
        result = scrub("secret=hunter2")
        assert "hunter2" not in result
        assert "secret=[redacted]" in result

    def test_email_param(self):
        result = scrub("email=alice@example.com&page=1")
        assert "alice@example.com" not in result
        assert "=[redacted]" in result
        assert "page=1" in result

    def test_case_insensitive_param_name(self):
        result = scrub("PASSWORD=letmein")
        assert "letmein" not in result


class TestEdgeCases:
    """
    Tests for edge cases in ``scrub()``.

    Notes
    -----
    An empty string must be returned unchanged. Clean text with no PII must
    pass through unmodified. Multiple PII types in a single string must all
    be scrubbed simultaneously.
    """

    def test_empty_string_returned_unchanged(self):
        assert scrub("") == ""

    def test_none_like_falsy_handled(self):
        assert scrub("") == ""

    def test_clean_text_returned_unchanged(self):
        text = "SLO target is 99.9% for payments service"
        assert scrub(text) == text

    def test_multiple_pii_types_in_one_string(self):
        text = "user reviewer@corp.com from 10.0.0.1 session 550e8400-e29b-41d4-a716-446655440000"
        result = scrub(text)
        assert "[email]" in result
        assert "[ipv4]" in result
        assert "[uuid]" in result
        assert "reviewer@corp.com" not in result
        assert "10.0.0.1" not in result
        assert "550e8400" not in result


class TestScrubDict:
    """
    Tests for the ``scrub_dict()`` shallow-copy scrubbing function.

    Notes
    -----
    String values must be scrubbed. Non-string values (int, float, bool)
    must be passed through unchanged. The original dict must not be mutated.
    """

    def test_string_values_scrubbed(self):
        d = {"reviewer": "alice@example.com", "decision": "approve"}
        result = scrub_dict(d)
        assert result["reviewer"] == "[email]"
        assert result["decision"] == "approve"

    def test_non_string_values_untouched(self):
        d = {"count": 42, "score": 0.95, "flag": True}
        result = scrub_dict(d)
        assert result == d

    def test_returns_new_dict_not_mutated(self):
        d = {"email": "x@y.com"}
        result = scrub_dict(d)
        assert d["email"] == "x@y.com"
        assert result["email"] == "[email]"

    def test_mixed_types(self):
        d = {
            "ip":    "192.168.0.1",
            "count": 100,
            "name":  "payments-service",
        }
        result = scrub_dict(d)
        assert result["ip"] == "[ipv4]"
        assert result["count"] == 100
        assert result["name"] == "payments-service"
