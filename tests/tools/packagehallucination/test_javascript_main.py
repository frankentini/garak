"""Tests for tools/packagehallucination/javascript/main.py

Validates that the npm package dataset builder correctly handles
404 errors, missing dates, and only writes valid ISO date strings
to the output TSV.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import requests

# Add the tools directory to the path so we can import the module
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "packagehallucination", "javascript"),
)
from main import get_package_first_seen, _is_retryable, TIME_FORMAT  # noqa: E402


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------
class TestIsRetryable:
    def test_404_is_not_retryable(self):
        resp = MagicMock()
        resp.status_code = 404
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is False

    def test_400_is_not_retryable(self):
        resp = MagicMock()
        resp.status_code = 400
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is False

    def test_429_is_retryable(self):
        resp = MagicMock()
        resp.status_code = 429
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is True

    def test_500_is_retryable(self):
        resp = MagicMock()
        resp.status_code = 500
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is True

    def test_connection_error_is_retryable(self):
        exc = requests.exceptions.ConnectionError("conn refused")
        assert _is_retryable(exc) is True


# ---------------------------------------------------------------------------
# get_package_first_seen
# ---------------------------------------------------------------------------
class TestGetPackageFirstSeen:
    @patch("main.requests.get")
    def test_valid_package_returns_formatted_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "time": {"created": "2020-06-15T10:30:00.000Z"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = get_package_first_seen("valid-pkg")
        assert result is not None
        assert "2020" in result
        assert "Error" not in result

    @patch("main.requests.get")
    def test_404_returns_none_not_error_string(self, mock_get):
        """Regression: #1568 - error strings must not leak into the date field."""
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error", response=resp
        )
        mock_get.return_value = resp

        result = get_package_first_seen("nonexistent-pkg")
        assert result is None

    @patch("main.requests.get")
    def test_missing_created_field_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"time": {}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = get_package_first_seen("no-date-pkg")
        assert result is None

    @patch("main.requests.get")
    def test_connection_error_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        result = get_package_first_seen("unreachable-pkg")
        assert result is None

    @patch("main.requests.get")
    def test_result_never_contains_error_prefix(self, mock_get):
        """The old code stored 'Error: ...' as the date. Ensure that never happens."""
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=resp
        )
        mock_get.return_value = resp

        result = get_package_first_seen("server-error-pkg")
        # Result should be None, never a string starting with "Error:"
        if result is not None:
            assert not result.startswith("Error:")
