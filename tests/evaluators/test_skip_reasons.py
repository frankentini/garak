# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for detector skip reason logging in evaluators."""

import logging
import uuid

import pytest

import garak.attempt
from garak import _config
from garak.evaluators.base import ThresholdEvaluator


@pytest.fixture(autouse=True)
def setup_config(tmp_path):
    """Set up minimal config for evaluator tests."""
    _config.system.narrow_output = False
    _config.system.verbose = 0
    _config.system.show_z = False
    report_path = tmp_path / "test.report.jsonl"
    _config.transient.reportfile = open(report_path, "w", buffering=1)
    _config.transient.hitlogfile = None
    _config.transient.run_id = uuid.uuid4()
    _config.run.generations = 1
    yield
    _config.transient.reportfile.close()
    if _config.transient.hitlogfile and not _config.transient.hitlogfile.closed:
        _config.transient.hitlogfile.close()


def _make_attempt(probe_name, detector_name, scores):
    """Create a minimal attempt with detector results."""
    a = garak.attempt.Attempt(
        probe_classname=probe_name,
        status=garak.attempt.ATTEMPT_STARTED,
    )
    a.prompt = garak.attempt.Turn(role="user", text="test prompt")
    a.outputs = [
        garak.attempt.Turn(role="assistant", text=f"output_{i}")
        for i in range(len(scores))
    ]
    a.detector_results[detector_name] = scores
    a.notes = {}
    return a


class TestSkipReasonLogging:
    """Test that skip reasons are logged when detectors produce no evaluable results."""

    def test_all_none_scores_logs_reason(self, caplog):
        """When all detector scores are None, log that results were None."""
        evaluator = ThresholdEvaluator()
        attempts = [_make_attempt("probes.test.Test", "test.Detector", [None, None])]

        with caplog.at_level(logging.INFO):
            evaluator.evaluate(attempts)

        assert any(
            "all 2 detector result(s) were None" in record.message
            for record in caplog.records
        ), f"Expected None-scores skip reason in log, got: {[r.message for r in caplog.records]}"

    def test_no_outputs_logs_reason(self, caplog):
        """When there are no outputs at all, log that there was nothing to evaluate."""
        evaluator = ThresholdEvaluator()
        attempts = [_make_attempt("probes.test.Test", "test.Detector", [])]

        with caplog.at_level(logging.INFO):
            evaluator.evaluate(attempts)

        assert any(
            "no outputs to evaluate" in record.message
            for record in caplog.records
        ), f"Expected no-outputs skip reason in log, got: {[r.message for r in caplog.records]}"

    def test_passing_scores_no_skip_log(self, caplog):
        """When scores are present and valid, no skip reason should be logged."""
        evaluator = ThresholdEvaluator()
        attempts = [_make_attempt("probes.test.Test", "test.Detector", [0.0, 0.1])]

        with caplog.at_level(logging.INFO):
            evaluator.evaluate(attempts)

        skip_messages = [
            r.message for r in caplog.records if "skipped" in r.message.lower()
        ]
        assert len(skip_messages) == 0, f"Unexpected skip log: {skip_messages}"


class TestSkipReasonDisplay:
    """Test that skip reasons appear in printed output."""

    def test_wide_format_shows_skip_reason(self, capsys):
        """Wide format should display skip reason when detector is skipped."""
        evaluator = ThresholdEvaluator()
        attempts = [_make_attempt("probes.test.Test", "test.Detector", [None])]

        _config.system.narrow_output = False
        evaluator.evaluate(attempts)

        captured = capsys.readouterr()
        assert "reason:" in captured.out, (
            f"Expected skip reason in wide output, got: {captured.out}"
        )

    def test_narrow_format_shows_skip_reason(self, capsys):
        """Narrow format should display skip reason when detector is skipped."""
        evaluator = ThresholdEvaluator()
        attempts = [_make_attempt("probes.test.Test", "test.Detector", [None])]

        _config.system.narrow_output = True
        evaluator.evaluate(attempts)

        captured = capsys.readouterr()
        assert "skip reason:" in captured.out, (
            f"Expected skip reason in narrow output, got: {captured.out}"
        )

    def test_no_skip_reason_on_pass(self, capsys):
        """When scores lead to PASS, no skip reason should be displayed."""
        evaluator = ThresholdEvaluator()
        attempts = [_make_attempt("probes.test.Test", "test.Detector", [0.0])]

        _config.system.narrow_output = False
        evaluator.evaluate(attempts)

        captured = capsys.readouterr()
        assert "reason:" not in captured.out, (
            f"Unexpected skip reason in output: {captured.out}"
        )
