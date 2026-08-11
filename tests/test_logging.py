from __future__ import annotations

import logging
import time

import pytest

from manga_animation.core.logging import StageTimer, get_gpu_memory_mb, get_logger, setup_logging


def test_get_logger_returns_named_logger():
    logger = get_logger("manga_animation.test")
    assert logger.name == "manga_animation.test"


def test_stage_timer_logs_start_and_done(caplog):
    logger = get_logger("manga_animation.test.timer_done")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with StageTimer("segmentation", logger, device="cpu", model="sam2-base") as timer:
            timer.input_shape = (1, 3, 64, 64)
            timer.output_shape = (1, 1, 64, 64)
            timer.confidence = 0.87

    messages = [r.message for r in caplog.records]
    assert any("status=start" in m and "stage=segmentation" in m for m in messages)
    done_messages = [m for m in messages if "status=done" in m]
    assert len(done_messages) == 1
    assert "confidence=0.87" in done_messages[0]
    assert "input_shape=(1, 3, 64, 64)" in done_messages[0]


def test_stage_timer_records_positive_elapsed_time(caplog):
    logger = get_logger("manga_animation.test.timer_elapsed")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with StageTimer("dummy", logger):
            time.sleep(0.01)

    done = next(r.message for r in caplog.records if "status=done" in r.message)
    elapsed = float(done.split("elapsed_s=")[1].split(" ")[0])
    assert elapsed >= 0.01


def test_stage_timer_logs_error_and_reraises(caplog):
    logger = get_logger("manga_animation.test.timer_error")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(ValueError, match="boom"):
            with StageTimer("dummy", logger):
                raise ValueError("boom")

    messages = [r.message for r in caplog.records]
    assert any("status=error" in m and "boom" in m for m in messages)
    assert not any("status=done" in m for m in messages)


def test_get_gpu_memory_mb_does_not_raise_without_torch():
    result = get_gpu_memory_mb()
    assert result is None or isinstance(result, float)


def test_setup_logging_sets_root_level():
    try:
        setup_logging(debug=True)
        assert logging.getLogger().level == logging.DEBUG
        setup_logging(debug=False)
        assert logging.getLogger().level == logging.INFO
    finally:
        setup_logging(debug=False)
