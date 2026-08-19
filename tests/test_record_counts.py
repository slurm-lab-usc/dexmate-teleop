"""Saved-episode counters shown in the Status card's Record row.

Counts come from the recorder's "EPISODE SAVED" stdout marker only — discards
never count — and are grouped by calendar day and task label, persisted to a
small JSON next to save_dir so totals survive backend restarts.
"""

from __future__ import annotations

from datetime import datetime

from omniteleop.app.backend import app_backend as ab
from omniteleop.app.backend.app_backend import TeleopApp

DAY = "08-14-2026"


def make_app(tmp_path, counts: dict | None = None) -> TeleopApp:
    app = TeleopApp.__new__(TeleopApp)
    app.save_dir = tmp_path
    app._record_counts_path = tmp_path / ".record_counts.json"
    app._record_counts = dict(counts or {})
    app._current_task_label = ""
    return app


def test_saved_event_counts_under_current_tag(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ab, "datetime", type("D", (), {"now": staticmethod(lambda: datetime(2026, 8, 14, 12, 0))}))
    app = make_app(tmp_path)
    app._current_task_label = "pick_cup"

    app._observe_recorder_line("💾 EPISODE SAVED - Episode 1")

    assert app.recording_status == "saved"
    assert app._record_counts[DAY] == {"pick_cup": 1}
    assert app._record_counts_status() == {
        "tag_label": "pick_cup",
        "tag_count": 1,
        "today_total": 1,
    }


def test_discard_never_counts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ab, "datetime", type("D", (), {"now": staticmethod(lambda: datetime(2026, 8, 14, 12, 0))}))
    app = make_app(tmp_path)
    app._current_task_label = "pick_cup"

    app._observe_recorder_line("🗑️ EPISODE DISCARDED - Episode 1")

    assert app.recording_status == "discarded"
    assert app._record_counts == {}
    assert app._record_counts_status() == {
        "tag_label": "pick_cup",
        "tag_count": 0,
        "today_total": 0,
    }


def test_no_tag_counts_roll_up_into_today_total(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ab, "datetime", type("D", (), {"now": staticmethod(lambda: datetime(2026, 8, 14, 12, 0))}))
    app = make_app(tmp_path)

    app._on_episode_saved()
    app._on_episode_saved()

    assert app._record_counts[DAY] == {"": 2}
    assert app._record_counts_status() == {
        "tag_label": "",
        "tag_count": 2,
        "today_total": 2,
    }


def test_tags_accumulate_independently(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ab, "datetime", type("D", (), {"now": staticmethod(lambda: datetime(2026, 8, 14, 12, 0))}))
    app = make_app(tmp_path)
    app._current_task_label = "pick_cup"
    app._on_episode_saved()
    app._current_task_label = "pour_water"
    app._on_episode_saved()
    app._on_episode_saved()

    assert app._record_counts[DAY] == {"pick_cup": 1, "pour_water": 2}
    status = app._record_counts_status()
    assert status["tag_count"] == 2
    assert status["today_total"] == 3


def test_counts_persist_and_reload(tmp_path) -> None:
    first = make_app(tmp_path)
    first._record_counts = {DAY: {"pick_cup": 4}}
    first._persist_record_counts()

    second = make_app(tmp_path)
    assert second._load_record_counts() == {DAY: {"pick_cup": 4}}


def test_corrupt_index_reloads_empty(tmp_path) -> None:
    (tmp_path / ".record_counts.json").write_text("not json{{")
    app = make_app(tmp_path)

    assert app._load_record_counts() == {}


def test_days_are_isolated(tmp_path) -> None:
    first = make_app(tmp_path)
    first._record_counts = {"08-13-2026": {"pick_cup": 7}}
    first._persist_record_counts()

    monkey_date = type("D", (), {"now": staticmethod(lambda: datetime(2026, 8, 14, 9, 0))})
    app = make_app(tmp_path)
    app._record_counts = app._load_record_counts()
    app._current_task_label = "pick_cup"

    # Not monkeypatching ab.datetime here — the fake is applied below.
    old_datetime = ab.datetime
    ab.datetime = monkey_date
    try:
        status = app._record_counts_status()
    finally:
        ab.datetime = old_datetime

    # Yesterday's pick_cup count must not leak into today.
    assert status["tag_count"] == 0
    assert status["today_total"] == 0
