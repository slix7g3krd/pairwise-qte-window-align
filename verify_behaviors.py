"""针对 QTE 判定窗口对齐六类缺陷的行为验证。

运行：

    python3 verify_behaviors.py

这些断言比 ``tests/test_qte_window_align.py`` 的冒烟契约更严格，分别覆盖：

1. 窗口按逻辑时间轴定义（不按渲染帧裁剪）；
2. 输入墙钟时间戳换算成逻辑帧；
3. 暂停冻结逻辑时钟但不清空输入队列；
4. 重试清空上一轮残留输入与判定结果；
5. 慢动作统一时间缩放；
6. 判定结果与帧率解耦（回放确定性）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gameplay.qte_window_align as qte


# --- 1. 窗口按逻辑时间轴定义 ----------------------------------------------


def test_window_is_logical_not_render_clipped() -> None:
    window = qte.window_timeline(duration_frames=12, tick_rate=60)
    assert window["start_frame"] == 0
    assert window["end_frame"] == 12
    assert window["logical_duration_ms"] == 200.0
    assert window["timeline"] == "logical"
    assert window["render_aligned"] is False

    # 同一逻辑窗口在慢动作下墙钟时长变长，但逻辑帧定义不变。
    slow = qte.window_timeline(duration_frames=12, tick_rate=60, scale=0.25)
    assert slow["duration_frames"] == 12
    assert slow["logical_duration_ms"] == 200.0
    assert slow["wall_duration_ms"] == 800.0

    # 窗口起点落在逻辑时间轴上，而不是被渲染帧网格推进。
    engine = qte.QteEngine(tick_rate=60)
    first = engine.open_window(wall_ms=0.0, duration_frames=12)
    assert (first.start_frame, first.end_frame) == (0, 12)
    later = engine.open_window(wall_ms=13.0, duration_frames=12)
    assert later.start_frame == 1  # 13ms -> 0.78 帧 -> 量化到 1 帧
    assert later.end_frame - later.start_frame == 12


# --- 2. 墙钟时间戳换算到逻辑帧 --------------------------------------------


def test_wall_timestamp_converted_to_logical_frame() -> None:
    engine = qte.QteEngine(tick_rate=60)
    converted = qte.input_timestamp(engine, wall_ms=100.0)
    assert converted["source"] == "wall_clock"
    assert converted["target"] == "logical_frame"
    assert converted["frame"] == 6
    assert converted["logical_ms"] == 100.0

    # 暂停期间任何墙钟时刻都映射到冻结的逻辑帧。
    engine.pause(100.0)
    assert qte.input_timestamp(engine, wall_ms=5000.0)["frame"] == 6
    engine.resume(5000.0)
    # 恢复后墙钟继续推进；5000ms 时逻辑仍停在 100ms。
    assert qte.input_timestamp(engine, wall_ms=5100.0)["logical_ms"] == 200.0

    # 慢动作下换算经过统一速率：墙钟 200ms 只推进 100ms 逻辑时间。
    slow = qte.QteEngine(tick_rate=60, scale=0.5)
    assert qte.input_timestamp(slow, wall_ms=200.0)["logical_ms"] == 100.0


# --- 3. 暂停冻结队列不清空 -------------------------------------------------


def test_pause_freezes_clock_without_dropping_queue() -> None:
    engine = qte.QteEngine(tick_rate=60)
    engine.open_window(0.0, duration_frames=12)
    engine.enqueue(40.0, button="A")
    engine.enqueue(60.0, button="B")

    frozen = qte.pause_freeze(engine, wall_ms=80.0)
    assert frozen["paused"] is True
    assert frozen["queue_size"] == 2
    assert frozen["queue_preserved"] is True

    # 暂停很久，队列仍在，逻辑时间不动。
    frame_before = engine.frame_at(80.0)
    assert engine.frame_at(10_000.0) == frame_before
    assert len(engine.queue) == 2

    # 暂停期间到达的输入也按冻结帧入队，恢复后照常判定。
    engine.enqueue(9_000.0, button="C")
    assert engine.queue[-1].frame == frame_before
    resumed = engine.resume(10_000.0)
    assert resumed["paused"] is False
    assert resumed["queue_size"] == 3

    result = engine.judge(10_100.0)
    assert len(result["judged"]) == 3
    assert all(event["grade"] is not None for event in result["judged"])


# --- 4. 重试清空上一轮判定 -------------------------------------------------


def test_retry_discards_previous_round() -> None:
    engine = qte.QteEngine(tick_rate=60)
    engine.open_window(0.0, duration_frames=12)
    engine.enqueue(100.0, button="A")
    first = engine.judge(100.0)
    assert first["outcome"] == "perfect"
    assert len(engine.judged) == 1

    # 再入队一条尚未判定的输入，然后重试。
    engine.enqueue(150.0, button="B")
    retried = qte.retry_reset(engine, wall_ms=300.0)
    assert retried["attempt"] == 2
    assert retried["dropped_inputs"] == 1
    assert retried["cleared_results"] == 1
    assert retried["queue_size"] == 0
    assert engine.judged == []

    # 旧回合的晚到输入不会混进新窗口。
    stale = qte.enqueue(engine, wall_ms=310.0, attempt=1)
    assert stale["accepted"] is False
    assert stale["status"] == "stale"
    assert engine.queue == []

    # 新窗口从重试时刻的逻辑帧起算，判定只包含新回合输入。
    assert engine.window.start_frame == 18  # 300ms -> 18 帧
    engine.enqueue(engine.window.center_frame * 1000.0 / 60.0, button="C")
    second = engine.judge(350.0)
    assert len(second["judged"]) == 1
    assert second["judged"][0]["attempt"] == 2
    assert second["outcome"] == "perfect"


# --- 5. 慢动作统一时间缩放 -------------------------------------------------


def test_time_scale_unified() -> None:
    engine = qte.QteEngine(tick_rate=60)
    engine.open_window(0.0, duration_frames=12)
    normal = engine.window.to_dict()

    applied = qte.time_scale(engine, 0.5)
    assert applied["scale"] == 0.5
    assert applied["window_frames_unchanged"] is True
    slow = engine.window.to_dict()
    assert slow["duration_frames"] == normal["duration_frames"] == 12
    assert slow["logical_duration_ms"] == normal["logical_duration_ms"] == 200.0
    assert slow["wall_duration_ms"] == 400.0

    # 输入换算与窗口用同一个缩放：墙钟 400ms 才走到窗口末尾。
    assert qte.input_timestamp(engine, wall_ms=400.0)["frame"] == 12
    # scale 回正常后墙钟窗口时长恢复，逻辑定义始终不变。
    qte.time_scale(engine, 1.0, 400.0)
    assert engine.window.wall_duration_ms == 200.0
    assert engine.window.duration_frames == 12


# --- 6. 判定结果与帧率解耦 -------------------------------------------------


def test_judgement_decoupled_from_frame_rate() -> None:
    result = qte.frame_decouple(render_fps=[24, 30, 60, 90, 120, 144])
    assert result["consistent"] is True
    digests = {entry["digest"] for entry in result["results"].values()}
    assert len(digests) == 1
    # 判定帧是逻辑帧，与渲染帧率无关。
    frames = {entry["frame"] for entry in result["results"].values()}
    assert frames == {6}

    repeat = qte.determinism(runs=6, render_fps=[24, 30, 60, 90, 120, 144])
    assert repeat["identical"] is True
    assert repeat["deterministic"] is True
    assert len({entry["digest"] for entry in repeat["results"]}) == 1


# --- 组合场景：暂停 + 慢动作 + 帧率抖动 -----------------------------------


def test_combined_replay_is_deterministic() -> None:
    baseline = qte.frame_decouple(
        render_fps=[24, 144],
        input_wall_ms=100.0,
        pause_at=20.0,
        resume_at=520.0,
        scale=0.5,
    )
    assert baseline["consistent"] is True
    assert qte.determinism(
        runs=4,
        render_fps=[30, 144],
        input_wall_ms=100.0,
        pause_at=20.0,
        resume_at=520.0,
        scale=0.5,
    )["identical"] is True


# --- 校验 / 兜底 / 审计 ----------------------------------------------------


def test_validation_and_fallback() -> None:
    ok = qte.validation(1)
    assert ok["ok"] is True
    assert ok["checks"]["window_logical"] is False

    bad = qte.validation(tick_rate=-5, scale=0, window_frames=0)
    assert bad["ok"] is False
    assert len(bad["errors"]) == 3

    repaired = qte.fallback(
        tick_rate=0,
        scale=-2,
        window_frames=0,
        lead_frames=-3,
        perfect_frames=-1,
        quantize="bogus",
    )
    assert repaired["repaired"] == 6
    assert repaired["config"]["tick_rate"] == 60.0
    assert repaired["config"]["scale"] == 1.0
    assert repaired["config"]["window_frames"] >= 1
    assert repaired["config"]["quantize"] == "nearest"


def test_audit_and_metrics_reflect_state() -> None:
    engine = qte.QteEngine(tick_rate=60)
    engine.open_window(0.0, duration_frames=12)
    engine.enqueue(100.0, button="A")
    engine.judge(100.0)

    audit = qte.audit(engine)
    # 判定后输入已离开队列，但判定历史仍保留。
    assert audit["queue_size"] == 0
    assert audit["judged_total"] == 1
    assert audit["grade_counts"] == {"perfect": 1}
    assert all(audit["objectives"].values())

    metrics = qte.metrics(engine)
    assert metrics["hit_rate"] == 1.0
    assert metrics["offset_frames"]["count"] == 1

    consumed = qte.consume(engine)
    assert len(consumed) == 1
    assert consumed[0]["grade"] == "perfect"
    assert qte.consume(engine) == []


def test_snapshot_round_trip_preserves_timeline() -> None:
    engine = qte.QteEngine(tick_rate=60)
    engine.open_window(0.0, duration_frames=12)
    engine.enqueue(100.0, button="A")
    engine.set_time_scale(0.5, 120.0)
    engine.pause(130.0)

    restored = qte.QteEngine.from_snapshot(engine.to_dict())
    assert restored.clock.paused is True
    assert restored.clock.scale == 0.5
    assert len(restored.queue) == 1
    assert restored.frame_at(1_000.0) == engine.frame_at(1_000.0)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:  # pragma: no cover - 失败路径
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
