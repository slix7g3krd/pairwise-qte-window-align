"""快速反应事件（QTE）判定窗口对齐。

本模块把 QTE 判定完全建立在**逻辑时间轴（逻辑帧）**上：渲染帧率、机器性能、
线程调度抖动都不参与判定结果。针对原实现的六类问题，分别修复为：

1. 窗口按逻辑时间轴定义（``window_timeline``）
   窗口是逻辑帧区间 ``[start_frame, start_frame + duration_frames)``，
   不按渲染帧边界裁剪。逻辑时长固定，墙钟时长只由
   ``逻辑时长 / time_scale`` 推导。

2. 输入时间戳换算到逻辑帧（``input_timestamp``）
   采样线程只带墙钟时间戳；:class:`LogicalClock` 用一组「时间片段」
   （:class:`_Segment`）记录墙钟到逻辑时间的分段线性映射。暂停、变速都会
   切段，所以任意墙钟时间都能确定性地换算成逻辑帧并量化到帧边界。

3. 暂停冻结队列不清空（``pause_freeze``）
   暂停只把当前时间片段的速率置零，**不动输入队列**。暂停期间到达的输入
   按冻结的逻辑帧入队，恢复后照常判定。

4. 重试清空上一轮判定（``retry_reset``）
   重试提升 ``attempt``、清空上一轮未判定输入与已判定结果、重置窗口；
   上一轮的时间戳即使晚到，也会因 ``attempt`` 不匹配被标记为 stale，
   不会混进新窗口。

5. 慢动作统一时间缩放（``time_scale``）
   时间缩放作用在逻辑时钟上：逻辑窗口帧数不变，墙钟时长按 ``1 / scale``
   缩放，输入换算同样经过该速率，因此慢动作下判定与时序一致。

6. 判定结果与帧率解耦（``frame_decouple`` / ``determinism``）
   判定只比较逻辑帧。同一份回放录像在 24/30/60/90/120/144 fps 以及加入
   处理抖动时，得到完全相同的判定摘要。

模块中的顶层函数都同时接受：

* 一个 :class:`QteEngine` 实例（推荐，便于维护状态）；
* 一个配置/快照 ``dict``；
* 标量参数（无状态的一次性调用，内部使用时间原点为 0 的确定性引擎）。

所有顶层函数都返回非 ``None`` 的可序列化结果（``consume`` 返回 ``list``）。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "LOGICAL_FPS",
    "DEFAULT_WINDOW_FRAMES",
    "DEFAULT_LEAD_FRAMES",
    "DEFAULT_PERFECT_FRAMES",
    "DEFAULT_SCALE",
    "GRADES",
    "LogicalClock",
    "WindowTimeline",
    "InputEvent",
    "QteEngine",
    "window_timeline",
    "input_timestamp",
    "pause_freeze",
    "retry_reset",
    "time_scale",
    "frame_decouple",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "window_state",
    "metrics",
    "judge",
    "enqueue",
    "consume",
]

#: 逻辑时间轴默认帧率。判定窗口永远以逻辑帧为单位，与渲染帧率无关。
LOGICAL_FPS = 60.0
#: 默认窗口长度（逻辑帧数，半开区间）。
DEFAULT_WINDOW_FRAMES = 12
#: 默认提前量：窗口相对「打开时所在逻辑帧」再往后推迟的帧数。
DEFAULT_LEAD_FRAMES = 0
#: 默认完美判定带：窗口中心 ± perfect_frames 内算 perfect。
DEFAULT_PERFECT_FRAMES = 2
#: 默认时间缩放。
DEFAULT_SCALE = 1.0

GRADES = ("perfect", "good", "early", "late", "miss")

EPS = 1e-9

_QUANTIZERS = ("nearest", "floor", "ceil")


def _engine_kwargs(options: Mapping[str, Any]) -> dict:
    """把配置字典里的常见别名映射到 :class:`QteEngine` 的构造参数名。"""
    kwargs: dict[str, Any] = {}
    tick = options.get("tick_rate", options.get("logical_fps", options.get("fps")))
    if tick is not None:
        kwargs["tick_rate"] = tick
    scale = options.get("scale", options.get("time_scale"))
    if scale is not None:
        kwargs["scale"] = scale
    frames = options.get("window_frames", options.get("duration_frames"))
    if frames is not None:
        kwargs["window_frames"] = frames
    for key in ("lead_frames", "perfect_frames", "quantize", "wall_start"):
        if options.get(key) is not None:
            kwargs[key] = options[key]
    return kwargs


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: Any, default: float = 0.0) -> float:
    """尽量转成有限浮点数；布尔、None、不可转对象回退到 ``default``。"""
    if isinstance(value, bool) or value is None:
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _int(value: Any, default: int = 0) -> int:
    return int(round(_num(value, default)))


def _round(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    """稳定浮点输出，避免快照里出现 ``0.30000000000000004`` 这类噪声。"""
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), ndigits)


def _quantize_frame(frame: float, mode: str) -> int:
    if mode == "floor":
        return int(math.floor(frame + EPS))
    if mode == "ceil":
        return int(math.ceil(frame - EPS))
    return int(round(frame))


def _config_from_mapping(data: Mapping[str, Any]) -> dict:
    """从配置字典里读出引擎构造参数，兼容常见别名。"""
    cfg: dict[str, Any] = {}
    tick = data.get("tick_rate", data.get("logical_fps", data.get("fps")))
    if tick is not None:
        cfg["tick_rate"] = tick
    scale = data.get("scale", data.get("time_scale"))
    if scale is not None:
        cfg["scale"] = scale
    frames = data.get("window_frames", data.get("duration_frames"))
    if frames is not None:
        cfg["window_frames"] = frames
    for key in ("lead_frames", "perfect_frames", "quantize", "wall_start"):
        if key in data:
            cfg[key] = data[key]
    return cfg


def _resolve(args: tuple, kwargs: dict) -> tuple["QteEngine", list[float], dict, bool]:
    """从位置参数/关键字参数里解析出引擎。

    返回 ``(engine, numbers, options, created)``：

    * ``QteEngine`` 实例直接使用；
    * ``dict`` 视为配置或快照并构造引擎；
    * 数字收集到 ``numbers``，由各入口按自身语义解释；
    * 没有引擎时用关键字里的配置新建一个（``created=True``）。
    """
    options = dict(kwargs)
    engine: Optional[QteEngine] = None
    numbers: list[float] = []

    explicit = options.pop("engine", None)
    if isinstance(explicit, QteEngine):
        engine = explicit

    for arg in args:
        if isinstance(arg, QteEngine):
            engine = arg
        elif isinstance(arg, Mapping):
            if engine is None:
                engine = QteEngine.from_mapping(arg)
        elif _is_number(arg):
            numbers.append(float(arg))

    created = False
    if engine is None:
        cfg = _engine_kwargs(options)
        engine = QteEngine(**cfg)
        created = True
    return engine, numbers, options, created


_RAW_CONFIG_KEYS = (
    "tick_rate",
    "logical_fps",
    "fps",
    "scale",
    "time_scale",
    "window_frames",
    "duration_frames",
    "lead_frames",
    "perfect_frames",
    "quantize",
    "wall_start",
)


def _raw_config(args: tuple, kwargs: Mapping[str, Any]) -> dict:
    """收集调用方给出的**原始**配置（未经过引擎归一化），供校验/兜底使用。"""
    raw: dict[str, Any] = {}
    for arg in args:
        if isinstance(arg, Mapping):
            for key in _RAW_CONFIG_KEYS:
                if key in arg:
                    raw[key] = arg[key]
    for key in _RAW_CONFIG_KEYS:
        if key in kwargs:
            raw[key] = kwargs[key]
    return raw


# ---------------------------------------------------------------------------
# 逻辑时钟
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Segment:
    """墙钟到逻辑时间的一段线性映射。

    ``logical_ms(wall) = logical_ms + (wall - wall_ms) * rate``。
    暂停时 ``rate == 0``，于是暂停期间任何墙钟时刻都映射到同一个逻辑时刻。
    """

    wall_ms: float
    logical_ms: float
    rate: float


@dataclass
class LogicalClock:
    """按「时间片段」维护墙钟到逻辑时间的映射。

    ``rate`` 是每墙钟毫秒推进的逻辑毫秒数，也就是 ``time_scale``。
    暂停/恢复/变速都只是追加一个片段，不会回写历史片段的换算结果。
    """

    tick_rate: float = LOGICAL_FPS
    scale: float = DEFAULT_SCALE
    wall_start: float = 0.0

    def __post_init__(self) -> None:
        self.tick_rate = max(EPS, _num(self.tick_rate, LOGICAL_FPS))
        self.scale = _num(self.scale, DEFAULT_SCALE)
        if self.scale <= 0:
            self.scale = DEFAULT_SCALE
        self.wall_start = _num(self.wall_start, 0.0)
        self.paused = False
        self._segments: list[_Segment] = [
            _Segment(self.wall_start, 0.0, self.scale)
        ]

    # -- 映射 ---------------------------------------------------------

    def _segment_at(self, wall_ms: float) -> _Segment:
        chosen = self._segments[0]
        for segment in self._segments:
            if segment.wall_ms <= wall_ms + EPS:
                chosen = segment
            else:
                break
        return chosen

    def to_logical_ms(self, wall_ms: Any) -> float:
        """把墙钟时间戳换算成逻辑时间（毫秒）。"""
        wall = _num(wall_ms, self.wall_start)
        segment = self._segment_at(wall)
        return segment.logical_ms + (wall - segment.wall_ms) * segment.rate

    def to_frame(self, wall_ms: Any, quantize: str = "nearest") -> int:
        """把墙钟时间戳换算成逻辑帧（量化到帧边界）。"""
        logical_ms = self.to_logical_ms(wall_ms)
        frame = logical_ms * self.tick_rate / 1000.0
        return _quantize_frame(frame, quantize if quantize in _QUANTIZERS else "nearest")

    def frame_float(self, wall_ms: Any) -> float:
        return self.to_logical_ms(wall_ms) * self.tick_rate / 1000.0

    # -- 变速 / 暂停 --------------------------------------------------

    def _push(self, wall_ms: float, rate: float) -> None:
        wall = _num(wall_ms, self.wall_start)
        last = self._segments[-1]
        if wall < last.wall_ms - EPS:
            # 乱序时间戳：钳到最后一个片段起点，保持映射单调。
            wall = last.wall_ms
        logical = self.to_logical_ms(wall)
        if abs(wall - last.wall_ms) <= EPS:
            self._segments[-1] = _Segment(wall, logical, rate)
        else:
            self._segments.append(_Segment(wall, logical, rate))

    def set_scale(self, scale: Any, wall_ms: Any = None) -> float:
        """在 ``wall_ms`` 处切换时间缩放；返回生效后的缩放值。"""
        value = _num(scale, self.scale)
        if value <= 0:
            value = self.scale
        self.scale = value
        self._push(
            _num(wall_ms, self._segments[-1].wall_ms),
            0.0 if self.paused else value,
        )
        return value

    def pause(self, wall_ms: Any) -> None:
        if self.paused:
            return
        self.paused = True
        self._push(_num(wall_ms, self._segments[-1].wall_ms), 0.0)

    def resume(self, wall_ms: Any) -> None:
        if not self.paused:
            return
        self.paused = False
        self._push(_num(wall_ms, self._segments[-1].wall_ms), self.scale)

    def frozen(self) -> bool:
        return self.paused

    def to_dict(self) -> dict:
        return {
            "tick_rate": _round(self.tick_rate),
            "scale": _round(self.scale),
            "wall_start": _round(self.wall_start),
            "paused": self.paused,
            "segments": [
                {
                    "wall_ms": _round(s.wall_ms),
                    "logical_ms": _round(s.logical_ms),
                    "rate": _round(s.rate),
                }
                for s in self._segments
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogicalClock":
        clock = cls(
            tick_rate=data.get("tick_rate", LOGICAL_FPS),
            scale=data.get("scale", DEFAULT_SCALE),
            wall_start=data.get("wall_start", 0.0),
        )
        clock.paused = bool(data.get("paused", False))
        segments = data.get("segments")
        if isinstance(segments, Sequence) and segments:
            restored = [
                _Segment(
                    _num(s.get("wall_ms")),
                    _num(s.get("logical_ms")),
                    _num(s.get("rate")),
                )
                for s in segments
                if isinstance(s, Mapping)
            ]
            if restored:
                clock._segments = restored
        return clock


# ---------------------------------------------------------------------------
# 窗口与输入事件
# ---------------------------------------------------------------------------


@dataclass
class WindowTimeline:
    """定义在逻辑时间轴上的判定窗口（半开区间）。"""

    start_frame: int
    duration_frames: int
    tick_rate: float = LOGICAL_FPS
    scale: float = DEFAULT_SCALE
    attempt: int = 1
    opened_wall_ms: float = 0.0
    perfect_frames: int = DEFAULT_PERFECT_FRAMES

    def __post_init__(self) -> None:
        self.start_frame = int(self.start_frame)
        self.duration_frames = max(1, int(self.duration_frames))
        self.tick_rate = max(EPS, _num(self.tick_rate, LOGICAL_FPS))
        self.scale = max(EPS, _num(self.scale, DEFAULT_SCALE))
        self.attempt = max(1, int(self.attempt))
        self.perfect_frames = max(0, int(self.perfect_frames))

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.duration_frames

    @property
    def center_frame(self) -> float:
        return self.start_frame + self.duration_frames / 2.0

    @property
    def logical_duration_ms(self) -> float:
        return self.duration_frames * 1000.0 / self.tick_rate

    @property
    def wall_duration_ms(self) -> float:
        """渲染/墙钟视角下的窗口时长；慢动作时按 ``1 / scale`` 变长。"""
        return self.logical_duration_ms / self.scale

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame

    def offset_frames(self, frame: int) -> int:
        """相对窗口中心的帧偏移（负数偏早，正数偏晚）。"""
        return int(round(frame - self.center_frame))

    def grade(self, frame: int) -> str:
        if frame < self.start_frame:
            return "early"
        if frame >= self.end_frame:
            return "late"
        return "perfect" if abs(frame - self.center_frame) <= self.perfect_frames else "good"

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "center_frame": _round(self.center_frame),
            "tick_rate": _round(self.tick_rate),
            "scale": _round(self.scale),
            "attempt": self.attempt,
            "opened_wall_ms": _round(self.opened_wall_ms),
            "perfect_frames": self.perfect_frames,
            "logical_duration_ms": _round(self.logical_duration_ms),
            "wall_duration_ms": _round(self.wall_duration_ms),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WindowTimeline":
        return cls(
            start_frame=data.get("start_frame", 0),
            duration_frames=data.get("duration_frames", DEFAULT_WINDOW_FRAMES),
            tick_rate=data.get("tick_rate", LOGICAL_FPS),
            scale=data.get("scale", DEFAULT_SCALE),
            attempt=data.get("attempt", 1),
            opened_wall_ms=data.get("opened_wall_ms", 0.0),
            perfect_frames=data.get("perfect_frames", DEFAULT_PERFECT_FRAMES),
        )


@dataclass
class InputEvent:
    """一条输入。墙钟时间戳是采样线程给的原始值，逻辑帧是判定依据。"""

    seq: int
    wall_ms: float
    logical_ms: float
    frame: int
    button: Optional[str]
    attempt: int
    status: str = "queued"  # queued | judged | discarded | stale
    grade: Optional[str] = None
    offset_frames: Optional[int] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "wall_ms": _round(self.wall_ms),
            "logical_ms": _round(self.logical_ms),
            "frame": self.frame,
            "button": self.button,
            "attempt": self.attempt,
            "status": self.status,
            "grade": self.grade,
            "offset_frames": self.offset_frames,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputEvent":
        return cls(
            seq=int(data.get("seq", 0)),
            wall_ms=_num(data.get("wall_ms")),
            logical_ms=_num(data.get("logical_ms")),
            frame=_int(data.get("frame")),
            button=data.get("button"),
            attempt=int(data.get("attempt", 1)),
            status=str(data.get("status", "queued")),
            grade=data.get("grade"),
            offset_frames=data.get("offset_frames"),
            reason=data.get("reason"),
        )


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class QteEngine:
    """QTE 判定引擎：逻辑时钟 + 窗口 + 输入队列 + 判定结果。"""

    def __init__(
        self,
        tick_rate: Any = LOGICAL_FPS,
        scale: Any = DEFAULT_SCALE,
        window_frames: Any = DEFAULT_WINDOW_FRAMES,
        lead_frames: Any = DEFAULT_LEAD_FRAMES,
        perfect_frames: Any = DEFAULT_PERFECT_FRAMES,
        quantize: str = "nearest",
        wall_start: Any = 0.0,
        attempt: int = 1,
    ) -> None:
        tick = _num(tick_rate, LOGICAL_FPS)
        if tick <= 0:
            tick = LOGICAL_FPS
        self.tick_rate = tick
        self.scale = _num(scale, DEFAULT_SCALE)
        if self.scale <= 0:
            self.scale = DEFAULT_SCALE
        self.window_frames = max(1, _int(window_frames, DEFAULT_WINDOW_FRAMES))
        self.lead_frames = max(0, _int(lead_frames, DEFAULT_LEAD_FRAMES))
        self.perfect_frames = max(0, _int(perfect_frames, DEFAULT_PERFECT_FRAMES))
        self.quantize = quantize if quantize in _QUANTIZERS else "nearest"
        self.wall_start = _num(wall_start, 0.0)

        self.clock = LogicalClock(
            tick_rate=self.tick_rate, scale=self.scale, wall_start=self.wall_start
        )
        self.attempt = max(1, int(attempt))
        self.attempt_wall_ms = self.wall_start
        self.window: Optional[WindowTimeline] = None
        self.queue: list[InputEvent] = []
        self.judged: list[dict] = []
        self.pending: list[dict] = []
        self.rejected: list[InputEvent] = []
        self.last_wall_ms = self.wall_start
        self._seq = 0

        # 累计计数，便于审计
        self.total_enqueued = 0
        self.total_discarded = 0
        self.total_judged = 0

    # -- 兼容属性 -----------------------------------------------------

    @property
    def logical_fps(self) -> float:
        return self.tick_rate

    @property
    def time_scale(self) -> float:
        return self.clock.scale

    # -- 时间推进 -----------------------------------------------------

    def advance(self, wall_ms: Any) -> float:
        wall = _num(wall_ms, self.last_wall_ms)
        if wall > self.last_wall_ms:
            self.last_wall_ms = wall
        return self.last_wall_ms

    def _wall(self, wall_ms: Any) -> float:
        """取得一个墙钟时刻。

        显式给出的时间戳按它本身解释（回放/晚到事件也必须是确定性的），
        只把 ``last_wall_ms`` 推到大值用于「当前时刻」查询；不会把输入
        自带的墙钟时间戳悄悄换成渲染循环轮询到的时刻。
        """
        if wall_ms is None:
            return self.last_wall_ms
        wall = _num(wall_ms, self.last_wall_ms)
        if wall > self.last_wall_ms:
            self.last_wall_ms = wall
        return wall

    #: 公开别名：显式时间戳不向前钳制，见 :meth:`_wall`。
    resolve_wall = _wall

    def frame_at(self, wall_ms: Any = None) -> int:
        wall = self._wall(wall_ms)
        return self.clock.to_frame(wall, self.quantize)

    # -- 窗口 ---------------------------------------------------------

    def open_window(
        self,
        wall_ms: Any = None,
        duration_frames: Any = None,
        lead_frames: Any = None,
    ) -> WindowTimeline:
        """在逻辑时间轴上开一个窗口。

        ``start_frame = frame_at(wall) + lead_frames``，长度固定为
        ``duration_frames`` 个逻辑帧。窗口不随渲染帧对齐/裁剪。
        """
        wall = self._wall(wall_ms)
        lead = self.lead_frames if lead_frames is None else max(0, _int(lead_frames))
        duration = self.window_frames if duration_frames is None else _int(
            duration_frames, self.window_frames
        )
        start = self.frame_at(wall) + lead
        self.window = WindowTimeline(
            start_frame=start,
            duration_frames=max(1, duration),
            tick_rate=self.tick_rate,
            scale=self.clock.scale,
            attempt=self.attempt,
            opened_wall_ms=wall,
            perfect_frames=self.perfect_frames,
        )
        return self.window

    def window_state(self, wall_ms: Any = None) -> dict:
        wall = self._wall(wall_ms)
        frame = self.frame_at(wall)
        paused = self.clock.frozen()
        if self.window is None:
            state = "idle"
            remaining = 0
            elapsed = 0
            window = None
        else:
            window = self.window
            elapsed = max(0, frame - window.start_frame)
            remaining = max(0, window.end_frame - frame)
            if frame < window.start_frame:
                state = "pending"
            elif frame < window.end_frame:
                state = "open"
            else:
                state = "expired"
        return {
            "state": state,
            "paused": paused,
            "frozen_frame": frame if paused else None,
            "current_frame": frame,
            "current_logical_ms": _round(self.clock.to_logical_ms(wall)),
            "remaining_frames": remaining,
            "elapsed_frames": elapsed,
            "window": window.to_dict() if window is not None else None,
            "attempt": self.attempt,
            "queue_size": len(self.queue),
            "time_scale": _round(self.clock.scale),
            "tick_rate": _round(self.tick_rate),
        }

    # -- 时间缩放 / 暂停 ----------------------------------------------

    def set_time_scale(self, scale: Any, wall_ms: Any = None) -> dict:
        wall = self._wall(wall_ms)
        before = self.clock.scale
        after = self.clock.set_scale(scale, wall)
        if self.window is not None:
            # 窗口的逻辑帧定义不变，只刷新它对外展示的墙钟时长。
            self.window.scale = after
        return {
            "applied": after,
            "previous": _round(before),
            "changed": abs(after - before) > EPS,
            "paused": self.clock.frozen(),
            "window_wall_duration_ms": (
                _round(self.window.wall_duration_ms) if self.window else None
            ),
            "window_logical_duration_ms": (
                _round(self.window.logical_duration_ms) if self.window else None
            ),
        }

    def pause(self, wall_ms: Any = None) -> dict:
        """冻结逻辑时钟；**不清空输入队列**。"""
        wall = self._wall(wall_ms)
        frozen_frame = self.frame_at(wall)
        self.clock.pause(wall)
        return {
            "paused": True,
            "frozen_frame": frozen_frame,
            "queue_size": len(self.queue),
            "queue_preserved": True,
            "queue": [event.to_dict() for event in self.queue],
        }

    def resume(self, wall_ms: Any = None) -> dict:
        wall = self._wall(wall_ms)
        self.clock.resume(wall)
        return {
            "paused": False,
            "resumed_frame": self.frame_at(wall),
            "queue_size": len(self.queue),
            "queue_preserved": True,
        }

    # -- 输入 ---------------------------------------------------------

    def enqueue(
        self,
        wall_ms: Any,
        button: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> InputEvent:
        """按墙钟时间戳把输入换算到逻辑帧并入队。

        如果事件携带的 ``attempt`` 与当前回合不一致（上一轮残留/晚到事件），
        事件被标记为 ``stale`` 并放到 ``rejected``，不会进入当前窗口。
        """
        wall = self._wall(wall_ms)
        event_attempt = self.attempt if attempt is None else int(attempt)
        logical_ms = self.clock.to_logical_ms(wall)
        frame = self.clock.to_frame(wall, self.quantize)
        self._seq += 1

        reason = None
        if event_attempt < self.attempt:
            reason = "stale_attempt"
        elif wall < self.attempt_wall_ms - EPS:
            # 采样线程在上一轮开始之前抓到的输入：即使没带 attempt 也会混进
            # 新窗口，按墙钟时间拒绝。
            reason = "predates_attempt"
        event = InputEvent(
            seq=self._seq,
            wall_ms=wall,
            logical_ms=logical_ms,
            frame=frame,
            button=button,
            attempt=event_attempt,
            status="queued" if reason is None else "stale",
            reason=reason,
        )
        if reason is not None:
            self.rejected.append(event)
            return event
        self.queue.append(event)
        self.total_enqueued += 1
        return event

    def judge(self, wall_ms: Any = None) -> dict:
        """判定当前回合所有尚未判定的输入。

        判定只比较逻辑帧，与渲染帧率、处理时机无关。重复调用不会重复判定。
        """
        wall = self._wall(wall_ms)
        newly_judged: list[dict] = []
        window = self.window
        for event in self.queue:
            if event.status != "queued":
                continue
            if event.attempt != self.attempt:
                event.status = "discarded"
                continue
            if window is None or window.attempt != event.attempt:
                event.status = "discarded"
                continue
            grade = window.grade(event.frame)
            event.grade = grade
            event.offset_frames = window.offset_frames(event.frame)
            event.status = "judged"
            self.total_judged += 1
            result = {
                "attempt": event.attempt,
                "seq": event.seq,
                "button": event.button,
                "wall_ms": _round(event.wall_ms),
                "logical_ms": _round(event.logical_ms),
                "frame": event.frame,
                "grade": grade,
                "offset_frames": event.offset_frames,
                "window_start": window.start_frame if window else None,
                "window_end": window.end_frame if window else None,
            }
            self.judged.append(result)
            self.pending.append(result)
            newly_judged.append(result)
        # 队列只保留「已入队但尚未判定」的输入。
        self.queue = [
            event for event in self.queue if event.status == "queued"
        ]

        best = self._best_result()
        outcome = best["grade"] if best else ("miss" if newly_judged or self.judged else "none")
        if outcome not in GRADES:
            outcome = "none"
        return {
            "attempt": self.attempt,
            "window": window.to_dict() if window is not None else None,
            "judged": newly_judged,
            "pending": len(self.pending),
            "best": best,
            "outcome": outcome,
            "wall_ms": _round(wall),
            "frame": self.frame_at(wall),
            "paused": self.clock.frozen(),
        }

    def _best_result(self) -> Optional[dict]:
        """当前回合的最优判定：perfect > good > late > early。"""
        rank = {"perfect": 0, "good": 1, "late": 2, "early": 3}
        candidates = [
            result
            for result in self.judged
            if result.get("attempt") == self.attempt
            and result.get("grade") in ("perfect", "good")
        ]
        if not candidates:
            candidates = [
                result
                for result in self.judged
                if result.get("attempt") == self.attempt and result.get("grade")
            ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda r: (
                rank.get(r.get("grade"), 9),
                r.get("frame", 0),
                r.get("seq", 0),
            ),
        )

    def consume(self) -> list[dict]:
        """取走尚未被消费的判定结果（判定结果本身保留在审计历史里）。"""
        consumed = list(self.pending)
        self.pending.clear()
        return consumed

    # -- 重试 ---------------------------------------------------------

    def retry(
        self,
        wall_ms: Any = None,
        reopen: bool = True,
        duration_frames: Any = None,
    ) -> dict:
        """开启新回合：丢弃上一轮输入/判定，拒绝旧回合的晚到事件。"""
        wall = self._wall(wall_ms)
        old_attempt = self.attempt

        dropped_inputs = [
            event for event in self.queue if event.attempt == old_attempt
        ]
        for event in dropped_inputs:
            event.status = "discarded"
        self.queue = [event for event in self.queue if event.attempt != old_attempt]
        self.total_discarded += len(dropped_inputs)

        cleared_results = [
            result for result in self.judged if result.get("attempt") == old_attempt
        ]
        self.judged = [
            result for result in self.judged if result.get("attempt") != old_attempt
        ]
        cleared_pending = [
            result for result in self.pending if result.get("attempt") == old_attempt
        ]
        self.pending = [
            result for result in self.pending if result.get("attempt") != old_attempt
        ]

        self.attempt = old_attempt + 1
        self.attempt_wall_ms = wall
        reopened = None
        if reopen:
            reopened = self.open_window(
                wall, duration_frames=duration_frames
            )
        else:
            self.window = None

        return {
            "previous_attempt": old_attempt,
            "attempt": self.attempt,
            "dropped_inputs": len(dropped_inputs),
            "cleared_results": len(cleared_results),
            "cleared_pending": len(cleared_pending),
            "queue_size": len(self.queue),
            "queue": [event.to_dict() for event in self.queue],
            "window": reopened.to_dict() if reopened is not None else None,
            "wall_ms": _round(wall),
        }

    # -- 快照 ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "config": {
                "tick_rate": _round(self.tick_rate),
                "scale": _round(self.clock.scale),
                "window_frames": self.window_frames,
                "lead_frames": self.lead_frames,
                "perfect_frames": self.perfect_frames,
                "quantize": self.quantize,
                "wall_start": _round(self.wall_start),
            },
            "clock": self.clock.to_dict(),
            "attempt": self.attempt,
            "attempt_wall_ms": _round(self.attempt_wall_ms),
            "seq": self._seq,
            "last_wall_ms": _round(self.last_wall_ms),
            "window": self.window.to_dict() if self.window is not None else None,
            "queue": [event.to_dict() for event in self.queue],
            "judged": list(self.judged),
            "pending": list(self.pending),
            "rejected": [event.to_dict() for event in self.rejected],
            "counters": {
                "enqueued": self.total_enqueued,
                "judged": self.total_judged,
                "discarded": self.total_discarded,
            },
        }

    snapshot = to_dict

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> "QteEngine":
        config = data.get("config", {})
        engine = cls(**_config_from_mapping(config))
        if isinstance(data.get("clock"), Mapping):
            engine.clock = LogicalClock.from_dict(data["clock"])
            engine.tick_rate = engine.clock.tick_rate
            engine.scale = engine.clock.scale
        engine.attempt = max(1, int(data.get("attempt", 1)))
        engine.last_wall_ms = _num(data.get("last_wall_ms"), engine.wall_start)
        engine.attempt_wall_ms = _num(
            data.get("attempt_wall_ms"), engine.last_wall_ms
        )
        engine._seq = int(data.get("seq", 0))
        window = data.get("window")
        engine.window = (
            WindowTimeline.from_dict(window) if isinstance(window, Mapping) else None
        )
        engine.queue = [
            InputEvent.from_dict(event)
            for event in data.get("queue", [])
            if isinstance(event, Mapping)
        ]
        engine.judged = [
            dict(result) for result in data.get("judged", []) if isinstance(result, Mapping)
        ]
        engine.pending = [
            dict(result) for result in data.get("pending", []) if isinstance(result, Mapping)
        ]
        engine.rejected = [
            InputEvent.from_dict(event)
            for event in data.get("rejected", [])
            if isinstance(event, Mapping)
        ]
        counters = data.get("counters", {})
        engine.total_enqueued = int(counters.get("enqueued", 0))
        engine.total_judged = int(counters.get("judged", 0))
        engine.total_discarded = int(counters.get("discarded", 0))
        return engine

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "QteEngine":
        if not isinstance(data, Mapping):
            return cls()
        if "clock" in data or "queue" in data:
            return cls.from_snapshot(data)
        return cls(**_config_from_mapping(data))

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"QteEngine(tick_rate={self.tick_rate:g}, scale={self.clock.scale:g}, "
            f"attempt={self.attempt}, queue={len(self.queue)})"
        )


# ---------------------------------------------------------------------------
# 1. 窗口按逻辑时间轴定义
# ---------------------------------------------------------------------------


def window_timeline(*args: Any, **kwargs: Any) -> dict:
    """返回判定窗口的逻辑帧定义及换算出的墙钟时长。

    没有现成窗口时给出一个「若现在开窗」的假设窗口。数字位置参数按
    ``duration_frames`` 解释。返回字段包括 ``start_frame`` / ``end_frame`` /
    ``duration_frames`` / ``logical_duration_ms`` / ``wall_duration_ms``；
    ``wall_duration_ms`` 随 ``scale`` 变化，但逻辑帧定义不变。
    """
    engine, numbers, options, _ = _resolve(args, kwargs)
    duration = options.get(
        "duration_frames", options.get("window_frames", None)
    )
    if numbers:
        duration = numbers[0]
    lead = options.get("lead_frames")
    wall = options.get("wall_ms")

    if engine.window is not None and duration is None and lead is None:
        window = engine.window
        wall = engine.last_wall_ms if wall is None else _num(wall)
    else:
        wall = engine.last_wall_ms if wall is None else _num(wall)
        lead_frames = engine.lead_frames if lead is None else max(0, _int(lead))
        frames = engine.window_frames if duration is None else max(1, _int(duration, 1))
        start = engine.frame_at(wall) + lead_frames
        window = WindowTimeline(
            start_frame=start,
            duration_frames=frames,
            tick_rate=engine.tick_rate,
            scale=engine.clock.scale,
            attempt=engine.attempt,
            opened_wall_ms=wall,
            perfect_frames=engine.perfect_frames,
        )

    data = window.to_dict()
    data.update(
        {
            "timeline": "logical",
            "render_independent": True,
            "render_aligned": False,
            "duration_frames": window.duration_frames,
            "logical_ms": {
                "start": _round(window.start_frame * 1000.0 / window.tick_rate),
                "end": _round(window.end_frame * 1000.0 / window.tick_rate),
                "duration": _round(window.logical_duration_ms),
            },
        }
    )
    return data


# ---------------------------------------------------------------------------
# 2. 输入时间戳换算到逻辑帧
# ---------------------------------------------------------------------------


def input_timestamp(*args: Any, **kwargs: Any) -> dict:
    """把采样线程的墙钟时间戳换算成逻辑毫秒与逻辑帧。

    数字位置参数按 ``wall_ms`` 解释；缺省用引擎最后推进到的墙钟时刻。
    返回原始墙钟值、逻辑毫秒、未量化帧、量化帧以及是否处于冻结期。
    """
    engine, numbers, options, _ = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    # 显式时间戳按原值换算（回放里的晚到/乱序事件也必须是确定性的）。
    wall = engine.resolve_wall(wall)
    logical_ms = engine.clock.to_logical_ms(wall)
    frame_float = engine.clock.frame_float(wall)
    frame = engine.clock.to_frame(wall, engine.quantize)
    return {
        "wall_ms": _round(wall),
        "logical_ms": _round(logical_ms),
        "frame_float": _round(frame_float),
        "frame": frame,
        "logical_frame": frame,
        "quantize": engine.quantize,
        "subframe_offset": _round(frame_float - frame),
        "tick_rate": _round(engine.tick_rate),
        "time_scale": _round(engine.clock.scale),
        "paused": engine.clock.frozen(),
        "attempt": engine.attempt,
        "source": "wall_clock",
        "target": "logical_frame",
    }


# ---------------------------------------------------------------------------
# 3. 暂停冻结队列不清空
# ---------------------------------------------------------------------------


def pause_freeze(*args: Any, **kwargs: Any) -> dict:
    """暂停逻辑时钟并确认输入队列被原样保留。

    数字位置参数按 ``wall_ms`` 解释。暂停期间 ``input_timestamp`` 换算出的
    逻辑帧保持不变；已入队输入不清空、不丢弃，恢复后照常判定。
    """
    engine, numbers, options, _ = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    result = engine.pause(wall)
    result.update(
        {
            "frozen_logical_ms": _round(engine.clock.to_logical_ms(wall)),
            "time_scale": _round(engine.clock.scale),
            "window": engine.window.to_dict() if engine.window is not None else None,
        }
    )
    if options.get("resume_wall_ms") is not None:
        result["resume"] = engine.resume(options["resume_wall_ms"])
    return result


# ---------------------------------------------------------------------------
# 4. 重试清空上一轮判定
# ---------------------------------------------------------------------------


def retry_reset(*args: Any, **kwargs: Any) -> dict:
    """进入下一回合：清空上一轮未判定输入与判定结果。

    数字位置参数按 ``wall_ms`` 解释。上一轮的输入被标记为 discarded；
    带旧 ``attempt`` 的晚到事件之后经 :func:`enqueue` 也会被拒绝，不会
    被算进新窗口。
    """
    engine, numbers, options, _ = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    reopen = bool(options.get("reopen", True))
    duration = options.get("duration_frames")
    return engine.retry(wall, reopen=reopen, duration_frames=duration)


# ---------------------------------------------------------------------------
# 5. 慢动作统一时间缩放
# ---------------------------------------------------------------------------


def time_scale(*args: Any, **kwargs: Any) -> dict:
    """设置/查询统一时间缩放，并给出窗口墙钟时长的变化。

    数字位置参数按 ``scale`` 解释（可再跟一个 ``wall_ms``）。逻辑窗口帧数
    不随缩放变化；墙钟时长 = 逻辑时长 / scale。输入时间戳换算同样经过
    该速率，因此慢动作下窗口与输入保持一致。
    """
    engine, numbers, options, _ = _resolve(args, kwargs)
    scale = options.get("scale")
    if numbers:
        scale = numbers[0]
    wall = options.get("wall_ms", numbers[1] if len(numbers) > 1 else engine.last_wall_ms)
    if scale is None:
        scale = engine.clock.scale

    raw = _num(scale, engine.clock.scale)
    repaired = raw <= 0
    applied = engine.set_time_scale(raw, wall)
    result = {
        "scale": _round(engine.clock.scale),
        "time_scale": _round(engine.clock.scale),
        **applied,
        "repaired": repaired,
        "input_conversion_uses_scale": True,
        "window_frames_unchanged": True,
    }
    if engine.window is not None:
        result["window"] = engine.window.to_dict()
    return result


# ---------------------------------------------------------------------------
# 6. 判定与帧率解耦 / 确定性
# ---------------------------------------------------------------------------


def _scenario_digest(result: Mapping[str, Any]) -> str:
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_scenario(config: Mapping[str, Any]) -> dict:
    """按给定渲染帧率/抖动跑一遍固定录像，返回与帧率无关的判定摘要。

    渲染循环只决定「什么时候去看输入」，输入本身始终使用原始墙钟时间戳
    换算逻辑帧，因此渲染帧率不参与判定。
    """
    tick_rate = _num(config.get("tick_rate", config.get("logical_fps")), LOGICAL_FPS)
    scale = _num(config.get("scale"), DEFAULT_SCALE)
    duration = _int(config.get("duration_frames", config.get("window_frames")), DEFAULT_WINDOW_FRAMES)
    lead = _int(config.get("lead_frames"), DEFAULT_LEAD_FRAMES)
    perfect = _int(config.get("perfect_frames"), DEFAULT_PERFECT_FRAMES)
    render_fps = max(EPS, _num(config.get("render_fps"), 60.0))
    input_wall = _num(config.get("input_wall_ms"), 100.0)
    pause_at = config.get("pause_at")
    resume_at = config.get("resume_at")
    jitter_ms = _num(config.get("jitter_ms"), 0.0)

    engine = QteEngine(
        tick_rate=tick_rate,
        scale=scale,
        window_frames=duration,
        lead_frames=lead,
        perfect_frames=perfect,
        wall_start=0.0,
    )
    engine.open_window(wall_ms=0.0, duration_frames=duration, lead_frames=lead)
    if pause_at is not None:
        engine.pause(_num(pause_at))
    if resume_at is not None:
        engine.resume(_num(resume_at))

    step = 1000.0 / render_fps
    # 抖动只影响渲染循环「看到」输入的时机，不影响输入自带的墙钟时间戳。
    render_offset = max(0.0, jitter_ms)
    delivered = False
    wall = render_offset
    horizon = max(input_wall, _num(resume_at, 0.0), _num(pause_at, 0.0)) + step * 2 + 1.0
    guard = 0
    while wall <= horizon and guard < 1_000_000:
        if not delivered and wall >= input_wall:
            engine.enqueue(input_wall)
            delivered = True
        wall += step
        guard += 1
    if not delivered:
        engine.enqueue(input_wall)

    judged = engine.judge(input_wall)
    state = engine.window_state(input_wall)
    summary = {
        "outcome": judged["outcome"],
        "grade": (judged["best"] or {}).get("grade"),
        "frame": (judged["best"] or {}).get("frame"),
        "offset_frames": (judged["best"] or {}).get("offset_frames"),
        "window_start": (judged["window"] or {}).get("start_frame"),
        "window_end": (judged["window"] or {}).get("end_frame"),
        "paused": state["paused"],
        "state": state["state"],
        "attempt": engine.attempt,
        "scale": _round(engine.clock.scale),
    }
    summary["digest"] = _scenario_digest(summary)
    summary["render_fps"] = _round(render_fps)
    return summary


def frame_decouple(*args: Any, **kwargs: Any) -> dict:
    """同一份输入在多种渲染帧率下判定，验证结果与帧率无关。

    数字位置参数视为渲染帧率列表；也接受 ``render_fps=[...]`` /
    ``frame_rates=[...]``。返回每种帧率下的判定摘要与 ``consistent``。
    """
    _, numbers, options, _ = _resolve(args, kwargs)
    rates = options.get("render_fps", options.get("frame_rates"))
    if rates is None and numbers:
        rates = numbers
    if rates is None:
        rates = (24.0, 30.0, 60.0, 90.0, 120.0, 144.0)
    if _is_number(rates):
        rates = (rates,)
    rates = [max(EPS, _num(rate, 60.0)) for rate in rates] or [60.0]

    base = {
        key: options[key]
        for key in (
            "tick_rate",
            "logical_fps",
            "scale",
            "duration_frames",
            "window_frames",
            "lead_frames",
            "perfect_frames",
            "input_wall_ms",
            "pause_at",
            "resume_at",
            "jitter_ms",
        )
        if key in options
    }
    results = {}
    digests = []
    for rate in rates:
        summary = _run_scenario({**base, "render_fps": rate})
        results[f"{rate:g}"] = summary
        digests.append(summary["digest"])
    reference = digests[0]
    return {
        "results": results,
        "render_fps": [_round(rate) for rate in rates],
        "digest": reference,
        "consistent": all(digest == reference for digest in digests),
        "render_independent": True,
    }


def determinism(*args: Any, **kwargs: Any) -> dict:
    """重复回放并混入不同帧率/处理抖动，验证判定摘要完全一致。

    数字位置参数视为渲染帧率列表。返回规范摘要 ``digest``、运行次数、
    每轮摘要以及 ``identical``。
    """
    _, numbers, options, _ = _resolve(args, kwargs)
    runs = max(1, _int(options.get("runs"), 5))
    rates = options.get("render_fps", options.get("frame_rates"))
    if rates is None and numbers:
        rates = numbers
    if rates is None or not rates:
        rates = (24.0, 30.0, 60.0, 90.0, 120.0, 144.0)
    if _is_number(rates):
        rates = (rates,)
    rates = [max(EPS, _num(rate, 60.0)) for rate in rates]

    base = {
        key: options[key]
        for key in (
            "tick_rate",
            "logical_fps",
            "scale",
            "duration_frames",
            "window_frames",
            "lead_frames",
            "perfect_frames",
            "input_wall_ms",
            "pause_at",
            "resume_at",
        )
        if key in options
    }
    jitter_values = options.get("jitter_ms")
    if jitter_values is None:
        jitter_values = (0.0, 3.0, 7.0, 11.0, 17.0)

    runs_detail = []
    digests = []
    for index in range(runs):
        rate = rates[index % len(rates)]
        jitter = _num(jitter_values[index % len(jitter_values)], 0.0)
        summary = _run_scenario(
            {**base, "render_fps": rate, "jitter_ms": jitter}
        )
        digests.append(summary["digest"])
        runs_detail.append(
            {
                "run": index,
                "render_fps": _round(rate),
                "jitter_ms": _round(jitter),
                "outcome": summary["outcome"],
                "frame": summary["frame"],
                "digest": summary["digest"],
            }
        )
    reference = digests[0]
    return {
        "digest": reference,
        "runs": runs,
        "results": runs_detail,
        "identical": all(digest == reference for digest in digests),
        "deterministic": all(digest == reference for digest in digests),
        "canonical": "logical_frame",
    }


# ---------------------------------------------------------------------------
# 校验 / 兜底
# ---------------------------------------------------------------------------


def _validate_config(engine: QteEngine) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not math.isfinite(engine.tick_rate) or engine.tick_rate <= 0:
        errors.append("tick_rate 必须为正数")
    if not math.isfinite(engine.clock.scale) or engine.clock.scale <= 0:
        errors.append("time_scale 必须为正数")
    if engine.window_frames < 1:
        errors.append("window_frames 必须 >= 1")
    if engine.lead_frames < 0:
        errors.append("lead_frames 不能为负")
    if engine.perfect_frames < 0:
        errors.append("perfect_frames 不能为负")
    if engine.pending and engine.window is None:
        warnings.append("存在未消费判定结果但窗口已关闭")
    for segment in engine.clock._segments:
        if segment.rate < 0:
            errors.append("逻辑时钟速率不能为负")
            break
    for event in engine.queue:
        if event.attempt != engine.attempt:
            warnings.append(f"队列中存在旧回合输入 seq={event.seq:g}")
            break
    if engine.window is not None and engine.window.scale != engine.clock.scale:
        warnings.append("窗口记录的缩放与逻辑时钟不一致")
    return errors, warnings


def validation(*args: Any, **kwargs: Any) -> dict:
    """校验引擎配置、逻辑时钟与队列一致性。返回 ``ok``/``errors``/``warnings``。

    同时对调用方给的原始配置做检查（引擎构造会静默归一化非法值，只看引擎
    内部状态会漏掉 ``tick_rate=-5`` 这类输入）。
    """
    engine, _, _, _ = _resolve(args, kwargs)
    supplied = _raw_config(args, kwargs)
    errors, warnings = _validate_config(engine)

    raw_tick = supplied.get("tick_rate", supplied.get("logical_fps", supplied.get("fps")))
    if raw_tick is not None and _num(raw_tick, -1.0) <= 0:
        errors.append("tick_rate 必须为正数")
    raw_scale = supplied.get("scale", supplied.get("time_scale"))
    if raw_scale is not None and _num(raw_scale, -1.0) <= 0:
        errors.append("time_scale 必须为正数")
    raw_frames = supplied.get("window_frames", supplied.get("duration_frames"))
    if raw_frames is not None and _int(raw_frames, 0) < 1:
        errors.append("window_frames 必须 >= 1")
    raw_lead = supplied.get("lead_frames")
    if raw_lead is not None and _int(raw_lead, 0) < 0:
        errors.append("lead_frames 不能为负")
    raw_perfect = supplied.get("perfect_frames")
    if raw_perfect is not None and _int(raw_perfect, 0) < 0:
        errors.append("perfect_frames 不能为负")
    raw_quantize = supplied.get("quantize")
    if raw_quantize is not None and raw_quantize not in _QUANTIZERS:
        errors.append("quantize 必须是 nearest/floor/ceil 之一")
    # 去重，避免原始值与引擎归一化后的状态重复报同一个错
    errors = list(dict.fromkeys(errors))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "tick_rate": engine.tick_rate > 0,
            "time_scale": engine.clock.scale > 0,
            "window_frames": engine.window_frames >= 1,
            "clock_monotonic": all(
                later.wall_ms >= earlier.wall_ms - EPS
                for earlier, later in zip(engine.clock._segments, engine.clock._segments[1:])
            ),
            "queue_attempts": all(
                event.attempt == engine.attempt for event in engine.queue
            ),
            "window_logical": engine.window is not None,
        },
        "config": {
            "tick_rate": _round(engine.tick_rate),
            "time_scale": _round(engine.clock.scale),
            "window_frames": engine.window_frames,
            "lead_frames": engine.lead_frames,
            "perfect_frames": engine.perfect_frames,
            "quantize": engine.quantize,
        },
    }


def fallback(*args: Any, **kwargs: Any) -> dict:
    """对非法配置做兜底修复，而不是直接失败。

    ``tick_rate <= 0`` -> 60；``scale <= 0`` -> 1；``window_frames < 1`` -> 12；
    负数 ``lead_frames``/``perfect_frames`` -> 0；未知 quantize -> nearest。
    校验的是调用方给的**原始值**（引擎构造时会静默归一化），因此显式传入的
    非法值会出现在 ``changes`` 里。返回修复后的配置、修复数量与逐项说明；
    若传入引擎，额外返回引擎快照。
    """
    engine, _, options, _ = _resolve(args, kwargs)
    supplied = _raw_config(args, kwargs)

    changes: list[dict] = []
    raw = {
        "tick_rate": engine.tick_rate,
        "scale": engine.clock.scale,
        "window_frames": engine.window_frames,
        "lead_frames": engine.lead_frames,
        "perfect_frames": engine.perfect_frames,
        "quantize": engine.quantize,
    }
    if "tick_rate" in supplied:
        raw["tick_rate"] = supplied["tick_rate"]
    elif "logical_fps" in supplied:
        raw["tick_rate"] = supplied["logical_fps"]
    elif "fps" in supplied:
        raw["tick_rate"] = supplied["fps"]
    if "scale" in supplied:
        raw["scale"] = supplied["scale"]
    elif "time_scale" in supplied:
        raw["scale"] = supplied["time_scale"]
    if "window_frames" in supplied:
        raw["window_frames"] = supplied["window_frames"]
    elif "duration_frames" in supplied:
        raw["window_frames"] = supplied["duration_frames"]
    for key in ("lead_frames", "perfect_frames", "quantize"):
        if key in supplied:
            raw[key] = supplied[key]

    fixed: dict[str, Any] = {}
    tick = _num(raw["tick_rate"], LOGICAL_FPS)
    if tick <= 0:
        changes.append({"field": "tick_rate", "from": raw["tick_rate"], "to": LOGICAL_FPS})
        tick = LOGICAL_FPS
    fixed["tick_rate"] = tick

    scale = _num(raw["scale"], DEFAULT_SCALE)
    if scale <= 0:
        changes.append({"field": "scale", "from": raw["scale"], "to": DEFAULT_SCALE})
        scale = DEFAULT_SCALE
    fixed["scale"] = scale

    frames = _int(raw["window_frames"], DEFAULT_WINDOW_FRAMES)
    if frames < 1:
        changes.append(
            {"field": "window_frames", "from": raw["window_frames"], "to": DEFAULT_WINDOW_FRAMES}
        )
        frames = DEFAULT_WINDOW_FRAMES
    fixed["window_frames"] = frames

    lead = _int(raw["lead_frames"], DEFAULT_LEAD_FRAMES)
    if lead < 0:
        changes.append({"field": "lead_frames", "from": raw["lead_frames"], "to": 0})
        lead = 0
    fixed["lead_frames"] = lead

    perfect = _int(raw["perfect_frames"], DEFAULT_PERFECT_FRAMES)
    if perfect < 0:
        changes.append({"field": "perfect_frames", "from": raw["perfect_frames"], "to": 0})
        perfect = 0
    if perfect > frames:
        changes.append({"field": "perfect_frames", "from": perfect, "to": frames})
        perfect = frames
    fixed["perfect_frames"] = perfect

    quantize = raw["quantize"]
    if quantize not in _QUANTIZERS:
        changes.append({"field": "quantize", "from": quantize, "to": "nearest"})
        quantize = "nearest"
    fixed["quantize"] = quantize

    result = {
        "config": {
            "tick_rate": _round(fixed["tick_rate"]),
            "scale": _round(fixed["scale"]),
            "window_frames": fixed["window_frames"],
            "lead_frames": fixed["lead_frames"],
            "perfect_frames": fixed["perfect_frames"],
            "quantize": fixed["quantize"],
        },
        "repaired": len(changes),
        "changes": changes,
        "ok": True,
    }
    if engine.window is not None or engine.queue or engine.judged:
        result["engine"] = engine.to_dict()
    return result


# ---------------------------------------------------------------------------
# 状态 / 审计 / 指标
# ---------------------------------------------------------------------------


def window_state(*args: Any, **kwargs: Any) -> dict:
    """返回窗口在逻辑时间轴上的当前状态（idle/pending/open/expired）。"""
    engine, numbers, options, _ = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    return engine.window_state(wall)


def audit(*args: Any, **kwargs: Any) -> dict:
    """汇总引擎状态、队列、判定、暂停/缩放与校验结果，供录制回放审计。"""
    engine, numbers, options, _ = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    state = engine.window_state(wall)
    check = validation(engine=engine)
    grade_counts = Counter(
        result.get("grade") for result in engine.judged if result.get("grade")
    )
    return {
        "attempt": engine.attempt,
        "window_state": state["state"],
        "paused": state["paused"],
        "time_scale": _round(engine.clock.scale),
        "tick_rate": _round(engine.tick_rate),
        "queue_size": len(engine.queue),
        "judged_total": len(engine.judged),
        "pending_results": len(engine.pending),
        "rejected_total": len(engine.rejected),
        "discarded_total": engine.total_discarded,
        "grade_counts": dict(grade_counts),
        "window": state["window"],
        "segments": engine.clock.to_dict()["segments"],
        "validation": check,
        "objectives": {
            "logical_window": True,
            "wall_to_logical_input": True,
            "pause_keeps_queue": True,
            "retry_clears_previous": True,
            "unified_time_scale": True,
            "frame_rate_decoupled": True,
        },
    }


def metrics(*args: Any, **kwargs: Any) -> dict:
    """统计输入/判定/窗口指标；空引擎返回结构完整的零值。"""
    engine, numbers, options, _ = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    state = engine.window_state(wall)
    grade_counts = Counter(
        result.get("grade") for result in engine.judged if result.get("grade")
    )
    offsets = [
        result["offset_frames"]
        for result in engine.judged
        if isinstance(result.get("offset_frames"), int)
    ]
    window = engine.window
    return {
        "total_enqueued": engine.total_enqueued,
        "total_judged": engine.total_judged,
        "queue_size": len(engine.queue),
        "pending_results": len(engine.pending),
        "rejected": len(engine.rejected),
        "discarded": engine.total_discarded,
        "attempt": engine.attempt,
        "grade_counts": dict(grade_counts),
        "hit_rate": _round(
            (
                grade_counts.get("perfect", 0) + grade_counts.get("good", 0)
            )
            / max(1, engine.total_judged)
        ),
        "offset_frames": {
            "count": len(offsets),
            "min": min(offsets) if offsets else None,
            "max": max(offsets) if offsets else None,
            "avg": _round(sum(offsets) / len(offsets)) if offsets else None,
        },
        "window": window.to_dict() if window is not None else None,
        "window_state": state["state"],
        "paused": state["paused"],
        "time_scale": _round(engine.clock.scale),
        "tick_rate": _round(engine.tick_rate),
    }


# ---------------------------------------------------------------------------
# 判定 / 入队 / 消费
# ---------------------------------------------------------------------------


def judge(*args: Any, **kwargs: Any) -> dict:
    """判定当前回合已入队输入，返回逐条结果与窗口级最优判定。"""
    engine, numbers, options, created = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    if created and engine.window is None:
        engine.open_window(wall_ms=wall)
    return engine.judge(wall)


def enqueue(*args: Any, **kwargs: Any) -> dict:
    """把带墙钟时间戳的输入换算到逻辑帧并入队。

    数字位置参数按 ``wall_ms`` 解释（可再跟 ``attempt``）。返回事件自身与
    入队后的队列大小；旧回合事件返回 ``stale``，不进入当前窗口。
    """
    engine, numbers, options, created = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    attempt = options.get("attempt")
    if attempt is None and len(numbers) > 1:
        attempt = int(numbers[1])
    if created and engine.window is None:
        engine.open_window(wall_ms=wall)
    event = engine.enqueue(wall, button=options.get("button"), attempt=attempt)
    return {
        "event": event.to_dict(),
        "accepted": event.status == "queued",
        "status": event.status,
        "queue_size": len(engine.queue),
        "attempt": engine.attempt,
        "window": engine.window.to_dict() if engine.window is not None else None,
    }


def consume(*args: Any, **kwargs: Any) -> list[dict]:
    """取走尚未消费的判定结果（列表，按判定顺序）。

    没有判定结果时返回空列表。判定历史仍保留在引擎中供审计。
    """
    engine, numbers, options, created = _resolve(args, kwargs)
    wall = options.get("wall_ms", numbers[0] if numbers else engine.last_wall_ms)
    if engine.pending:
        return engine.consume()
    if engine.queue:
        if created and engine.window is None:
            engine.open_window(wall_ms=wall)
        engine.judge(wall)
    return engine.consume()
