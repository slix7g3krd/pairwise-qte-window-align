import pytest
import gameplay.qte_window_align as mod

CASES = [
    ("window_timeline", (1,)),
    ("input_timestamp", (1,)),
    ("pause_freeze", (1,)),
    ("retry_reset", (1,)),
    ("time_scale", (1,)),
    ("frame_decouple", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("window_state", (1,)),
    ("metrics", (1,)),
    ("judge", (1,)),
    ("enqueue", (1,)),
    ("consume", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
