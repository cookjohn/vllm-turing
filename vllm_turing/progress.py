"""Live request progress for the monitoring proxy (tools: scripts/python/vllm_monitor). vLLM's metrics count a
request's prompt tokens only when its first output token appears, so a long prefill shows no progress until it is
over. After each scheduling step the engine core writes its waiting and running requests (request id, prompt tokens,
prompt tokens computed so far, prefix-cache hit tokens, output tokens, status) and a cumulative count of computed
prompt tokens to /dev/shm/vllm_progress.json (atomic replace, at most every 0.2 s unless the request set changes).
The monitor tags requests with an X-Request-Id header, which vLLM uses as the request id. VLLM_TURING_PROGRESS=0
disables, VLLM_TURING_PROGRESS_FILE sets the path."""
import json
import logging
import os
import time

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_PROGRESS", "1") != "0"
PATH = os.environ.get("VLLM_TURING_PROGRESS_FILE", "/dev/shm/vllm_progress.json")
_state = {"t": 0.0, "keys": None, "prefill_total": 0}


def _snapshot(sched, out):
    now = time.time()
    for rid, n in out.num_scheduled_tokens.items():
        req = sched.requests.get(rid)
        if req is not None:
            # num_computed_tokens is already advanced by this step's tokens
            before = req.num_computed_tokens - n
            if before < req.num_prompt_tokens:
                _state["prefill_total"] += min(n, req.num_prompt_tokens - before)
    reqs = list(sched.running) + list(sched.waiting)
    keys = tuple((r.request_id, r.status) for r in reqs)
    prefilling = any(r.num_computed_tokens < r.num_prompt_tokens for r in reqs)
    if keys == _state["keys"] and now - _state["t"] < (0.2 if prefilling else 1.0):
        return
    _state["t"], _state["keys"] = now, keys
    data = {"t": now, "prefill_total": _state["prefill_total"],
            "reqs": [{"id": r.request_id, "prompt": r.num_prompt_tokens,
                      "computed": min(r.num_computed_tokens, r.num_prompt_tokens),
                      "cached": max(getattr(r, "num_cached_tokens", -1), -1), "out": r.num_output_tokens,
                      "status": getattr(r.status, "name", str(r.status)), "arrival": r.arrival_time} for r in reqs]}
    tmp = PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, PATH)


def install():
    if not ENABLED:
        return
    try:
        from vllm.v1.core.sched import scheduler as sm
    except Exception:   # noqa: BLE001
        return
    S = sm.Scheduler
    if getattr(S, "_vllm_turing_progress", False):
        return
    orig = S.schedule

    def schedule(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            _snapshot(self, out)
        except Exception as e:   # noqa: BLE001
            if not _state.get("warned"):
                _state["warned"] = True
                logger.warning("vllm_turing progress: %s", e)
        return out

    S.schedule, S._vllm_turing_progress = schedule, True
