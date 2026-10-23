#!/usr/bin/env python3
"""Monitoring reverse proxy for the production vLLM server. Clients keep using :8080; the proxy forwards every request
to vLLM on an internal port unchanged (streams pass through chunk by chunk, a client disconnect closes the upstream
request so vLLM aborts it) and records each generation request (OpenAI chat / completions / responses, Anthropic
messages): the input (messages, tools, parameters), the live output (reasoning, content, tool calls), timings (waiting
= queue + prompt processing until the first output, output rate, time since the last output), finish reason, usage
(prompt / cached / completion tokens), HTTP errors and client disconnects. Live web dashboard at /monitor (login with
the API key: /monitor?key=...), daily gzip JSONL logs of finished requests (--log-dir, kept --retain-days days and at
most --max-log-gb). Server panel: vLLM /metrics (running / waiting requests, KV cache, prefix cache hits, token
throughput, draft acceptance), nvidia-smi, recent vLLM warnings / errors from the journal of the vllm-server unit.
Two request changes: a long agent conversation (Claude Code, opencode) is forwarded with the tool traffic of its older
part cleared (ctxedit.py, VMON_CTX_*: a ~1M-token conversation fits in a ~200K prompt; the original body stays in the
records), and a request whose prompt + max_tokens exceeds the context is resent with the room left (VMON_MIN_OUTPUT).
usage: vllm_monitor.py [--listen 0.0.0.0:8080] [--upstream http://127.0.0.1:8090] [--model-dir DIR] [--log-dir DIR]"""
import argparse
import asyncio
import collections
import gzip
import hashlib
import itertools
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from aiohttp import web

import ctxedit

log = logging.getLogger("vmon")
GEN_PATHS = {"/v1/chat/completions": "chat", "/v1/completions": "completions", "/v1/messages": "anthropic",
             "/v1/responses": "responses"}
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
       "upgrade", "host", "content-length", "content-encoding", "accept-encoding"}
HERE = os.path.dirname(os.path.abspath(__file__))
# written by the vllm_turing plugin's progress hook in the vLLM engine core (per-request prefill progress)
PROGRESS_FILE = os.environ.get("VLLM_TURING_PROGRESS_FILE", "/dev/shm/vllm_progress.json")
KEEPALIVE = float(os.environ.get("VMON_KEEPALIVE_S", "10"))    # SSE comment after this many silent seconds
# vLLM rejects a request whose prompt + max_tokens exceeds the context (Claude Code asks for 32000 output tokens at any
# conversation length); its message differs from the Anthropic API's, so the client's own max_tokens reduction never
# runs. The proxy resends such a request with max_tokens = the room left (prompt length from vLLM's own count), if that
# is at least MIN_OUTPUT tokens; otherwise the client gets the Anthropic API's "prompt is too long" error (Claude Code
# then compacts)
CTX_RE = re.compile(rb"maximum context length is (\d+) tokens\. However, you requested (\d+) output tokens and your "
                    rb"prompt contains (at least )?(\d+) input tokens")
MIN_OUTPUT = int(os.environ.get("VMON_MIN_OUTPUT", "4096"))
# a generation request waits this long for vLLM to come back (restart) before it fails with 502
UPSTREAM_WAIT = float(os.environ.get("VMON_UPSTREAM_WAIT_S", "900"))
FIT_MARGIN = 64     # count_tokens renders without the thinking / effort switches: a few tokens short of the request


def _text_of(content):
    """Text of an OpenAI / Anthropic message content (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in content if isinstance(content, list) else [content]:
        if isinstance(p, dict):
            t = p.get("type", "")
            if "text" in p and isinstance(p["text"], str):
                out.append(p["text"])
            elif t in ("tool_result",):
                out.append(_text_of(p.get("content")))
            elif t in ("tool_use",):
                out.append(json.dumps(p.get("input", {}), ensure_ascii=False))
            elif "image" in t:
                out.append("[image]")
        elif isinstance(p, str):
            out.append(p)
    return "\n".join(out)


def prompt_total(u):
    """Prompt tokens of a usage block: OpenAI prompt_tokens, or Anthropic input_tokens, which excludes the tokens read
    from or written to the prompt cache."""
    if not u:
        return None
    if u.get("prompt_tokens") is not None:
        return u["prompt_tokens"]
    if u.get("input_tokens") is None:
        return None
    return u["input_tokens"] + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)


class Rec:
    """One generation request."""

    def __init__(self, rid, request, body, kind):
        self.id = rid
        self.kind = kind
        self.path = request.path
        self.t0 = time.time()
        self.client = request.remote or "?"
        self.ua = request.headers.get("User-Agent", "")[:120]
        self.state = "waiting"          # waiting | streaming | done | error | aborted
        self.status = None
        self.t_first = self.t_last = self.t_end = None
        self.content, self.reasoning = [], []
        self.tools = {}                 # index -> {"id", "name", "args"}
        self.finish = None
        self.usage = {}
        self.resp_id = None
        self.err = None
        self.chunks = 0
        self.buf = b""
        self.est_prompt = None          # prompt estimate: tokenizer count of the texts x calibration ratio
        self.raw_est = None             # tokenizer count of the message / tool texts (no chat template)
        self.out_tok = 0                # tokenizer estimate of the output so far (content + reasoning + tool calls)
        self.out_tok_r = 0              # of which reasoning
        self.sent_tools = 0             # published tool-call argument characters
        self.vreq = None                # X-Request-Id sent to vLLM (part of its request id)
        self.v = None                   # engine view: prompt / computed / cached tokens, status
        self.v_rate = None              # this request's prompt processing rate (tokens/s)
        self.v_seen = None              # (time, computed) of the last engine snapshot
        self.sent = [0, 0]              # published content / reasoning lengths
        self.dirty = False
        try:
            self.body = json.loads(body) if body else {}
        except Exception:   # noqa: BLE001
            self.body = {"_raw": body[:200000].decode("utf-8", "replace")}
        self.fwd = self.body            # the body forwarded to vLLM (context edits, max_tokens fitting)
        b = self.body if isinstance(self.body, dict) else {}
        self.model = b.get("model", "")
        self.stream = bool(b.get("stream"))
        self.params = {k: b[k] for k in ("max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
                                          "stop", "tool_choice", "ignore_eos", "n", "presence_penalty",
                                          "repetition_penalty", "thinking", "chat_template_kwargs", "reasoning",
                                          "response_format") if k in b}
        msgs = b.get("messages") or b.get("input") or []
        if isinstance(msgs, str):
            msgs = [{"role": "user", "content": msgs}]
        self.n_msgs = len(msgs) if isinstance(msgs, list) else 0
        tools = b.get("tools") or []
        self.n_tools = len(tools) if isinstance(tools, list) else 0
        self.prompt_text = self._prompt_text(b, msgs if isinstance(msgs, list) else [], tools)
        last = ""
        if isinstance(msgs, list) and msgs:
            m = msgs[-1]
            last = f"[{m.get('role', '?')}] " + _text_of(m.get("content")) if isinstance(m, dict) else str(m)
        elif "prompt" in b:
            last = str(b["prompt"])[-300:]
        self.last_input = last[-300:]

    @staticmethod
    def _prompt_text(b, msgs, tools):
        parts = []
        if b.get("system"):
            parts.append(_text_of(b["system"]))
        for m in msgs:
            if isinstance(m, dict):
                parts.append(_text_of(m.get("content")))
                for tc in m.get("tool_calls") or []:
                    parts.append(json.dumps(tc, ensure_ascii=False))
        if "prompt" in b:
            parts.append(b["prompt"] if isinstance(b["prompt"], str) else json.dumps(b["prompt"]))
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        return "\n".join(parts)

    # ---- output parsing -------------------------------------------------------------------------------------------
    def _out(self, content="", reasoning=""):
        if content or reasoning:
            now = time.time()
            if self.t_first is None:
                self.t_first = now
                self.state = "streaming"
            self.t_last = now
            if content:
                self.content.append(content)
            if reasoning:
                self.reasoning.append(reasoning)
            self.dirty = True

    def _tool(self, index, tid=None, name=None, args=""):
        t = self.tools.setdefault(index, {"id": "", "name": "", "args": ""})
        if tid:
            t["id"] = tid
        if name:
            t["name"] += name
        if args:
            t["args"] += args
        now = time.time()
        if self.t_first is None:
            self.t_first = now
            self.state = "streaming"
        self.t_last = now
        self.dirty = True

    def _event(self, d):
        if not isinstance(d, dict):
            return
        if d.get("id") and not self.resp_id:
            self.resp_id = d["id"]
        if d.get("usage"):
            self.usage = d["usage"]
        typ = d.get("type", "")
        if self.kind == "anthropic" or typ.startswith(("message", "content_block")):
            if typ == "message_start":
                m = d.get("message") or {}
                self.resp_id = m.get("id", self.resp_id)
                self.usage = dict(m.get("usage") or {})
            elif typ == "content_block_start":
                cb = d.get("content_block") or {}
                if cb.get("type") == "tool_use":
                    self._tool(d.get("index", 0), cb.get("id"), cb.get("name"))
                elif cb.get("type") == "text" and cb.get("text"):
                    self._out(cb["text"])
            elif typ == "content_block_delta":
                dl = d.get("delta") or {}
                if dl.get("type") == "text_delta":
                    self._out(dl.get("text", ""))
                elif dl.get("type") == "thinking_delta":
                    self._out(reasoning=dl.get("thinking", ""))
                elif dl.get("type") == "input_json_delta":
                    self._tool(d.get("index", 0), args=dl.get("partial_json", ""))
            elif typ == "message_delta":
                dl = d.get("delta") or {}
                if dl.get("stop_reason"):
                    self.finish = dl["stop_reason"]
                if d.get("usage"):
                    self.usage = {**self.usage, **d["usage"]}
            elif typ == "error":
                self.err = json.dumps(d.get("error"), ensure_ascii=False)[:2000]
            return
        if typ.startswith("response."):                      # OpenAI Responses API
            if typ.endswith("output_text.delta"):
                self._out(d.get("delta", ""))
            elif "reasoning" in typ and typ.endswith(".delta"):
                self._out(reasoning=d.get("delta", ""))
            elif typ.endswith("function_call_arguments.delta"):
                self._tool(d.get("output_index", 0), args=d.get("delta", ""))
            elif typ == "response.output_item.added" and (d.get("item") or {}).get("type") == "function_call":
                it = d["item"]
                self._tool(d.get("output_index", 0), it.get("call_id"), it.get("name"))
            elif typ in ("response.completed", "response.failed", "response.incomplete"):
                r = d.get("response") or {}
                self.usage = r.get("usage") or self.usage
                self.finish = r.get("status")
            return
        if "error" in d and not d.get("choices"):
            self.err = json.dumps(d["error"], ensure_ascii=False)[:2000]
        for ch in d.get("choices") or []:
            if self.kind == "completions":
                self._out(ch.get("text") or "")
            dl = ch.get("delta") or {}
            self._out(dl.get("content") or "", dl.get("reasoning_content") or dl.get("reasoning") or "")
            for tc in dl.get("tool_calls") or []:
                fn = tc.get("function") or {}
                self._tool(tc.get("index", 0), tc.get("id"), fn.get("name"), fn.get("arguments") or "")
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]

    def feed(self, chunk):
        self.chunks += 1
        self.buf += chunk
        while True:
            i = self.buf.find(b"\n\n")
            if i < 0:
                break
            ev, self.buf = self.buf[:i], self.buf[i + 2:]
            for line in ev.split(b"\n"):
                line = line.strip()
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data and data != b"[DONE]":
                        try:
                            self._event(json.loads(data))
                        except Exception:   # noqa: BLE001
                            pass

    def complete_json(self, status, data):
        self.status = status
        try:
            d = json.loads(data)
        except Exception:   # noqa: BLE001
            d = None
        if status >= 400 or d is None:
            self.fail((data[:4000].decode("utf-8", "replace") if isinstance(data, bytes) else str(data)), status)
            return
        self.resp_id = d.get("id")
        self.usage = d.get("usage") or {}
        if self.kind == "anthropic":
            for b in d.get("content") or []:
                if b.get("type") == "text":
                    self._out(b.get("text", ""))
                elif b.get("type") == "thinking":
                    self._out(reasoning=b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    self._tool(len(self.tools), b.get("id"), b.get("name"),
                               json.dumps(b.get("input", {}), ensure_ascii=False))
            self.finish = d.get("stop_reason")
        elif self.kind == "responses":
            for it in d.get("output") or []:
                if it.get("type") == "message":
                    self._out(_text_of(it.get("content")))
                elif it.get("type") == "reasoning":
                    self._out(reasoning=_text_of(it.get("content") or it.get("summary")))
                elif it.get("type") == "function_call":
                    self._tool(len(self.tools), it.get("call_id"), it.get("name"), it.get("arguments", ""))
            self.finish = d.get("status")
        else:
            for ch in d.get("choices") or []:
                if self.kind == "completions":
                    self._out(ch.get("text") or "")
                m = ch.get("message") or {}
                self._out(m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or "")
                for i, tc in enumerate(m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    self._tool(i, tc.get("id"), fn.get("name"), fn.get("arguments") or "")
                self.finish = ch.get("finish_reason")
        self.end("done")

    def fail(self, msg, status=None):
        self.err = (self.err + " | " if self.err else "") + str(msg)[:4000]
        if status:
            self.status = status
        self.end("error")

    def end(self, state):
        if self.t_end is None:
            self.t_end = time.time()
        if self.state not in ("error", "aborted") or state in ("error", "aborted"):
            self.state = state
        self.dirty = True

    # ---- views ----------------------------------------------------------------------------------------------------
    def summary(self):
        u = self.usage or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("cache_read_input_tokens")
        return {"id": self.id, "kind": self.kind, "client": self.client, "ua": self.ua, "model": self.model,
                "stream": self.stream, "state": self.state, "status": self.status, "t0": self.t0,
                "t_first": self.t_first, "t_last": self.t_last, "t_end": self.t_end, "n_msgs": self.n_msgs,
                "n_tools": self.n_tools, "prompt_chars": len(self.prompt_text), "est_prompt": self.est_prompt,
                "prompt_tokens": prompt_total(u), "cached_tokens": cached,
                "completion_tokens": u.get("completion_tokens") or u.get("output_tokens"), "out_tok": self.out_tok,
                "out_tok_r": self.out_tok_r, "v": self.v, "v_rate": self.v_rate, "vreq": self.vreq,
                "out_chars": sum(map(len, self.content)), "reasoning_chars": sum(map(len, self.reasoning)),
                "tool_calls": [t["name"] for t in self.tools.values()], "finish": self.finish,
                "err": (self.err or "")[:300], "params": self.params, "last_input": self.last_input,
                "resp_id": self.resp_id}

    def detail(self):
        d = self.summary()
        d.update(body=self.body, content="".join(self.content), reasoning="".join(self.reasoning),
                 tools=list(self.tools.values()), usage=self.usage, err=self.err)
        return d


class Past:
    """A finished request reloaded from the logs, so the dashboard keeps its history across monitor restarts."""

    def __init__(self, rid, d):
        self.d = dict(d, id=rid)
        self.id, self.t_end, self.dirty, self.logged = rid, d.get("t_end") or d.get("t0"), False, True

    def summary(self):
        return {k: v for k, v in self.d.items() if k not in ("body", "content", "reasoning", "tools", "usage")}

    def detail(self):
        return self.d


class Monitor:
    def __init__(self, args):
        self.args = args
        self.upstream = args.upstream.rstrip("/")
        self.recs = collections.OrderedDict()
        self.seq = itertools.count(1)
        self.subs = set()
        self.server = {}
        self.journal = []
        self.pool = ThreadPoolExecutor(3)
        self.tok = None
        self.ctx = ctxedit.ContextEditor(self.count)
        self.key = open(args.key_file).read().strip()
        self.cookie = hashlib.sha256(("vmon:" + self.key).encode()).hexdigest()
        self.session = None
        self.prev_counters = None
        self.ratio = 1.0
        self.prog_mtime = None          # progress feed of the engine (vllm_turing progress hook)
        self.prog_last = None
        self.prompt_live = None
        self.prog_seen = 0.0
        self.kv_hist = collections.deque(maxlen=60)   # KV-usage fallback for the live prompt rate
        self.kv_den = None
        self.kv_prev = None
        self.kv_win = collections.deque()
        self.block_size = None
        os.makedirs(args.log_dir, exist_ok=True)

    # ---- tokenizer (estimates only, off the event loop) ------------------------------------------------------------
    def load_tokenizer(self):
        try:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(os.path.join(self.args.model_dir, "tokenizer.json"))
        except Exception as e:   # noqa: BLE001
            log.warning("no tokenizer (%s): token counts are estimated from characters", e)

    def count(self, text):
        if not text:
            return 0
        if self.tok is None:
            return int(len(text) / 3.2)
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    async def estimate_prompt(self, rec):
        n = await asyncio.get_running_loop().run_in_executor(self.pool, self.count, rec.prompt_text)
        if rec.fwd is not rec.body:     # context-edited: the estimate is the edited size (edit_context)
            return
        rec.raw_est = n
        rec.est_prompt = int(rec.raw_est * self.ratio)
        rec.dirty = True

    def calibrate(self, rec):
        """The chat template (roles, rendered tool schemas) adds tokens the text count misses: track the ratio of the
        real prompt size (usage) to the text count over large prompts and scale the estimates of running requests."""
        u = rec.usage or {}
        real = prompt_total(u)
        if real and rec.raw_est and rec.raw_est > 2000:
            self.ratio = 0.7 * self.ratio + 0.3 * min(max(real / rec.raw_est, 0.8), 1.6)

    # ---- records / events ------------------------------------------------------------------------------------------
    def add(self, rec):
        self.recs[rec.id] = rec
        while len(self.recs) > self.args.keep:
            self.recs.popitem(last=False)
        self.publish({"t": "s", "rec": rec.summary()})
        asyncio.get_running_loop().create_task(self.estimate_prompt(rec))

    def publish(self, ev):
        msg = json.dumps(ev, ensure_ascii=False)
        for q in list(self.subs):
            if q.qsize() < 2000:
                q.put_nowait(msg)

    async def publisher(self):
        """Every 0.4 s: new output text of active requests, state changes."""
        while True:
            await asyncio.sleep(0.4)
            for rec in list(self.recs.values()):
                if not rec.dirty:
                    continue
                rec.dirty = False
                c, r = "".join(rec.content), "".join(rec.reasoning)
                targs = "".join(t["args"] for t in rec.tools.values())
                nc, nr, nt = c[rec.sent[0]:], r[rec.sent[1]:], targs[rec.sent_tools:]
                if nc or nr or nt:
                    tr = self.count(nr)
                    rec.out_tok += self.count(nc) + tr + self.count(nt)
                    rec.out_tok_r += tr
                rec.sent, rec.sent_tools = [len(c), len(r)], len(targs)
                self.publish({"t": "p", "id": rec.id, "c": nc, "r": nr, "tools": list(rec.tools.values()),
                              "rec": rec.summary()})
                if rec.t_end is not None and not getattr(rec, "logged", False):
                    rec.logged = True
                    self.calibrate(rec)
                    asyncio.get_running_loop().run_in_executor(self.pool, self.write_log, rec.detail())

    def write_log(self, d):
        try:
            day = time.strftime("%Y-%m-%d", time.localtime(d["t0"]))
            path = os.path.join(self.args.log_dir, f"requests-{day}.jsonl.gz")
            with gzip.open(path, "ab") as f:                 # one gzip member per record: a valid concatenation
                f.write((json.dumps(d, ensure_ascii=False) + "\n").encode())
        except Exception as e:   # noqa: BLE001
            log.warning("log write failed: %s", e)

    def read_history(self):
        """The last --keep records of the two newest log files (a record cut by a kill at the end is skipped)."""
        lines = collections.deque(maxlen=self.args.keep)
        for f in sorted(f for f in os.listdir(self.args.log_dir) if f.startswith("requests-"))[-2:]:
            try:
                with gzip.open(os.path.join(self.args.log_dir, f), "rb") as fh:
                    for line in fh:
                        lines.append(line)
            except Exception:   # noqa: BLE001
                pass
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:   # noqa: BLE001
                pass
        return out

    async def load_history(self):
        past = await asyncio.get_running_loop().run_in_executor(self.pool, self.read_history)
        for i, d in enumerate(reversed(past)):          # newest = -1
            rec = Past(-(i + 1), d)
            self.recs[rec.id] = rec
        log.info("reloaded %d finished requests from the logs", len(past))

    def prune_logs(self):
        files = sorted((os.path.join(self.args.log_dir, f) for f in os.listdir(self.args.log_dir)
                        if f.startswith("requests-")), key=os.path.getmtime)
        now, total = time.time(), sum(os.path.getsize(f) for f in files)
        for f in files:
            if now - os.path.getmtime(f) > self.args.retain_days * 86400 or total > self.args.max_log_gb * 2**30:
                total -= os.path.getsize(f)
                os.remove(f)

    # ---- server state ----------------------------------------------------------------------------------------------
    async def poll_server(self):
        while True:
            st = {"t": time.time()}
            try:
                async with self.session.get(self.upstream + "/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    st["health"] = r.status
            except Exception as e:   # noqa: BLE001
                st["health"] = f"down ({type(e).__name__})"
            try:
                async with self.session.get(self.upstream + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    st.update(self.parse_metrics(await r.text()))
            except Exception:   # noqa: BLE001
                pass
            try:
                p = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,"
                    "power.draw,clocks.sm", "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
                out, _ = await p.communicate()
                st["gpus"] = [dict(zip(("i", "util", "mem", "mem_total", "temp", "power", "sm_mhz"),
                                       [x.strip() for x in line.split(",")]))
                              for line in out.decode().strip().splitlines()]
            except Exception:   # noqa: BLE001
                pass
            st["active"] = sum(1 for r in self.recs.values() if r.t_end is None)
            # live prompt processing rate: exact from the engine progress feed when it is updating, else estimated
            # from KV-block growth
            feed = self.prog_mtime is not None and (time.time() - self.prog_seen < 5 or not st["active"])
            st["progress_feed"] = feed
            st["prompt_live"] = (self.prompt_live or 0.0) if feed else None
            self.server = st
            self.publish({"t": "srv", "srv": st})
            await asyncio.sleep(1)

    async def poll_progress(self):
        """Engine progress feed (vllm_turing/progress.py): per-request computed / cached prompt tokens, and the
        cumulative computed prompt tokens for the live prompt rate."""
        while True:
            await asyncio.sleep(0.25)
            try:
                mtime = os.stat(PROGRESS_FILE).st_mtime
            except OSError:
                continue
            if mtime == self.prog_mtime:
                continue
            self.prog_mtime, self.prog_seen = mtime, time.time()
            try:
                with open(PROGRESS_FILE) as f:
                    data = json.load(f)
            except Exception:   # noqa: BLE001
                continue
            t, total = data.get("t", 0.0), data.get("prefill_total", 0)
            reqs = data.get("reqs") or []
            # the engine advances computed tokens a whole chunk at a time (6720 tokens, ~3 s at depth): rates are
            # averages, over the last 5 s (server) and over the request's prefill so far (request)
            hist = self.prog_last if isinstance(self.prog_last, collections.deque) else collections.deque()
            hist.append((t, total))
            while len(hist) > 2 and t - hist[0][0] > 5:
                hist.popleft()
            self.prog_last = hist
            if len(hist) >= 2 and t > hist[0][0] and total >= hist[0][1]:
                self.prompt_live = (total - hist[0][1]) / (t - hist[0][0])
            if not any(e.get("computed", 0) < e.get("prompt", 0) for e in reqs):
                self.prompt_live = 0.0
            active = [r for r in self.recs.values() if isinstance(r, Rec) and r.t_end is None and r.vreq]
            for e in reqs:
                rec = next((r for r in active if r.vreq in e.get("id", "")), None)
                if rec is None:
                    continue
                c = e.get("computed", 0)
                if rec.v_seen is None and c > 0:
                    rec.v_seen = (t, c)              # first scheduled: cached prefix + the first chunk
                elif rec.v_seen is not None and t > rec.v_seen[0] and c > rec.v_seen[1]:
                    rec.v_rate = (c - rec.v_seen[1]) / (t - rec.v_seen[0])
                rec.v = e
                rec.dirty = True

    def _infer_kv_den(self):
        """kv_cache_usage_perc is used / total blocks: the smallest total that makes every observed value whole."""
        vals = sorted(set(v for v in self.kv_hist if v > 0))
        for n in range(20, 5000):
            if all(abs(v * n - round(v * n)) < 0.02 for v in vals):
                return n
        return None

    def parse_metrics(self, text):
        vals = collections.defaultdict(float)
        for line in text.splitlines():
            if not line.startswith("vllm:"):
                continue
            m = re.match(r"(vllm:[a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+|NaN)$", line)
            if m and m.group(3) != "NaN":
                vals[m.group(1)] += float(m.group(3))
        out = {"running": vals.get("vllm:num_requests_running"), "waiting": vals.get("vllm:num_requests_waiting"),
               "kv_usage": vals.get("vllm:kv_cache_usage_perc", vals.get("vllm:gpu_cache_usage_perc"))}
        now = time.time()
        ctr = {k: vals.get(k, 0.0) for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                                             "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                                             "vllm:spec_decode_num_drafts_total",
                                             "vllm:spec_decode_num_accepted_tokens_total")}
        if self.prev_counters:
            t0, prev = self.prev_counters
            dt = max(now - t0, 1e-3)
            out["prompt_tps"] = (ctr["vllm:prompt_tokens_total"] - prev["vllm:prompt_tokens_total"]) / dt
            out["gen_tps"] = (ctr["vllm:generation_tokens_total"] - prev["vllm:generation_tokens_total"]) / dt
        q = ctr["vllm:prefix_cache_queries_total"]
        out["prefix_hit"] = ctr["vllm:prefix_cache_hits_total"] / q if q else None
        drafts = ctr["vllm:spec_decode_num_drafts_total"]
        out["accept_len"] = 1 + ctr["vllm:spec_decode_num_accepted_tokens_total"] / drafts if drafts else None
        self.prev_counters = (now, ctr)
        # fallback live prompt rate (vLLM's token counters only move when a request's first output token appears):
        # prefill chunks allocate KV blocks as they are computed; jumps of more than ~8 blocks per second are blocks
        # attached when a request is scheduled (prefix-cache hits, linear-attention state blocks), not computation
        bs = re.search(r'vllm:cache_config_info\{[^}]*\bblock_size="(\d+)"', text)
        if bs:
            self.block_size = int(bs.group(1))
        kv = out["kv_usage"]
        if kv:
            self.kv_hist.append(round(kv, 6))
            if self.kv_den is None and len(set(self.kv_hist)) >= 3:
                self.kv_den = self._infer_kv_den()
        if kv is not None and self.kv_den and self.kv_prev is not None:
            db = round((kv - self.kv_prev[1]) * self.kv_den)
            if 0 < db <= 8 * max(now - self.kv_prev[0], 1.0):
                self.kv_win.append((now, db))
        if kv is not None:
            self.kv_prev = (now, kv)
        while self.kv_win and now - self.kv_win[0][0] > 8:
            self.kv_win.popleft()
        prefilling = any(isinstance(r, Rec) and r.t_end is None and r.t_first is None for r in self.recs.values())
        out["prompt_live_kv"] = (sum(b for _, b in self.kv_win) * self.block_size / 8.0
                                 if prefilling and self.kv_den and self.block_size else 0.0)
        out["kv_blocks"] = self.kv_den
        return out

    async def poll_journal(self):
        while True:
            try:
                p = await asyncio.create_subprocess_exec(
                    "journalctl", "-u", "vllm-server", "-n", "400", "--no-pager", "-o", "short-iso",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await p.communicate()
                lines = [ln for ln in out.decode("utf-8", "replace").splitlines()
                         if re.search(r"\b(WARNING|ERROR|CRITICAL)\b|Traceback|Error:|out of memory", ln)
                         and "FA version 2" not in ln]
                self.journal = lines[-80:] if p.returncode == 0 else [err.decode()[:300]]
            except Exception as e:   # noqa: BLE001
                self.journal = [f"journalctl unavailable: {e}"]
            await asyncio.sleep(10)

    async def housekeeping(self):
        while True:
            await asyncio.get_running_loop().run_in_executor(self.pool, self.prune_logs)
            await asyncio.sleep(3600)

    # ---- proxy -----------------------------------------------------------------------------------------------------
    async def count_prompt(self, rec, headers):
        """Prompt length of a request as vLLM renders and tokenizes it (Anthropic count_tokens, /tokenize), or None."""
        b = rec.fwd if isinstance(rec.fwd, dict) else {}
        if rec.kind == "anthropic":
            path, keys = "/v1/messages/count_tokens", ("model", "messages", "system", "tools", "tool_choice",
                                                       "chat_template_kwargs")
        elif rec.kind == "chat":
            path, keys = "/tokenize", ("model", "messages", "tools", "chat_template", "chat_template_kwargs")
        elif rec.kind == "completions" and isinstance(b.get("prompt"), str):
            path, keys = "/tokenize", ("model", "prompt")
        else:
            return None
        h = {k: v for k, v in headers.items() if k.lower() not in ("x-request-id", "content-type")}
        try:
            async with self.session.post(self.upstream + path, json={k: b[k] for k in keys if k in b},
                                         headers=h) as r:
                d = await r.json(content_type=None)
            n = d.get("input_tokens") if rec.kind == "anthropic" else d.get("count")
            return n if isinstance(n, int) else None
        except Exception as e:   # noqa: BLE001
            log.warning("rec %s: prompt count failed: %s: %s", rec.id, type(e).__name__, e)
            return None

    async def fit_output(self, rec, headers, m, attempt):
        """For vLLM's prompt + max_tokens > context error (match m): ("retry", body with max_tokens = the room left),
        ("error", Anthropic "prompt is too long" body) or None (pass vLLM's error on)."""
        total, asked, prompt = int(m[1]), int(m[2]), int(m[4])
        if m[3]:            # "at least": vLLM's tokenizer stopped at the input limit, the real length is unknown
            prompt = max(prompt, await self.count_prompt(rec, headers) or 0)
        room = total - prompt - FIT_MARGIN * (attempt + 1)
        if room >= MIN_OUTPUT:
            b = dict(rec.fwd)
            key = next((k for k in ("max_completion_tokens", "max_output_tokens") if k in b), "max_tokens")
            b[key] = room
            th = b.get("thinking")
            if isinstance(th, dict) and isinstance(th.get("budget_tokens"), int) and th["budget_tokens"] >= room:
                b["thinking"] = dict(th, budget_tokens=max(1024, room // 2))
            rec.params["max_tokens_sent"] = room
            rec.dirty = True
            log.info("rec %s: prompt %d + %s %d > %d: resent with %d", rec.id, prompt, key, asked, total, room)
            return "retry", json.dumps(b, ensure_ascii=False).encode()
        log.info("rec %s: prompt %d leaves %d of %d tokens for the output", rec.id, prompt, total - prompt, total)
        if rec.kind == "anthropic":
            return "error", json.dumps({"type": "error", "error": {
                "type": "invalid_request_error",
                "message": f"prompt is too long: {prompt} tokens > {total - MIN_OUTPUT} maximum"}}).encode()
        return None

    async def edit_context(self, rec, body):
        """The request body to forward: a long agent conversation with its older tool traffic cleared (ctxedit.py), else
        the client's body unchanged (also when the edit fails: the proxy must not break a request)."""
        try:
            edited, info = await asyncio.get_running_loop().run_in_executor(self.pool, self.ctx.edit, rec.kind,
                                                                            rec.body)
        except asyncio.CancelledError:
            raise
        except Exception:   # noqa: BLE001
            log.exception("rec %s: context edit failed, forwarding the request unchanged", rec.id)
            return body
        if edited is None:
            return body
        rec.fwd = edited
        rec.params["ctx_edit"] = info
        rec.raw_est = info["sent_tokens"]
        rec.est_prompt = int(rec.raw_est * self.ratio)
        rec.dirty = True
        log.info("rec %s: context edit %s", rec.id, json.dumps(info))
        return json.dumps(edited, ensure_ascii=False).encode()

    def strip_billing_header(self, rec, body):
        """Claude Code's first system block ("x-anthropic-billing-header: cc_version=<version>.<suffix>; ...") changes from
        session to session and renders right after the tool definitions (~16k tokens): every new sub-agent diverged
        there from all earlier sessions, past the only GDN state vLLM keeps of a prompt (near its end), so none could
        reuse another's prefill (10-23: sub-agent starts 20k tokens uncached each, waves of 10+). Without it sessions
        share tools + system prompt up to their own messages (test: 16.7-17.9k of 18k cached, first token 4.9 -> 0.2-0.5
        s). The header means nothing to a local model."""
        b = rec.fwd
        s = b.get("system") if isinstance(b, dict) else None
        if not (isinstance(s, list) and s and isinstance(s[0], dict)
                and str(s[0].get("text", "")).startswith("x-anthropic-billing-header:")):
            return body
        rec.fwd = b = dict(b, system=s[1:])
        rec.params["billing_header_dropped"] = True
        rec.dirty = True
        return json.dumps(b, ensure_ascii=False).encode()

    def no_thinking(self, rec, body):
        """Anthropic semantics: a request without `thinking` gets no extended thinking. Qwen's chat template thinks unless
        enable_thinking is false (and turns effort "high" into its maximum, xhigh): Claude Code's helper calls (no tools,
        no `thinking`: web page summaries and the like) spent ~5000 reasoning tokens each (10-23: 95 % of their output,
        84 calls with 3x the generation time of 350 agent turns, the queue behind them up to 15 requests). Requests that
        ask for thinking (adaptive / enabled) or set chat_template_kwargs themselves are forwarded unchanged.
        --keep-thinking (env VMON_KEEP_THINKING=1) turns this off."""
        b = rec.fwd
        if (self.args.keep_thinking or not isinstance(b, dict) or b.get("thinking") is not None
                or "chat_template_kwargs" in b):
            return body
        rec.fwd = b = dict(b, chat_template_kwargs={"enable_thinking": False})
        rec.params["thinking_off"] = True
        rec.dirty = True
        return json.dumps(b, ensure_ascii=False).encode()

    async def request_upstream(self, request, headers, body, rec):
        """The upstream request. A generation request that finds vLLM down (restart: connection refused) waits for it
        up to UPSTREAM_WAIT seconds instead of failing: clients see a slow answer rather than an error."""
        t0 = time.time()
        while True:
            try:
                up = await self.session.request(request.method, self.upstream + request.path_qs, headers=headers,
                                                data=body or None, allow_redirects=False)
            except aiohttp.ClientConnectorError:
                if rec is None or time.time() - t0 > UPSTREAM_WAIT:
                    raise
                if "upstream_wait" not in rec.params:
                    log.info("rec %s: vLLM unreachable, holding the request until it is back", rec.id)
                rec.params["upstream_wait"] = round(time.time() - t0)
                rec.dirty = True
                await asyncio.sleep(2)
                continue
            if rec and "upstream_wait" in rec.params:
                log.info("rec %s: vLLM back after %.0f s", rec.id, time.time() - t0)
            return up

    async def proxy(self, request):
        body = await request.read()
        kind = GEN_PATHS.get(request.path) if request.method == "POST" else None
        rec = None
        if kind:
            rec = Rec(next(self.seq), request, body, kind)
            self.add(rec)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        headers["Accept-Encoding"] = "identity"
        if rec:
            # vLLM takes X-Request-Id as its request id: the engine progress feed names the request (a client's own
            # id is kept)
            xr = next((v for k, v in headers.items() if k.lower() == "x-request-id"), None)
            if xr is None:
                xr = headers["X-Request-Id"] = f"vmon{int(rec.t0)}-{rec.id}"
            rec.vreq = xr
        data = None         # body of a non-streamed response, when already read
        try:
            if rec and rec.kind in ("anthropic", "chat") and ctxedit.ENABLED:
                body = await self.edit_context(rec, body)
            if rec and rec.kind == "anthropic":
                body = self.strip_billing_header(rec, body)
                body = self.no_thinking(rec, body)
            for attempt in range(3):
                up = await self.request_upstream(request, headers, body, rec)
                if not (rec and up.status == 400 and "text/event-stream" not in up.headers.get("Content-Type", "")):
                    break
                data = await up.read()
                m = CTX_RE.search(data)
                if m is None:
                    break
                fit = await self.fit_output(rec, headers, m, attempt)
                if fit is None:
                    break
                if fit[0] == "error":
                    data = fit[1]
                    break
                up.release()
                body, data = fit[1], None
        except asyncio.CancelledError:
            if rec:
                rec.err = "client disconnected while waiting for vLLM"
                rec.end("aborted")
            raise
        except Exception as e:   # noqa: BLE001
            if rec:
                rec.fail(f"vLLM unreachable: {type(e).__name__}: {e}", 502)
            return web.json_response({"error": {"message": f"upstream unavailable: {e}", "type": "proxy_error"}},
                                     status=502)
        aborted = False
        try:
            rh = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
            if "text/event-stream" in up.headers.get("Content-Type", ""):
                resp = web.StreamResponse(status=up.status, headers=rh)
                await resp.prepare(request)
                if rec:
                    rec.status = up.status
                at_boundary = True          # the bytes forwarded so far end at an SSE event boundary
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(up.content.readany(), KEEPALIVE)
                        except asyncio.TimeoutError:
                            # vLLM sends nothing until the first output token (a long prompt takes a minute): an SSE
                            # comment (ignored by every SSE parser, unlike a ping event before message_start) keeps
                            # client / VPN idle timers from dropping the connection
                            if at_boundary:
                                await resp.write(b": keep-alive\n\n")
                            continue
                        if not chunk:
                            break
                        await resp.write(chunk)
                        at_boundary = chunk.endswith(b"\n\n")
                        if rec:
                            rec.feed(chunk)
                    await resp.write_eof()
                except (asyncio.CancelledError, ConnectionResetError) as e:
                    aborted = True
                    if rec:
                        rec.err = "client disconnected during the stream"
                        rec.end("aborted")
                    if isinstance(e, asyncio.CancelledError):
                        raise
                    return resp
                except aiohttp.ClientError as e:
                    if rec:
                        rec.fail(f"vLLM stream broke: {type(e).__name__}: {e}")
                    return resp
                if rec:
                    if rec.err or up.status >= 400:
                        rec.fail(rec.err or f"HTTP {up.status}", up.status)
                    else:
                        rec.end("done")
                return resp
            try:
                if data is None:
                    data = await up.read()
            except asyncio.CancelledError:
                aborted = True
                if rec:
                    rec.err = "client disconnected while waiting for vLLM"
                    rec.end("aborted")
                raise
            if rec:
                rec.complete_json(up.status, data)
            return web.Response(status=up.status, headers=rh, body=data)
        finally:
            if aborted:
                up.close()          # drop the connection: vLLM sees the disconnect and aborts the request
            else:
                up.release()

    # ---- dashboard -------------------------------------------------------------------------------------------------
    def authed(self, request):
        if request.cookies.get("vmon") == self.cookie:
            return True
        auth = request.headers.get("Authorization", "")
        return auth == "Bearer " + self.key or request.query.get("key") == self.key

    async def page(self, request):
        if request.query.get("key") == self.key:
            resp = web.Response(status=302, headers={"Location": "/monitor"})
            resp.set_cookie("vmon", self.cookie, max_age=30 * 86400, httponly=True, samesite="Strict")
            return resp
        if not self.authed(request):
            return web.Response(status=401, content_type="text/html",
                                text="<p>vLLM monitor: open <code>/monitor?key=&lt;API key&gt;</code> once to log in.</p>")
        return web.FileResponse(os.path.join(HERE, "monitor.html"), headers={"Cache-Control": "no-store"})

    async def api_list(self, request):
        if not self.authed(request):
            raise web.HTTPUnauthorized()
        return web.json_response({"recs": [r.summary() for r in self.recs.values()], "srv": self.server,
                                  "now": time.time()}, dumps=lambda o: json.dumps(o, ensure_ascii=False))

    async def api_req(self, request):
        if not self.authed(request):
            raise web.HTTPUnauthorized()
        rec = self.recs.get(int(request.match_info["rid"]))
        if rec is None:
            raise web.HTTPNotFound()
        return web.json_response(rec.detail(), dumps=lambda o: json.dumps(o, ensure_ascii=False))

    async def api_journal(self, request):
        if not self.authed(request):
            raise web.HTTPUnauthorized()
        return web.json_response({"lines": self.journal})

    async def api_events(self, request):
        if not self.authed(request):
            raise web.HTTPUnauthorized()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        q = asyncio.Queue()
        self.subs.add(q)
        try:
            await resp.write(f"data: {json.dumps({'t': 'hello', 'now': time.time()})}\n\n".encode())
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), 15)
                    await resp.write(f"data: {msg}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": ping\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self.subs.discard(q)
        return resp

    async def start(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=64, keepalive_timeout=30), auto_decompress=False)
        await asyncio.get_running_loop().run_in_executor(self.pool, self.load_tokenizer)
        for coro in (self.publisher(), self.poll_server(), self.poll_progress(), self.poll_journal(),
                     self.housekeeping(), self.load_history()):
            asyncio.get_running_loop().create_task(coro)

    async def stop(self, app):
        await self.session.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:8080")
    ap.add_argument("--upstream", default="http://127.0.0.1:8090")
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/Qwen3.8-27B-SQ06"))
    ap.add_argument("--key-file", default=os.path.expanduser("~/models/api-key.txt"))
    ap.add_argument("--log-dir", default=os.path.expanduser("~/monitor/logs"))
    ap.add_argument("--keep", type=int, default=200, help="requests kept in memory for the dashboard (agent requests ~1 MB each)")
    ap.add_argument("--retain-days", type=int, default=14)
    ap.add_argument("--max-log-gb", type=float, default=5.0)
    ap.add_argument("--keep-thinking", action="store_true", default=os.environ.get("VMON_KEEP_THINKING") == "1",
                    help="forward Anthropic requests without `thinking` unchanged (the model then thinks at full depth)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mon = Monitor(args)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(mon.start)
    app.on_cleanup.append(mon.stop)
    app.router.add_get("/monitor", mon.page)
    app.router.add_get("/monitor/api/list", mon.api_list)
    app.router.add_get("/monitor/api/req/{rid}", mon.api_req)
    app.router.add_get("/monitor/api/journal", mon.api_journal)
    app.router.add_get("/monitor/api/events", mon.api_events)
    app.router.add_route("*", "/{tail:.*}", mon.proxy)
    host, port = args.listen.rsplit(":", 1)
    log.info("vLLM monitor on %s -> %s, dashboard /monitor, logs %s", args.listen, args.upstream, args.log_dir)
    web.run_app(app, host=host, port=int(port), access_log=None, handler_cancellation=True, print=None)


if __name__ == "__main__":
    main()
