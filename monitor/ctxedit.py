"""Server-side context editing for long agent conversations (Claude Code over /v1/messages, opencode over
/v1/chat/completions) in the monitoring proxy. The client keeps sending its full history; once the prompt passes TRIGGER
tokens the proxy forwards a copy in which the older part of the conversation keeps every user message and every
assistant text, but not the bulk of the tool traffic:
- tool results become a note that names the call ("[Output of Read(file_path=...) cleared ... Run the tool again if you
  need it.]"): the agent can get the content back (restorable compression, as Manus describes it); short results and
  short errors stay, and so do the results of EXCLUDE tools (todo list, subagent reports, skills, questions to the user)
- long string arguments of tool calls (file contents written, long commands) keep their first characters only
- thinking blocks / reasoning are dropped.
The same as the Anthropic API's clear_tool_uses / clear_thinking context edits, i.e. the observation masking that did as
well as LLM summarization for coding agents at lower cost (Lindenbauer et al., "The Complexity Trap", NeurIPS 2025). In
a 230K-token Claude Code prompt, tool results / tool inputs / earlier thinking were 83K / 56K / 62K tokens, the text of
the user and the agent 7.6K.

The edit is a pure function of the request (no server state, survives restarts): token offsets are counted over the
original history (contents only, so moving cache_control markers do not shift them), and the older part ends at the
last message boundary below X = floor((T - KEEP) / STEP) * STEP, T = the original prompt size. X stays put while the
conversation grows by STEP tokens, so consecutive requests share their edited prefix (vLLM prefix cache hits); when X
moves, the cache is invalidated from the previous cut on, once per STEP tokens (Anthropic's clear_at_least). The
prompt vLLM sees then stays between about KEEP and KEEP + STEP verbatim tokens + the cleared older part (~8% of its
original size in Claude Code sessions): a ~1M-token conversation fits in ~200K."""
import collections
import hashlib
import json
import os
import threading

ENABLED = os.environ.get("VMON_CTX_EDIT", "1") != "0"
TRIGGER = int(os.environ.get("VMON_CTX_TRIGGER", "160000"))     # original prompt tokens at which editing starts
KEEP = int(os.environ.get("VMON_CTX_KEEP", "40000"))             # the most recent tokens always stay verbatim
STEP = int(os.environ.get("VMON_CTX_STEP", "60000"))             # the cut advances in steps of this many tokens
MIN_RESULT = int(os.environ.get("VMON_CTX_MIN_RESULT", "100"))   # smaller tool results stay
MIN_ERROR = int(os.environ.get("VMON_CTX_MIN_ERROR", "1000"))    # smaller error results stay (the agent learns from them)
INPUT_CHARS = int(os.environ.get("VMON_CTX_INPUT_CHARS", "300"))  # longer tool-call string arguments are shortened
HEAD_CHARS = 120                                                  # ... to their first HEAD_CHARS characters
EXCLUDE = set(filter(None, os.environ.get(
    "VMON_CTX_EXCLUDE", "TodoWrite,Task,Agent,AskUserQuestion,ExitPlanMode,EnterPlanMode,Skill,todowrite,task,skill"
).split(",")))
IMAGE_TOKENS = 1500
KEY_ARGS = ("file_path", "filePath", "path", "notebook_path", "command", "pattern", "url", "query", "description",
            "prompt")
CLEARED = "[Output of "


def _dump(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True) if x is not None else ""


def _key_arg(inp):
    """The argument that identifies a tool call (file, command, pattern, URL), for the note that replaces its output."""
    if not isinstance(inp, dict):
        return ""
    for k in KEY_ARGS:
        v = inp.get(k)
        if isinstance(v, str) and v:
            v = " ".join(v.split())
            return f"{k}={v[:117] + '...' if len(v) > 120 else v}"
    return ""


def _shorten(x):
    """x with every string longer than INPUT_CHARS cut to its head; -> (new x, number of strings shortened)."""
    if isinstance(x, str):
        if len(x) > INPUT_CHARS:
            return f"{x[:HEAD_CHARS]}... [{len(x) - HEAD_CHARS} more characters cleared by the server]", 1
        return x, 0
    if isinstance(x, dict):
        out, n = {}, 0
        for k, v in x.items():
            out[k], m = _shorten(v)
            n += m
        return (out, n) if n else (x, 0)
    if isinstance(x, list):
        out, n = [], 0
        for v in x:
            v2, m = _shorten(v)
            out.append(v2)
            n += m
        return (out, n) if n else (x, 0)
    return x, 0


class ContextEditor:
    def __init__(self, count):
        self.count = count                              # text -> tokens
        self.cache = collections.OrderedDict()          # sha1(text) -> tokens
        self.lock = threading.Lock()

    def tokens(self, text):
        if not text:
            return 0
        h = hashlib.sha1(text.encode("utf-8", "surrogatepass")).digest()
        with self.lock:
            n = self.cache.get(h)
            if n is not None:
                self.cache.move_to_end(h)
                return n
        n = self.count(text)
        with self.lock:
            self.cache[h] = n
            while len(self.cache) > 300000:
                self.cache.popitem(last=False)
        return n

    # ---- sizes ---------------------------------------------------------------------------------------------------
    def content_tokens(self, c):
        """Tokens of a message / tool-result content: string or list of parts (text, images)."""
        if isinstance(c, str):
            return self.tokens(c)
        n = 0
        for p in c if isinstance(c, list) else [c] if c else []:
            if not isinstance(p, dict):
                n += self.tokens(str(p))
            elif isinstance(p.get("text"), str):
                n += self.tokens(p["text"])
            elif "image" in str(p.get("type", "")):
                n += IMAGE_TOKENS
            elif p.get("type") == "tool_result":
                n += self.content_tokens(p.get("content"))
            else:
                n += self.tokens(_dump({k: v for k, v in p.items() if k != "cache_control"}))
        return n

    def msg_tokens(self, kind, m):
        if not isinstance(m, dict):
            return self.tokens(_dump(m))
        if kind == "chat":
            n = self.content_tokens(m.get("content"))
            n += self.tokens(m.get("reasoning_content") or m.get("reasoning") or "")
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                n += self.tokens(fn.get("name") or "") + self.tokens(_dump(fn.get("arguments")))
            return n
        c = m.get("content")
        if not isinstance(c, list):
            return self.content_tokens(c)
        n = 0
        for b in c:
            t = b.get("type") if isinstance(b, dict) else None
            if t == "thinking":
                n += self.tokens(b.get("thinking") or "")
            elif t == "redacted_thinking":
                n += len(b.get("data") or "") // 4
            elif t == "tool_use":
                n += self.tokens(b.get("name") or "") + self.tokens(_dump(b.get("input")))
            else:
                n += self.content_tokens([b])
        return n

    # ---- edits ---------------------------------------------------------------------------------------------------
    def _note(self, name, inp, content, is_error):
        """The note replacing a tool result, or None when the result stays."""
        if name in EXCLUDE:
            return None
        if isinstance(content, str) and content.startswith(CLEARED):
            return None
        n = self.content_tokens(content)
        if n < (MIN_ERROR if is_error else MIN_RESULT):
            return None
        arg = _key_arg(inp)
        return (f"{CLEARED}{name}({arg}) cleared by the server to save context ({n} tokens). "
                f"Run the tool again if you need it.]"), n

    def _edit_anthropic(self, m, calls, st):
        c = m.get("content")
        if not isinstance(c, list):
            return m
        out, changed, dropped = [], 0, 0
        for b in c:
            t = b.get("type") if isinstance(b, dict) else None
            if t == "tool_use":
                calls[b.get("id")] = (b.get("name") or "?", b.get("input"))
                if b.get("name") not in EXCLUDE:
                    inp, k = _shorten(b.get("input"))
                    if k:
                        b = dict(b, input=inp)
                        st["inputs_shortened"] += 1
                        changed += 1
            elif t in ("thinking", "redacted_thinking") and m.get("role") == "assistant":
                dropped += 1
                continue
            elif t == "tool_result":
                name, inp = calls.get(b.get("tool_use_id"), ("?", None))
                note = self._note(name, inp, b.get("content"), b.get("is_error"))
                if note:
                    b = dict(b, content=note[0])
                    st["results_cleared"] += 1
                    changed += 1
            out.append(b)
        if dropped and not out:             # nothing but thinking in it: keep the message as it is
            return m
        if dropped:
            st["thinking_dropped"] += dropped
            changed += 1
        return dict(m, content=out) if changed else m

    def _edit_chat(self, m, calls, st):
        role = m.get("role")
        if role == "assistant":
            m2, changed = dict(m), False
            tcs = m.get("tool_calls") or []
            if tcs:
                new = []
                for tc in tcs:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except ValueError:
                        parsed = None
                    calls[tc.get("id")] = (fn.get("name") or "?", parsed)
                    if fn.get("name") not in EXCLUDE and isinstance(parsed, dict):
                        sh, k = _shorten(parsed)
                        if k:
                            tc = dict(tc, function=dict(fn, arguments=json.dumps(sh, ensure_ascii=False)))
                            st["inputs_shortened"] += 1
                            changed = True
                    new.append(tc)
                m2["tool_calls"] = new
            if (m.get("content") or tcs) and (m.get("reasoning_content") or m.get("reasoning")):
                m2.pop("reasoning_content", None)
                m2.pop("reasoning", None)
                st["thinking_dropped"] += 1
                changed = True
            return m2 if changed else m
        if role == "tool":
            name, inp = calls.get(m.get("tool_call_id"), (m.get("name") or "?", None))
            note = self._note(name, inp, m.get("content"), False)
            if note:
                st["results_cleared"] += 1
                return dict(m, content=note[0])
        return m

    def edit(self, kind, body):
        """-> (edited copy of the request body or None, info dict or None)"""
        if not ENABLED or kind not in ("anthropic", "chat") or not isinstance(body, dict):
            return None, None
        msgs = body.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 4:
            return None, None
        base = self.tokens(_dump(body.get("system"))) + self.tokens(_dump(body.get("tools")))
        sizes = [self.msg_tokens(kind, m) for m in msgs]
        total = base + sum(sizes)
        if total < TRIGGER:
            return None, None
        x = (total - KEEP) // STEP * STEP
        end, cut = base, 0
        for s in sizes:
            if end + s > x:
                break
            end += s
            cut += 1
        cut = min(cut, len(msgs) - 2)
        info = {"orig_tokens": total, "cut_msg": cut, "msgs": len(msgs), "cut_at": x}
        if cut <= 0:
            return None, info
        st = collections.Counter()
        calls = {}
        f = self._edit_anthropic if kind == "anthropic" else self._edit_chat
        new = [f(m, calls, st) if isinstance(m, dict) else m for m in msgs[:cut]] + msgs[cut:]
        if not st:
            return None, info
        info.update(st)
        info["sent_tokens"] = base + sum(self.msg_tokens(kind, m) for m in new[:cut]) + sum(sizes[cut:])
        return dict(body, messages=new), info
