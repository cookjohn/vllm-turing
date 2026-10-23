#!/usr/bin/env python3
"""Unit test of ctxedit.py on synthetic agent conversations (Anthropic messages and OpenAI chat, one token = 4
characters): nothing changes below the trigger; above it the older part loses its tool results (notes naming the call),
long tool arguments and thinking, while user / assistant texts, small results, short errors, excluded tools and the
recent part stay verbatim; growing the conversation turn by turn keeps the edited prefix identical except when the cut
advances (once per STEP tokens); moving cache_control markers do not move the cut. usage: test_ctxedit.py"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxedit   # noqa: E402

ctxedit.TRIGGER, ctxedit.KEEP, ctxedit.STEP = 20000, 5000, 6000
ed = ctxedit.ContextEditor(lambda s: len(s) // 4 + 1)
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))


def words(n, seed):
    return " ".join(f"w{(seed * 7919 + i * 31) % 1000}" for i in range(n))


TOOLS = ["Read", "Bash", "Grep", "Edit", "TodoWrite", "Write"]


def anthropic_step(k):
    """One agent step: assistant (thinking, text, tool call), user (tool result, reminder)."""
    name = TOOLS[k % len(TOOLS)]
    inp = {"file_path": f"/src/mod{k}.py"} if name in ("Read", "Edit", "Write") else {"command": f"make test{k}"}
    if name in ("Write", "Edit"):
        inp["content"] = words(400, k)
    if name == "TodoWrite":
        inp = {"todos": [{"content": f"item {k}", "status": "pending"}]}
    result = words(10, k) if k % 7 == 3 else words(600, k)          # some small results
    is_err = k % 11 == 5
    if is_err:
        result = "Error: " + words(60, k)
    return [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": words(300, k + 1000), "signature": "sig"},
                {"type": "text", "text": f"Step {k}: looking at it."},
                {"type": "tool_use", "id": f"t{k}", "name": name, "input": inp}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{k}", "content": result, "is_error": is_err},
                {"type": "text", "text": f"<system-reminder>reminder {k}</system-reminder>"}]}]


def anthropic_conv(steps):
    msgs = [{"role": "user", "content": "Fix the failing tests in /src."}]
    for k in range(steps):
        msgs += anthropic_step(k)
    return {"model": "q", "max_tokens": 32000, "system": [{"type": "text", "text": words(500, 99)}],
            "tools": [{"name": t, "input_schema": {}} for t in TOOLS], "messages": msgs}


def chat_conv(steps):
    msgs = [{"role": "system", "content": words(500, 99)}, {"role": "user", "content": "Fix the failing tests."}]
    for k in range(steps):
        name = ["read", "bash", "todowrite"][k % 3]
        msgs.append({"role": "assistant", "content": f"Step {k}.", "reasoning_content": words(300, k),
                     "tool_calls": [{"id": f"c{k}", "type": "function", "function": {
                         "name": name, "arguments": json.dumps({"filePath": f"/src/f{k}.ts", "content": words(300, k)})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{k}", "content": words(600, k)})
    return {"model": "q", "max_tokens": 32000, "messages": msgs}


def texts(body):
    """User / assistant texts (what must survive), as a list."""
    out = []
    for m in body["messages"]:
        c = m.get("content")
        if isinstance(c, str):
            out.append((m["role"], c))
        elif isinstance(c, list):
            out += [(m["role"], b["text"]) for b in c if b.get("type") == "text"]
    return out


# ---- below the trigger ---------------------------------------------------------------------------------------------
small = anthropic_conv(10)
check("below the trigger: unchanged", ed.edit("anthropic", small)[0] is None)

# ---- above the trigger ---------------------------------------------------------------------------------------------
big = anthropic_conv(60)
orig = copy.deepcopy(big)
out, info = ed.edit("anthropic", big)
check("above the trigger: edited", out is not None, json.dumps(info))
check("the request body itself is not modified", big == orig)
cut = info["cut_msg"]
check("cut leaves KEEP..KEEP+STEP tokens verbatim",
      ctxedit.KEEP <= info["orig_tokens"] - info["cut_at"] < ctxedit.KEEP + ctxedit.STEP, f"cut at message {cut}")
check("user and assistant texts all kept", texts(out) == texts(big))
check("messages after the cut verbatim", out["messages"][cut:] == big["messages"][cut:])
res = [(i, b) for i, m in enumerate(out["messages"][:cut]) if isinstance(m["content"], list)
       for b in m["content"] if b.get("type") == "tool_result"]
orig_res = {b["tool_use_id"]: b for m in big["messages"] for b in (m["content"] if isinstance(m["content"], list)
                                                                    else []) if b.get("type") == "tool_result"}
cleared = [b for _, b in res if str(b["content"]).startswith(ctxedit.CLEARED)]
kept = [b for _, b in res if not str(b["content"]).startswith(ctxedit.CLEARED)]
k_of = lambda b: int(b["tool_use_id"][1:])   # noqa: E731
check("large results cleared, small / error / TodoWrite results kept",
      all(TOOLS[k_of(b) % 6] != "TodoWrite" and k_of(b) % 7 != 3 and k_of(b) % 11 != 5 for b in cleared) and
      all(TOOLS[k_of(b) % 6] == "TodoWrite" or k_of(b) % 7 == 3 or k_of(b) % 11 == 5 for b in kept) and
      all(b == orig_res[b["tool_use_id"]] for b in kept), f"{len(cleared)} cleared, {len(kept)} kept")
note = cleared[0]["content"]
k0 = k_of(cleared[0])
want = "file_path=/src/mod%d.py" % k0 if TOOLS[k0 % 6] in ("Read", "Edit", "Write") else "command=make test%d" % k0
check("the note names the tool and its file / command", TOOLS[k0 % 6] + "(" + want + ")" in note, note)
think = sum(b.get("type") == "thinking" for m in out["messages"][:cut] if isinstance(m["content"], list)
            for b in m["content"])
check("thinking dropped before the cut, kept after",
      think == 0 and sum(b.get("type") == "thinking" for m in out["messages"][cut:] if isinstance(m["content"], list)
                         for b in m["content"]) > 0)
inputs = [b["input"] for m in out["messages"][:cut] if isinstance(m["content"], list) for b in m["content"]
          if b.get("type") == "tool_use"]
check("long tool inputs shortened, file paths kept",
      all(len(v) < ctxedit.INPUT_CHARS + 100 for i in inputs for v in i.values() if isinstance(v, str)) and
      all("file_path" in i for i in inputs if "content" in i))
check("sent_tokens < orig_tokens", info["sent_tokens"] < info["orig_tokens"] * 0.6, f"{info['orig_tokens']} -> "
      f"{info['sent_tokens']}")

# ---- prefix stability while the conversation grows -----------------------------------------------------------------
prev, jumps, worst, resent = None, 0, 0, []
for steps in range(40, 160):
    body = anthropic_conv(steps)
    out, info = ed.edit("anthropic", body)
    msgs = (out or body)["messages"]
    if out:
        worst = max(worst, info["sent_tokens"])
    if prev is not None:
        n = len(prev[0])
        same = next((i for i in range(n) if msgs[i] != prev[0][i]), n)
        if same < n - 1:             # more than the previous last message changed: the cut moved
            jumps += 1
            resent.append(same)
            if prev[1] is not None and same < prev[1]:
                check("a cut move keeps the prefix up to the previous cut", False, f"steps {steps}: {same} < {prev[1]}")
    prev = (msgs, info["cut_msg"] if out else None)
total = sum(ed.msg_tokens("anthropic", m) for m in anthropic_conv(159)["messages"])
check("cut moves rarely (once per STEP tokens)", 0 < jumps <= total // ctxedit.STEP + 1,
      f"{jumps} moves over 120 turns ({total} tokens)")
check("sent prompt stays bounded", worst < ctxedit.KEEP + ctxedit.STEP + 0.25 * total, f"max sent {worst}")

# ---- moving cache_control does not move the cut --------------------------------------------------------------------
b1 = anthropic_conv(60)
b2 = copy.deepcopy(b1)
for m in b2["messages"][-3:]:
    if isinstance(m["content"], list):
        m["content"][-1]["cache_control"] = {"type": "ephemeral"}
c1, c2 = ed.edit("anthropic", b1)[1]["cut_msg"], ed.edit("anthropic", b2)[1]["cut_msg"]
check("cache_control markers do not shift the cut", c1 == c2, f"{c1} / {c2}")

# ---- OpenAI chat ---------------------------------------------------------------------------------------------------
cb = chat_conv(40)
out, info = ed.edit("chat", cb)
cut = info["cut_msg"] if info else 0
ok = out is not None
if ok:
    pre = out["messages"][:cut]
    ok = (all(not m.get("reasoning_content") for m in pre if m["role"] == "assistant") and
          all(m["content"].startswith(ctxedit.CLEARED) for i, m in enumerate(pre) if m["role"] == "tool" and
              pre[i - 1]["tool_calls"][0]["function"]["name"] != "todowrite") and
          all(not m["content"].startswith(ctxedit.CLEARED) for i, m in enumerate(pre) if m["role"] == "tool" and
              pre[i - 1]["tool_calls"][0]["function"]["name"] == "todowrite") and
          all(json.loads(m["tool_calls"][0]["function"]["arguments"])["filePath"].startswith("/src/") for m in pre
              if m["role"] == "assistant") and
          out["messages"][cut:] == cb["messages"][cut:] and out["messages"][0] == cb["messages"][0])
check("chat: tool messages cleared (todowrite kept), reasoning dropped, args shortened, rest verbatim", ok,
      json.dumps(info))

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
