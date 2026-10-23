#!/usr/bin/env python3
"""Prefix-cache check of Claude Code turns (Anthropic /v1/messages) under vLLM's conversion: renders two consecutive
captured requests (monitor detail JSON files) with (a) the model's chat template, where vLLM merges Claude Code's
inline system reminders into the leading system prompt, and (b) a template that renders non-leading system messages
in place (no merge); reports the first differing token of the two turns and how much of turn 1 is a prefix of
turn 2. Also checks that the patched template renders conversations without inline system messages exactly like the
original. Writes the patched template to --out.
usage: check_inline_system.py <turn1.json> <turn2.json> --model-dir DIR --out patched.jinja"""
import argparse
import json

from transformers import AutoTokenizer

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages as S

ORIG = """    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}"""
NEW = """    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {#- non-leading system messages (e.g. Claude Code's per-turn reminders) render in place, so vLLM does not
                merge them into the leading system prompt, which would change the prompt prefix every turn #}
            {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
        {%- endif %}"""


EFFORT = """    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"""
EFFORT_NEW = """    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {#- Claude Code / OpenAI effort names (high, max) mean this model's highest level instead of an error #}
    {%- if resolved_reasoning_effort in ('high', 'max') %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"""


def render(tok, body, template, merge):
    req = S.to_chat_completion_request(AnthropicMessagesRequest(**body), merge_inline_system=merge)
    msgs = []
    for m in req.messages:
        m = dict(m)
        for tc in m.get("tool_calls") or []:          # as vLLM does before rendering: arguments as a dict
            f = tc.get("function") or {}
            if isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"])
                except Exception:   # noqa: BLE001
                    pass
        msgs.append(m)
    tools = [t.model_dump() if hasattr(t, "model_dump") else t for t in (req.tools or [])] or None
    kw = dict(req.chat_template_kwargs or {})
    text = tok.apply_chat_template(msgs, tools=tools, chat_template=template, tokenize=False,
                                   add_generation_prompt=True, **kw)
    return text, tok.encode(text, add_special_tokens=False), sum(1 for m in msgs if m["role"] == "system")


def compare(name, a, b):
    ta, tb = a[1], b[1]
    n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    print(f"{name}: turn 1 {len(ta)} tokens ({a[2]} system msgs), turn 2 {len(tb)} tokens; first difference at token "
          f"{n} = {100 * n / len(ta):.1f}% of turn 1 reusable from the prefix cache")
    return n / len(ta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("turn1")
    ap.add_argument("turn2")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    orig = open(f"{args.model_dir}/chat_template.jinja").read()
    assert orig.count(ORIG) == 1, "system-message branch not found"
    assert orig.count(EFFORT) == 1, "reasoning-effort check not found"
    patched = orig.replace(ORIG, NEW).replace(EFFORT, EFFORT_NEW)
    msgs = [{"role": "user", "content": "hi"}]
    eff = {e: tok.apply_chat_template(msgs, chat_template=patched, tokenize=False, add_generation_prompt=True,
                                      reasoning_effort=e) for e in ("xhigh", "high", "max", "medium", "low")}
    orig_eff = {e: tok.apply_chat_template(msgs, chat_template=orig, tokenize=False, add_generation_prompt=True,
                                           reasoning_effort=e) for e in ("xhigh", "medium", "low")}
    print("effort high / max render as xhigh:", eff["high"] == eff["xhigh"] == eff["max"],
          "| xhigh / medium / low unchanged:", all(eff[e] == orig_eff[e] for e in orig_eff))
    assert not S._detect_merge_inline_system(patched), "vLLM would still merge inline system messages"
    b1, b2 = (json.load(open(p))["body"] for p in (args.turn1, args.turn2))
    old = compare("original template (vLLM merges reminders)", render(tok, b1, orig, True), render(tok, b2, orig, True))
    new = compare("patched template (reminders in place)    ", render(tok, b1, patched, False),
                  render(tok, b2, patched, False))
    # conversations without inline system messages: identical rendering
    plain = {"model": "q", "max_tokens": 10, "system": "You are helpful.",
             "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                          {"role": "user", "content": "write code"}]}
    same = render(tok, plain, orig, True)[0] == render(tok, plain, patched, False)[0]
    print(f"plain conversation renders identically: {same}")
    open(args.out, "w").write(patched)
    print(f"patched template written to {args.out}; reusable prefix {100 * old:.1f}% -> {100 * new:.1f}%")


if __name__ == "__main__":
    main()
