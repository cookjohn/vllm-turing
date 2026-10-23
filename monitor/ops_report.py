#!/usr/bin/env python3
"""Operations report for a time window of the monitoring proxy's request logs (vllm_monitor.py): clients and outcomes,
concurrency over time (requests in flight / generating), first-token waits, and per-request generation speed by context
length and by the number of streams generating alongside, plus output throughput per 10 minutes.
usage: ops_report.py <requests-*.jsonl.gz> ... [--since HH:MM] [--until HH:MM]   (UTC, on the date of the first record;
       python3; a file still being appended is read up to its end)"""
import collections
import gzip
import json
import sys
import time
import zlib

args = sys.argv[1:]
since = until = None
paths = []
i = 0
while i < len(args):
    if args[i] in ("--since", "--until"):
        if args[i] == "--since":
            since = args[i + 1]
        else:
            until = args[i + 1]
        i += 2
    else:
        paths.append(args[i])
        i += 1


def records(ps):
    for p in ps:
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        except (EOFError, zlib.error, OSError):
            continue


rows = [r for r in records(paths) if r.get("t0")]
if not rows:
    sys.exit("no records")
day0 = time.gmtime(rows[0]["t0"])
base = time.mktime(time.struct_time((day0.tm_year, day0.tm_mon, day0.tm_mday, 0, 0, 0, 0, 0, 0))) - time.timezone


def at(hhmm):
    h, m = hhmm.split(":")
    return base + int(h) * 3600 + int(m) * 60


lo = at(since) if since else -1e18
hi = at(until) if until else 1e18
rows = [r for r in rows if lo <= r["t0"] < hi]
if not rows:
    sys.exit("no records in the window")
hm = lambda t: time.strftime("%H:%M:%S", time.gmtime(t))   # noqa: E731
t_lo, t_hi = min(r["t0"] for r in rows), max((r.get("t_end") or r["t0"]) for r in rows)
print(f"window {hm(t_lo)} - {hm(t_hi)} UTC ({(t_hi - t_lo) / 60:.0f} min), {len(rows)} requests")
cl = collections.Counter((r.get("client"), (r.get("ua") or "")[:40]) for r in rows)
print("clients:", "; ".join(f"{c} {u} x{n}" for (c, u), n in cl.most_common(5)))
st = collections.Counter((r.get("status"), r.get("finish"), (str(r.get("err"))[:60] if r.get("err") else None))
                         for r in rows)
print("outcomes:", "; ".join(f"{s}/{f}{' ' + e if e else ''} x{n}" for (s, f, e), n in st.most_common(8)))

ok = [r for r in rows if r.get("status") == 200 and r.get("t_first") and r.get("t_end")]

# concurrency: time-weighted share of the busy time with n requests in flight / n generating
def occupancy(iv):
    ev = sorted([(a, 1) for a, b in iv] + [(b, -1) for a, b in iv])
    occ = collections.Counter()
    n, last = 0, None
    for t, d in ev:
        if last is not None and n > 0:
            occ[n] += t - last
        n += d
        last = t
    return occ


fl = occupancy([(r["t0"], r["t_end"]) for r in ok])
gn = occupancy([(r["t_first"], r["t_end"]) for r in ok])
busy = sum(fl.values())
print(f"busy (>= 1 request in flight) {busy / 60:.1f} min of {(t_hi - t_lo) / 60:.0f}; in flight: " +
      ", ".join(f"{n}: {v / busy * 100:.0f}%" for n, v in sorted(fl.items())))
gb = sum(gn.values())
if gb:
    print(f"generating {gb / 60:.1f} min; streams generating: " +
          ", ".join(f"{n}: {v / gb * 100:.0f}%" for n, v in sorted(gn.items())))

# first-token wait vs uncached prompt
ttft = sorted(r["t_first"] - r["t0"] for r in ok)
n = len(ttft)
if n:
    print(f"first-token wait: median {ttft[n // 2]:.1f} s, p90 {ttft[int(n * .9)]:.1f} s, max {ttft[-1]:.0f} s; "
          f"total {sum(ttft) / 60:.1f} min")
    slow = sorted(ok, key=lambda r: r["t0"] - r["t_first"])[:6]
    for r in slow:
        unc = (r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0)
        others = sum(1 for o in ok if o is not r and o["t0"] < r["t_first"] and o["t_end"] > r["t0"])
        print(f"   {hm(r['t0'])} wait {r['t_first'] - r['t0']:6.1f} s: prompt {r.get('prompt_tokens')} "
              f"({unc} uncached), {others} other requests in flight meanwhile")


# generation speed per request: output tokens / (end - first token), by context and by streams alongside
def alongside(r):
    """time-weighted mean number of other requests generating during r's generation"""
    a, b = r["t_first"], r["t_end"]
    if b <= a:
        return 0.0
    s = 0.0
    for o in ok:
        if o is r:
            continue
        x, y = max(a, o["t_first"]), min(b, o["t_end"])
        if y > x:
            s += y - x
    return s / (b - a)


gen = [r for r in ok if (r.get("completion_tokens") or 0) >= 64 and r["t_end"] - r["t_first"] > 1]
tab = collections.defaultdict(lambda: [0, 0.0, 0])   # (ctx bucket, streams bucket) -> tokens, seconds, requests
for r in gen:
    p = r.get("prompt_tokens") or 0
    cb = "<16k" if p < 16384 else "16-64k" if p < 65536 else "64-128k" if p < 131072 else ">=128k"
    a = alongside(r)
    sb = "alone" if a < 0.25 else "1-2 others" if a < 2.5 else "3+ others"
    t = tab[(cb, sb)]
    t[0] += r["completion_tokens"]
    t[1] += r["t_end"] - r["t_first"]
    t[2] += 1
print("generation tok/s per request (requests with >= 64 output tokens), by context x streams alongside:")
for cb in ("<16k", "16-64k", "64-128k", ">=128k"):
    cells = []
    for sb in ("alone", "1-2 others", "3+ others"):
        t = tab.get((cb, sb))
        cells.append(f"{sb} {t[0] / t[1]:5.1f} ({t[2]})" if t else f"{sb}    -     ")
    print(f"   {cb:8s} " + " | ".join(cells))

# output throughput per 10 minutes (tokens attributed uniformly over each generation)
buck = collections.Counter()
for r in ok:
    a, b, o = r["t_first"], r["t_end"], r.get("completion_tokens") or 0
    if b <= a:
        continue
    t = a
    while t < b:
        e = min(b, (int((t - base) // 600) + 1) * 600 + base)
        buck[int((t - base) // 600)] += o * (e - t) / (b - a)
        t = e
print("output tok/s per 10 min:", " ".join(f"{hm(base + k * 600)[:5]} {v / 600:.0f}" for k, v in sorted(buck.items())))
