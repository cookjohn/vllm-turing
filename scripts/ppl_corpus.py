#!/usr/bin/env python3
"""Build three perplexity corpora on the server: Chinese text (markdown files, PPL_ZH_GLOB), C++ code, English docs."""
import glob, os, random, re

out = "/tmp/ppl"
os.makedirs(out, exist_ok=True)
random.seed(1)


def collect(paths, limit_chars, clean=None):
    buf, n = [], 0
    for p in paths:
        try:
            t = open(p, errors="ignore").read()
        except OSError:
            continue
        if clean:
            t = clean(t)
        if len(t) < 2000:
            continue
        buf.append(t)
        n += len(t)
        if n >= limit_chars:
            break
    return "\n\n".join(buf)[:limit_chars]


def clean_md(t):
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)          # images
    t = re.sub(r"<table>.*?</table>", "", t, flags=re.S)  # html tables
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


zh = glob.glob(os.environ.get("PPL_ZH_GLOB", os.path.expanduser("~/corpus/zh/**/*.md")), recursive=True)
random.shuffle(zh)
open(f"{out}/zh.txt", "w").write(collect(zh, 120000, clean_md))
code = sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/*.cpp"))) + sorted(glob.glob(os.path.expanduser("~/llama.cpp/common/*.cpp")))
open(f"{out}/code.txt", "w").write(collect(code, 330000))
en = sorted(glob.glob(os.path.expanduser("~/llama.cpp/docs/**/*.md"), recursive=True)) + sorted(glob.glob(os.path.expanduser("~/llama.cpp/tools/*/README.md")))
open(f"{out}/en.txt", "w").write(collect(en, 330000))
for k in ("zh", "code", "en"):
    print(k, os.path.getsize(f"{out}/{k}.txt"), "bytes")
