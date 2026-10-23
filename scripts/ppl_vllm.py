#!/usr/bin/env python3
"""Perplexity of a HF/vLLM model with llama-perplexity's exact scoring: the corpus is tokenized once, cut into
non-overlapping chunks of N_CTX tokens, and in each chunk the targets at positions N_CTX/2+1 .. N_CTX-1 are scored.
usage: ppl_vllm.py <model_dir> <corpus.txt>... [--chunks N]"""
import math, sys
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

N_CTX = 2048


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_chunks = int(sys.argv[sys.argv.index("--chunks") + 1]) if "--chunks" in sys.argv else 40
    model_dir, corpora = args[0], [a for a in args[1:] if not a.isdigit()]

    llm = LLM(model=model_dir, dtype="float16", max_model_len=N_CTX + 16, gpu_memory_utilization=0.85, max_num_seqs=1,
              max_num_batched_tokens=N_CTX + 16, enforce_eager=True, language_model_only=True, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)

    for path in corpora:
        ids = tok.encode(open(path).read(), add_special_tokens=False)
        n_chunks = min(max_chunks, len(ids) // N_CTX)
        prompts = [TokensPrompt(prompt_token_ids=ids[c*N_CTX:(c+1)*N_CTX]) for c in range(n_chunks)]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        nll, cnt = 0.0, 0
        for o, p in zip(outs, prompts):
            pl = o.prompt_logprobs
            t = p["prompt_token_ids"]
            for pos in range(N_CTX // 2 + 1, N_CTX):
                nll -= pl[pos][t[pos]].logprob
                cnt += 1
        print("%-10s tokens %7d chunks %3d  PPL %.4f" % (path.split("/")[-1], len(ids), n_chunks, math.exp(nll / cnt)), flush=True)


if __name__ == "__main__":
    main()
