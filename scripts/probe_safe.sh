#!/bin/bash
# Runs a test that needs the GPUs (default: vllm_util_probe.sh with the caller's env) with production stopped, under a
# hard time limit, then always restores production: a hung test (10-22: a bare `wait` in the probe kept production
# down 6 hours) is killed, and any server left on the test port is killed by its process group (looked up from the
# listening socket, never by a command-line pattern: pkill -f matches the caller's own shell).
# usage: PW=<sudo password> probe_safe.sh [limit seconds, default 2400] [command ...]   (output: the command's stdout)
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
LIMIT=${1:-2400}
shift
[ $# -gt 0 ] || set -- $TOOLS/vllm_util_probe.sh
echo "$PW" | sudo -S -p "" systemctl stop vllm-server
sleep 5
timeout --signal=TERM --kill-after=30 "$LIMIT" "$@"
rc=$?
[ $rc -eq 124 ] && echo "probe_safe: test killed after $LIMIT s"
# a leftover test server on :8081 (setsid: its PID is the process group)
for pid in $(ss -ltnpH 'sport = :8081' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
    echo "probe_safe: killing leftover test server (pid $pid, group $pg)"
    kill -INT -"$pg" 2>/dev/null
    sleep 20
    kill -9 -"$pg" 2>/dev/null
done
# production starts with a clean environment: the test's variables (LEN, TAG, EXTRA_ENV, ...) must not reach
# prod_start.sh (10-23: a probe's LEN=131072 restarted production with half its context window)
env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin PW="$PW" KEEPMON=1 $TOOLS/prod_start.sh 2>&1 | tail -1
exit $rc
