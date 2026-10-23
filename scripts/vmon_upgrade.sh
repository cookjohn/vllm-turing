#!/bin/bash
# Zero-downtime upgrade of the monitoring proxy (unit vllm-monitor on :8080 -> vLLM on 127.0.0.1:8090) to the code now
# in ~/monitor: a second instance (unit vllm-monitor-next, :8089) takes new external connections through an iptables
# redirect while the old one finishes its requests; vllm-monitor then restarts with the new code and the redirect is
# removed. An instance is stopped only when it has no request in flight. Local connections (127.0.0.1) are never
# redirected.
# usage: PW=<sudo password> vmon_upgrade.sh
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
K=$(cat $HOME/models/api-key.txt)
sudo_() { echo "$PW" | sudo -S -p "" "$@"; }
active() {  # requests in flight on the instance at port $1
    curl -s -m 5 -H "Authorization: Bearer $K" localhost:$1/monitor/api/list | $V/python -c \
        "import json,sys; print(sum(1 for r in json.load(sys.stdin)['recs'] if not r.get('t_end')))" 2>/dev/null || echo 0
}
drain() {
    for i in $(seq 1 900); do [ "$(active $1)" = 0 ] && { echo "  :$1 drained after ${i} s"; return 0; }; sleep 1; done
    echo "  :$1 still busy after 900 s, stopping anyway"
}
start_unit() {  # $1 = unit, $2 = port
    sudo_ systemctl reset-failed $1 2>/dev/null
    sudo_ systemd-run --unit=$1 -p Restart=always -p RestartSec=2 --uid=$(id -un) --gid=$(id -gn) \
        -p WorkingDirectory=$HOME --setenv=HOME=$HOME \
        $V/python $REPO/monitor/vllm_monitor.py --listen 0.0.0.0:$2 --upstream http://127.0.0.1:8090 \
        --log-dir $HOME/monitor/logs > /dev/null
    for i in $(seq 1 30); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$2/health)" = 200 ] && return 0
        sleep 1
    done
    echo "$1 did not come up on :$2"
    return 1
}
start_unit vllm-monitor-next 8089 || exit 1
sudo_ iptables -t nat -I PREROUTING -p tcp --dport 8080 -j REDIRECT --to-ports 8089
echo "new connections -> :8089 (new code), draining :8080"
drain 8080
sudo_ systemctl stop vllm-monitor
start_unit vllm-monitor 8080 || { echo "vllm-monitor failed: the redirect to :8089 stays in place"; exit 1; }
sudo_ iptables -t nat -D PREROUTING -p tcp --dport 8080 -j REDIRECT --to-ports 8089
echo "new connections -> :8080 (new code), draining :8089"
drain 8089
sudo_ systemctl stop vllm-monitor-next
sudo_ systemctl reset-failed vllm-monitor-next 2>/dev/null
echo "monitor upgraded: vllm-monitor $(systemctl is-active vllm-monitor) on :8080," \
     "redirect rules left: $(sudo_ iptables -t nat -S PREROUTING | grep -c REDIRECT)"
