"""Supervise a native Basalt child and its authenticated TLS bridge."""
from __future__ import annotations
import argparse, os, signal, subprocess, sys, time

def main(argv=None) -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--basalt-binary', required=True); p.add_argument('--config', required=True)
    p.add_argument('--cert', required=True); p.add_argument('--key', required=True)
    p.add_argument('--auth-token-file', required=True); p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True); p.add_argument('--bridge-port', type=int, required=True)
    a=p.parse_args(argv)
    child=subprocess.Popen([a.basalt_binary,'serve','--config',a.config,'--host','127.0.0.1','--port',str(a.port)], start_new_session=True)
    bridge=subprocess.Popen([sys.executable,'-m','llama_broker.basalt_bridge','--cert',a.cert,'--key',a.key,'--auth-token-file',a.auth_token_file,'--upstream-url',f'http://127.0.0.1:{a.port}','--host',a.host,'--port',str(a.bridge_port)], start_new_session=True)
    stopping=False
    def stop(signum, _frame):
        nonlocal stopping
        if stopping: return
        stopping=True
        for proc in (bridge, child):
            if proc.poll() is None:
                try: os.killpg(proc.pid, signum)
                except ProcessLookupError: pass
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        while True:
            if child.poll() is not None or bridge.poll() is not None:
                stop(signal.SIGTERM); return child.returncode or bridge.returncode or 1
            time.sleep(.05)
    finally:
        stop(signal.SIGTERM)
        for proc in (bridge, child):
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
    return 0

if __name__ == '__main__': raise SystemExit(main())
