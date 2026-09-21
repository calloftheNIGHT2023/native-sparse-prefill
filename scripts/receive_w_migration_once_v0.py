"""One-shot remote evidence pull; ephemeral signing key lives only in RAM."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'site'))
import paramiko

def utc(): return datetime.now(timezone.utc).isoformat()
state = {'status': 'starting', 'started_utc': utc(), 'bytes_received': 0,
         'private_key_persisted': False, 'private_key_exported': False,
         'model_calls': 0, 'gpu_calls': 0}
def save():
    p = BASE / 'receiver-status.tmp'
    p.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(p, BASE / 'receiver-status.json')

def main():
    key = paramiko.RSAKey.generate(3072)
    (BASE / 'ephemeral-public-key.txt').write_text(key.get_name()+' '+key.get_base64()+' babylm-w-readonly-once\n')
    state['status'] = 'waiting_for_public_transfer_plan'; save()
    start = time.monotonic(); client = None
    try:
        while not (BASE / 'plan.json').exists():
            if time.monotonic()-start > 480: raise TimeoutError('Transfer plan not supplied within bound')
            time.sleep(1)
        plan = json.loads((BASE / 'plan.json').read_text())
        target = Path(plan['destination']).resolve()
        assert target.is_relative_to(Path('/workspace/native-sparse-prefill')) and not target.exists()
        class PinnedHost(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, offered):
                assert offered.get_name() == plan['host_key_type'] and offered.get_base64() == plan['host_key_base64'], 'Source host key differs'
        client = paramiko.SSHClient(); client.set_missing_host_key_policy(PinnedHost())
        client.connect(plan['host'], port=plan['port'], username='root', pkey=key,
                       allow_agent=False, look_for_keys=False, timeout=20, auth_timeout=20, banner_timeout=20)
        transport = client.get_transport()
        channel = transport.open_session(window_size=16*1024*1024, max_packet_size=32768, timeout=20)
        channel.settimeout(120); channel.exec_command('fetch-readonly')
        state.update(status='receiving_pinned_archive', archive_sha256=plan['sha256'], expected_bytes=plan['bytes']); save()
        h = hashlib.sha256(); target.parent.mkdir(parents=True, exist_ok=True); previous = 0
        with target.open('xb') as f:
            while True:
                if time.monotonic()-start > 1200: raise TimeoutError('Bounded transfer deadline')
                chunk = channel.recv(1024*1024)
                if not chunk: break
                f.write(chunk); h.update(chunk); state['bytes_received'] += len(chunk)
                assert state['bytes_received'] <= plan['bytes'], 'Archive exceeds approved size'
                if state['bytes_received']-previous >= 32*1024*1024:
                    previous = state['bytes_received']; save()
            f.flush(); os.fsync(f.fileno())
        assert channel.recv_exit_status() == 0, 'Source transfer command failed'
        assert state['bytes_received'] == plan['bytes'] and h.hexdigest() == plan['sha256'], 'Archive bytes/SHA differ'
        state.update(status='archive_received_sha_verified', destination=str(target), actual_sha256=h.hexdigest())
    except BaseException as e:
        state.update(status='failed_evidence_preserved', error_type=type(e).__name__, error=str(e))
    finally:
        if client: client.close()
        del key
        state.update(finished_utc=utc(), ephemeral_private_key_lifetime_ended=True); save()
    return 0 if state['status']=='archive_received_sha_verified' else 2

if __name__ == '__main__': raise SystemExit(main())
