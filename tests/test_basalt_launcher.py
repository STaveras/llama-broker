"""Actual TLS/process tests with an explicit CPU-only native-server fixture."""
import http.client,json,os,pathlib,socket,ssl,subprocess,sys,tempfile,time,unittest

FIXTURE='''#!/usr/bin/env python3
import sys,json,os,time,subprocess
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
args=sys.argv
config=json.load(open(args[args.index('--config')+1]))
if config.get('exit'): sys.exit(7)
port=int(args[args.index('--port')+1]); token=args[args.index('--api-key')+1]
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  ok=self.headers.get('Authorization')=='Bearer '+token
  body=json.dumps({'ok':ok}).encode(); self.send_response(200 if ok else 401); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
s=ThreadingHTTPServer(('127.0.0.1',port),H)
grandchild=None
if config.get('grandchild'):
 grandchild=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']).pid
open(config['marker'],'w').write(json.dumps({'pid':os.getpid(),'port':port,'grandchild':grandchild}))
s.serve_forever()
'''

class LauncherTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')); self.p=pathlib.Path(self.tmp.name); self.procs=[]
  subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost','-addext','subjectAltName=IP:127.0.0.1','-keyout',str(self.p/'key'),'-out',str(self.p/'cert')],capture_output=True,check=True)
  (self.p/'token').write_text('public-test-token'); (self.p/'native').write_text(FIXTURE); (self.p/'native').chmod(0o700)
  self.marker=self.p/'started'; (self.p/'config').write_text(json.dumps({'marker':str(self.marker)}))
  self.context=ssl.create_default_context(cafile=str(self.p/'cert'))
  with socket.socket() as s: s.bind(('127.0.0.1',0)); self.port=s.getsockname()[1]
 def tearDown(self):
  for proc in self.procs:
   if proc.poll() is None: proc.terminate()
   try: proc.wait(timeout=8)
   except subprocess.TimeoutExpired: proc.kill(); proc.wait()
  self.tmp.cleanup()
 def launch(self,cert=None):
  args=[sys.executable,'-m','llama_broker.basalt_launcher','--basalt-binary',str(self.p/'native'),'--config',str(self.p/'config'),'--cert',str(cert or self.p/'cert'),'--key',str(self.p/'key'),'--auth-token-file',str(self.p/'token'),'--port','0','--bridge-port',str(self.port)]
  env=os.environ.copy(); env['PYTHONPATH']=str(pathlib.Path(__file__).resolve().parents[1]/'src')
  proc=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env); self.procs.append(proc); return proc
 def test_real_tls_auth_private_listener_and_cleanup(self):
  proc=self.launch(); ready=False
  for _ in range(100):
   if proc.poll() is not None: self.fail('Launcher exited before TLS readiness')
   try:
    c=http.client.HTTPSConnection('127.0.0.1',self.port,context=self.context,timeout=.2); c.request('GET','/health',headers={'Authorization':'Bearer public-test-token'}); r=c.getresponse(); data=r.read(); c.close()
    if r.status==200: ready=True; break
   except OSError: pass
   time.sleep(.05)
  self.assertTrue(ready); self.assertEqual(json.loads(data),{'ok':True})
  if 'second-key' in (self.p/'token').read_text():
   c=http.client.HTTPSConnection('127.0.0.1',self.port,context=self.context,timeout=1); c.request('GET','/health',headers={'Authorization':'Bearer second-key'}); r=c.getresponse(); self.assertEqual(r.status,200); r.read(); c.close()
  info=json.loads(self.marker.read_text()); c=http.client.HTTPConnection('127.0.0.1',info['port'],timeout=1); c.request('GET','/health',headers={'Authorization':'Bearer public-test-token'}); r=c.getresponse(); self.assertEqual(r.status,401); r.read(); c.close()
  c=http.client.HTTPSConnection('127.0.0.1',self.port,context=self.context,timeout=1); c.request('GET','/health'); r=c.getresponse(); self.assertEqual(r.status,401); r.read(); c.close()
  proc.terminate(); self.assertEqual(proc.wait(timeout=8),0)
  with self.assertRaises(ProcessLookupError): os.kill(info['pid'],0)
  with self.assertRaises(OSError): socket.create_connection(('127.0.0.1',self.port),timeout=.2)
 def test_multiple_backend_keys(self):
  (self.p/'token').write_text('# rotation keys\npublic-test-token\n\nsecond-key\n')
  self.test_real_tls_auth_private_listener_and_cleanup()
 def test_invalid_tls_never_spawns_native_child(self):
  proc=self.launch(self.p/'missing-cert'); self.assertNotEqual(proc.wait(timeout=5),0); self.assertFalse(self.marker.exists())
 def test_early_native_exit_is_detected_and_bounded(self):
  (self.p/'config').write_text(json.dumps({'exit':True})); start=time.monotonic(); proc=self.launch(); self.assertNotEqual(proc.wait(timeout=5),0); self.assertLess(time.monotonic()-start,5)
 def test_shutdown_kills_descendants_even_after_group_leader_exits(self):
  (self.p/'config').write_text(json.dumps({'marker':str(self.marker),'grandchild':True}))
  proc=self.launch()
  for _ in range(100):
   if self.marker.exists(): break
   time.sleep(.05)
  info=json.loads(self.marker.read_text()); descendant=info['grandchild']
  time.sleep(.2)
  try:
   proc.terminate(); proc.wait(timeout=8)
   status=pathlib.Path(f'/proc/{descendant}/stat')
   self.assertTrue(not status.exists() or status.read_text().split()[2]=='Z', 'native descendant survived launcher shutdown')
  finally:
   try: os.kill(descendant,9)
   except ProcessLookupError: pass
