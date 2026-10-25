import http.client,json,os,pathlib,ssl,subprocess,tempfile,threading,time,unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from llama_broker.basalt_bridge import BridgeHandler,BridgeServer,_loopback

class Upstream(BaseHTTPRequestHandler):
    seen=[]
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=='/events':
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            self.wfile.write(b'data: first\n\n'); self.wfile.flush(); time.sleep(1)
            self.wfile.write(b'data: last\n\n'); self.wfile.flush(); return
        self.seen.append(dict(self.headers)); body=b'{"ok":true}'
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    do_POST=do_GET

class BasaltBridgeNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')); p=pathlib.Path(cls.tmp.name)
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost','-addext','subjectAltName=IP:127.0.0.1','-keyout',str(p/'key.pem'),'-out',str(p/'cert.pem')],capture_output=True,check=True)
        cls.up=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        class Handler(BridgeHandler): pass
        Handler.upstream_host='127.0.0.1'; Handler.upstream_port=cls.up.server_port; Handler.bridge_token='test-token'; Handler.max_body_bytes=100
        Handler.upstream_timeout=900
        cls.bridge=BridgeServer(('127.0.0.1',0),Handler)
        server_context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); server_context.load_cert_chain(p/'cert.pem',p/'key.pem')
        cls.bridge.ssl_context=server_context
        cls.client_context=ssl.create_default_context(cafile=str(p/'cert.pem'))
        for s in (cls.up,cls.bridge): threading.Thread(target=s.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls):
        for s in (cls.bridge,cls.up): s.shutdown(); s.server_close()
        cls.tmp.cleanup()
    def connect(self): return http.client.HTTPSConnection('127.0.0.1',self.bridge.server_port,context=self.client_context,timeout=3)
    def test_tls_json_framing_and_credentials(self):
        c=self.connect(); c.request('GET','/health',headers={'X-Basalt-Bridge-Token':'test-token','Authorization':'Bearer provider-secret','X-Api-Key':'provider-secret','Connection':'x-secret','X-Secret':'private'})
        r=c.getresponse(); self.assertEqual(r.status,200); self.assertEqual(json.loads(r.read()),{'ok':True}); c.close()
        h={k.lower():v for k,v in Upstream.seen[-1].items()}
        for key in ('authorization','x-api-key','x-basalt-bridge-token','x-secret'): self.assertNotIn(key,h)
    def test_invalid_token(self):
        n=len(Upstream.seen); c=self.connect(); c.request('GET','/health',headers={'X-Basalt-Bridge-Token':'bad'}); r=c.getresponse(); self.assertEqual(r.status,401); r.read(); c.close(); self.assertEqual(len(Upstream.seen),n)
    def test_body_limit(self):
        c=self.connect(); c.request('POST','/x',body=b'x'*101,headers={'X-Basalt-Bridge-Token':'test-token'}); r=c.getresponse(); self.assertEqual(r.status,413); r.read(); c.close()
    def test_transfer_encoding_rejected(self):
        c=self.connect(); c.putrequest('POST','/x'); c.putheader('X-Basalt-Bridge-Token','test-token'); c.putheader('Transfer-Encoding','chunked'); c.endheaders(); c.send(b'0\r\n\r\n'); r=c.getresponse(); self.assertEqual(r.status,400); r.read(); c.close()
    def test_stream_flushes_before_upstream_finishes(self):
        c=self.connect(); start=time.monotonic(); c.request('GET','/events',headers={'X-Basalt-Bridge-Token':'test-token'}); r=c.getresponse(); self.assertEqual(r.readline(),b'data: first\n'); self.assertLess(time.monotonic()-start,.8); r.read(); c.close()
    def test_upstream_loopback_only(self):
        for url in ('http://example.com:80','https://127.0.0.1:80','http://user:password@127.0.0.1:80'):
            with self.assertRaises(ValueError): _loopback(url)

    def test_long_deadline_is_configurable_and_defaults_to_900(self):
        self.assertEqual(BridgeHandler.upstream_timeout, 900)
        self.assertEqual(BridgeHandler.validate_upstream_timeout(300), 300)
        with self.assertRaises(ValueError): BridgeHandler.validate_upstream_timeout(0)
        with self.assertRaises(ValueError): BridgeHandler.validate_upstream_timeout(901)

    def test_invalid_json_schema_is_rejected_before_upstream(self):
        n = len(Upstream.seen)
        c = self.connect()
        body = json.dumps({'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'type': 'not-a-type'}}}}).encode()
        c.request('POST', '/chat', body=body, headers={'X-Basalt-Bridge-Token': 'test-token', 'Content-Type': 'application/json'})
        r = c.getresponse()
        self.assertEqual(r.status, 400)
        self.assertIn(b'invalid response schema', r.read())
        c.close()
        self.assertEqual(len(Upstream.seen), n)
