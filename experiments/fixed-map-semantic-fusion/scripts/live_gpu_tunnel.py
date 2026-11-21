"""Localhost-only SSH tunnel; password is interactive and never saved."""
import getpass,paramiko,socketserver,select,hashlib,sys,webbrowser
class Forward(socketserver.BaseRequestHandler):
 def handle(self):
  channel=None
  try:
   channel=client.get_transport().open_channel('direct-tcpip',('127.0.0.1',18766),self.request.getpeername())
   while True:
    readable,_,_=select.select([self.request,channel],[],[],60)
    for s in readable:
     data=s.recv(65536)
     if not data:return
     (channel if s is self.request else self.request).sendall(data)
  finally:
   if channel:channel.close()
class Server(socketserver.ThreadingTCPServer):allow_reuse_address=True;daemon_threads=True
class PinnedHostKey(paramiko.MissingHostKeyPolicy):
 def missing_host_key(self,client,hostname,key):
  if hashlib.sha256(key.asbytes()).hexdigest()!='962677eaf342b0d70d757796b387fe83964884f33f66284ff37e2fc6cf1496a7':raise paramiko.SSHException('Server host key differs from verified project endpoint')
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(PinnedHostKey())
password=getpass.getpass('SSH password: ')
client.connect('SERVER_HOST',port=16680,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20);del password
print('READY http://127.0.0.1:8766/',flush=True)
if '--open' in sys.argv:webbrowser.open('http://127.0.0.1:8766/')
try:Server(('127.0.0.1',8766),Forward).serve_forever()
finally:client.close()
