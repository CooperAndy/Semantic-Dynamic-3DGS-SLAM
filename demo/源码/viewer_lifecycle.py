"""Managed Viewer leases: tabs, refresh grace, and abandoned-browser cleanup."""
import threading
import time


class ViewerLifecycle:
    def __init__(self, *, close_delay=10.0, lease_timeout=180.0, startup_timeout=120.0):
        self.close_delay = close_delay
        self.lease_timeout = lease_timeout
        self.startup_timeout = startup_timeout
        self.clients = {}
        self.updates = {}
        self.lock = threading.Lock()
        self.pending_launch_until = time.monotonic() + startup_timeout
        self.empty_since = None
        self.active_operations = 0
        self.stopping = False

    def launch(self):
        with self.lock:
            self.pending_launch_until = time.monotonic() + self.startup_timeout

    def presence(self, client, alive, sequence):
        with self.lock:
            now = time.monotonic()
            # A late in-flight heartbeat must not resurrect a closed page.
            if sequence <= self.updates.get(client, (-1, 0))[0]:
                return
            self.updates[client] = (sequence, now)
            if alive:
                self.clients[client] = now
                self.empty_since = None
                self.pending_launch_until = 0
            else:
                self.clients.pop(client, None)
                if not self.clients and self.empty_since is None:
                    self.empty_since = now

    def operation(self, delta):
        with self.lock:
            self.active_operations += delta

    def should_stop(self):
        with self.lock:
            now = time.monotonic()
            for client, seen in list(self.clients.items()):
                if now - seen > self.lease_timeout:
                    del self.clients[client]
            for client, (_, seen) in list(self.updates.items()):
                if client not in self.clients and now - seen > 2*self.lease_timeout:
                    del self.updates[client]
            if self.clients or now < self.pending_launch_until or self.active_operations:
                self.empty_since = None
                return False
            if self.empty_since is None:
                self.empty_since = now
            if now - self.empty_since >= self.close_delay:
                self.stopping = True
                return True
            return False

    def status(self):
        with self.lock:
            return dict(managed=True, active_viewers=len(self.clients), stopping=self.stopping)

    def watch(self, server, finished):
        while not finished.wait(0.25):
            if self.should_stop():
                print('All T07 Viewer pages closed; stopping local service.', flush=True)
                server.shutdown()
                return
