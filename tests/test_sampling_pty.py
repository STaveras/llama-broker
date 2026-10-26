"""Exercise the real curses editor on a disposable pseudo-terminal."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pty
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unittest
from pathlib import Path

from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.broker_tui import TUIModel
from llama_broker.sampling import qwen27b_sampling_defaults
from tests.test_manifest_watcher import make_profile_record


class EditorSession:
    def __init__(self, path, env):
        self.fd, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        code = "import sys;from llama_broker.broker_tui import run_tui;sys.exit(run_tui(sys.argv[1], 'main'))"
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-c", code, str(path)], env=env,
                stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
            )
        finally:
            os.close(slave)
        self.buffer = b""
        self.cursor = 0
        self.status = None

    def send(self, data):
        os.write(self.fd, data)

    def read(self, timeout=0.1):
        if select.select([self.fd], [], [], timeout)[0]:
            try:
                self.buffer += os.read(self.fd, 65536)
            except OSError:
                pass

    def expect(self, text, timeout=8):
        wanted = text.encode()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            index = self.buffer.find(wanted, self.cursor)
            if index >= 0:
                self.cursor = index + len(wanted)
                return
            self.read()
        raise AssertionError(f"Editor never displayed {text!r}: {self.buffer[-5000:]!r}")

    def finish(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self.status = status
                if self.status != 0:
                    raise AssertionError(f"Editor exit {self.status}: {self.buffer[-2000:]!r}")
                return
            self.read()
        raise AssertionError("Editor failed to exit")

    def close(self):
        if self.process.poll() is None:
            # Only the disposable editor child created above, never a service.
            self.process.terminate()
            self.process.wait(timeout=5)
        os.close(self.fd)


@unittest.skipUnless(os.name == "posix", "curses editor requires a POSIX terminal")
class SamplingEditorPTYTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        binary = root / "llama-server"
        binary.write_bytes(b"offline fixture, not a runnable model server")
        binary.chmod(0o755)
        model = root / "model.gguf"
        model.write_bytes(b"offline model fixture")
        record = make_profile_record("main", "main", model)
        record["server_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        record["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        self.manifest = {"schema_version": 1, "profiles": [record]}
        self.path = root / "profiles.json"
        self.path.write_text(json.dumps(self.manifest))
        self.original = self.path.read_bytes()
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("LLAMA_BROKER_", "BROKER_"))}
        self.env.update(TERM="xterm-256color", HOME=str(root / "home"),
                        XDG_CONFIG_HOME=str(root / "config"), XDG_STATE_HOME=str(root / "state"),
                        XDG_DATA_HOME=str(root / "data"), XDG_RUNTIME_DIR=str(root / "run"),
                        DBUS_SESSION_BUS_ADDRESS="unix:path=" + str(root / "no-session"),
                        DBUS_SYSTEM_BUS_ADDRESS="unix:path=" + str(root / "no-system"))

    def test_piped_profile_editor_exits_cleanly_without_terminal_escapes(self):
        result = subprocess.run(
            [sys.executable, "brokerctl.py", "--profiles", str(self.path), "profile", "edit", "main"],
            env=self.env, input="", capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires an interactive terminal", result.stderr)
        self.assertNotIn("\x1b", result.stdout + result.stderr)
        self.assertEqual(self.path.read_bytes(), self.original)

    def select_preset(self):
        session = EditorSession(self.path, self.env)
        self.addCleanup(session.close)
        session.expect("Press s to save")
        model = TUIModel(self.manifest)
        index = next(i for i, f in enumerate(model.visible_fields) if f.key == "sampling_preset")
        session.send(b"j" * index + b"\n")
        session.expect("choose a value")
        # The current Runtime defaults value is first; k wraps to Qwen 27B.
        session.send(b"k\n")
        session.expect("Field updated")
        return session

    def test_real_terminal_preset_save_round_trips_both_modes(self):
        session = self.select_preset()
        session.send(b"s")
        session.expect("Save profile config?")
        session.send(b"y")
        session.expect("Saved profile configuration.")
        session.send(b"q")
        session.finish()
        saved = json.loads(self.path.read_text())["profiles"][0]["sampling_defaults"]
        self.assertEqual(saved, qwen27b_sampling_defaults())
        parsed = ProfileRegistry.from_path(self.path).get("main")
        self.assertEqual(dict(parsed.sampling_defaults["instruct"]), saved["instruct"])
        self.assertNotIn(b"Traceback", session.buffer)

    def test_real_terminal_discard_does_not_write_a_draft(self):
        session = self.select_preset()
        session.send(b"q")
        session.expect("Discard unsaved changes?")
        session.send(b"y")
        session.finish()
        self.assertEqual(self.path.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
