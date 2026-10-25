"""A build log without a newline must not defeat the command deadline."""
import sys
import time

import pytest

from llama_broker.runtime_control import RuntimeControlError, _run


def test_partial_log_line_cannot_block_timeout():
    started = time.monotonic()
    with pytest.raises(RuntimeControlError, match="timed out"):
        _run([sys.executable, "-c", "import sys,time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(2)"], timeout=0.1)
    assert time.monotonic() - started < 1


def test_command_output_preserves_unicode_and_final_partial_line():
    logged = []
    output = _run([sys.executable, "-c", "import sys; sys.stdout.write('first\\nfinal \\u2603')"], log=logged.append)
    assert output == "first\nfinal \u2603"
    assert logged == ["first", "final \u2603"]
