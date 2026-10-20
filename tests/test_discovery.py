from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llama_broker import discovery


class DiscoveryContextTests(unittest.TestCase):
    def test_model_slot_context_takes_priority_over_active_backend_probes(self) -> None:
        models = [
            {"id": "text", "context_length": 131072, "parallel": 2,
             "slot_context_window": 131072},
            {"id": "vision", "context_length": 16384, "parallel": 2,
             "slot_context_window": 16384},
        ]
        for slots in ([], [{"n_ctx": 131072}]):
            with self.subTest(slots=slots):
                responses = [
                    Mock(status_code=200, json=Mock(return_value=body))
                    for body in ({"data": models}, {"total_slots": 2}, slots)
                ]
                client = SimpleNamespace(get=Mock(side_effect=responses))
                with patch.object(discovery, "requests", client):
                    endpoint = discovery._probe_endpoint("http://127.0.0.1:11434/v1")
                self.assertIsNotNone(endpoint)
                self.assertEqual(
                    [model.slot_context_window for model in endpoint.models],
                    [131072, 16384],
                )
