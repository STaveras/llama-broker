from __future__ import annotations

import unittest

from llama_broker import decisions
from llama_broker.decisions import (
    DecisionBackendError,
    DecisionRequestError,
    parse_input,
    parse_questions,
    systemone_request,
    translate_answer,
)
from tests import test_system1 as _s1


ROUTE = {
    "type": "choice",
    "name": "team",
    "instructions": "Which team should handle this?",
    "choices": [
        {"value": "billing", "description": "Charges and refunds"},
        {"value": "technical", "description": "Bugs"},
    ],
}
SEVERITY = {
    "type": "score",
    "name": "severity",
    "instructions": "How severe is the issue?",
    "levels": [
        {"label": "Cosmetic", "description": "Appearance only."},
        {"label": "Workaround available", "description": "Another way works."},
        {"label": "Fully blocked", "description": "No workaround."},
    ],
}
REFUND = {"type": "predicate", "name": "refund", "instructions": "Is it a refund request?"}


def _no_fetch(url: str) -> str:
    raise AssertionError(f"unexpected fetch of {url}")


class RequestTranslationTests(unittest.TestCase):
    def test_questions_map_to_d1_types_and_criteria(self) -> None:
        questions = parse_questions([REFUND, ROUTE, SEVERITY])
        request = systemone_request("charged twice", [], questions)
        self.assertEqual(
            request,
            {
                "state": "charged twice",
                "questions": {
                    "refund": {"type": "noul", "instructions": "Is it a refund request?"},
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Charges and refunds", "technical": "Bugs"},
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "How severe is the issue?",
                        "criteria": [
                            "Cosmetic: Appearance only.",
                            "Workaround available: Another way works.",
                            "Fully blocked: No workaround.",
                        ],
                    },
                },
            },
        )

    def test_invalid_questions_name_the_bad_param(self) -> None:
        cases = [
            ([], "questions"),
            ([{**REFUND, "type": "bool"}], "questions[0].type"),
            ([REFUND, REFUND], "questions[1].name"),
            ([{**REFUND, "instructions": ""}], "questions[0].instructions"),
            ([{**ROUTE, "choices": ROUTE["choices"][:1]}], "questions[0].choices"),
            (
                [{**ROUTE, "choices": [ROUTE["choices"][0], ROUTE["choices"][0]]}],
                "questions[0].choices[1].value",
            ),
            ([{**SEVERITY, "levels": None}], "questions[0].levels"),
        ]
        for raw, param in cases:
            with self.subTest(param=param), self.assertRaises(DecisionRequestError) as caught:
                parse_questions(raw)
            self.assertEqual(caught.exception.param, param)

    def test_message_input_becomes_state_and_images(self) -> None:
        fetched: list[str] = []

        def fetch(url: str) -> str:
            fetched.append(url)
            return "data:image/png;base64,QQ=="

        state, images = parse_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is the box damaged?"},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="},
                        {"type": "input_image", "image_url": "https://example.com/box.png"},
                    ],
                },
                {"role": "user", "content": "It arrived today."},
            ],
            fetch_image=fetch,
        )
        self.assertEqual(state, "Is the box damaged?\n\nIt arrived today.")
        self.assertEqual(images, ["data:image/jpeg;base64,AA==", "data:image/png;base64,QQ=="])
        self.assertEqual(fetched, ["https://example.com/box.png"])

    def test_images_alone_send_a_null_state(self) -> None:
        state, images = parse_input(
            [{"role": "user", "content": [{"type": "input_image", "image_url": "data:x"}]}],
            fetch_image=_no_fetch,
        )
        self.assertIsNone(state)
        self.assertEqual(images, ["data:x"])

    def test_bad_inputs_are_rejected(self) -> None:
        for raw in (
            "",
            None,
            [],
            [{"role": "assistant", "content": "hi"}],
            [{"role": "user", "content": [{"type": "input_image", "file_id": "file-1"}]}],
            [{"role": "user", "content": [{"type": "input_image", "image_url": "ftp://x"}]}],
            [{"role": "user", "content": [{"type": "input_audio"}]}],
        ):
            with self.subTest(raw=raw), self.assertRaises(DecisionRequestError):
                parse_input(raw, fetch_image=_no_fetch)

    def test_private_image_hosts_are_not_fetched(self) -> None:
        for url in ("http://127.0.0.1:11434/admin/status", "http://localhost/x.png"):
            with self.subTest(url=url), self.assertRaises(DecisionRequestError):
                decisions.fetch_image_as_data_url(url)


class ResponseTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.refund, self.team, self.severity = parse_questions([REFUND, ROUTE, SEVERITY])

    def test_predicate_reads_d1_noul(self) -> None:
        self.assertEqual(
            translate_answer(self.refund, {"type": "noul", "noul": 0.92}),
            {"type": "predicate", "name": "refund", "probability": 0.92},
        )

    def test_choice_lists_probabilities_in_question_order(self) -> None:
        answer = translate_answer(
            self.team,
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"technical": 0.07, "billing": 0.93},
                "confidence": 0.9,
            },
        )
        self.assertEqual(
            answer,
            {
                "type": "choice",
                "name": "team",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.93},
                    {"value": "technical", "probability": 0.07},
                ],
                "confidence": 0.9,
            },
        )

    def test_score_is_the_weighted_level_index(self) -> None:
        answer = translate_answer(
            self.severity,
            {
                "type": "score",
                "probabilities": {
                    "Cosmetic: Appearance only.": 0.1,
                    "Workaround available: Another way works.": 0.7,
                    "Fully blocked: No workaround.": 0.2,
                },
            },
        )
        self.assertAlmostEqual(answer["score"], 1.1)
        self.assertEqual(answer["confidence"], 0.7)
        self.assertEqual(
            answer["probabilities"],
            [
                {"value": 0, "label": "Cosmetic", "probability": 0.1},
                {"value": 1, "label": "Workaround available", "probability": 0.7},
                {"value": 2, "label": "Fully blocked", "probability": 0.2},
            ],
        )

    def test_score_probabilities_may_be_positional(self) -> None:
        answer = translate_answer(self.severity, {"score": {"probabilities": [0, 0, 1]}})
        self.assertEqual(answer["score"], 2.0)

    def test_refusal_and_missing_answers(self) -> None:
        self.assertEqual(
            translate_answer(self.team, {"type": "refusal"}),
            {"type": "refusal", "name": "team"},
        )
        with self.assertRaises(DecisionBackendError):
            translate_answer(self.team, None)
        with self.assertRaises(DecisionBackendError):
            translate_answer(self.team, {"probabilities": {"billing": 1.0}})


class DecisionsHttpTests(unittest.TestCase):
    setUp = _s1.System1HttpTests.setUp
    tearDown = _s1.System1HttpTests.tearDown
    setUp_base = _s1.System1HttpTests.setUp_base
    tearDown_base = _s1.System1HttpTests.tearDown_base
    _active_alias_ready = _s1.System1HttpTests._active_alias_ready
    _post = _s1.System1HttpTests._post
    _get = _s1.System1HttpTests._get

    def test_decisions_endpoint_translates_both_ways(self) -> None:
        self.d1.answers = {  # type: ignore[attr-defined]
            "refund": {"type": "noul", "noul": 0.97},
            "team": {"type": "choice", "probabilities": {"billing": 0.8, "technical": 0.2}},
        }
        status, response = self._post(
            {"model": "d1-3B", "input": "I was charged twice.", "questions": [REFUND, ROUTE]},
            path="/v1/decisions",
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["object"], "decision")
        self.assertEqual(response["model"], "d1-3B")
        self.assertEqual(
            response["answers"],
            [
                {"type": "predicate", "name": "refund", "probability": 0.97},
                {
                    "type": "choice",
                    "name": "team",
                    "choice": "billing",
                    "probabilities": [
                        {"value": "billing", "probability": 0.8},
                        {"value": "technical", "probability": 0.2},
                    ],
                    "confidence": 0.8,
                },
            ],
        )
        path, request, _ = self.d1.requests[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/v1/systemone")
        self.assertEqual(request["state"], "I was charged twice.")
        self.assertEqual(request["questions"]["refund"]["type"], "noul")
        self.assertEqual(self.controller.transitions, [])

    def test_decisions_rejects_other_models_and_bad_requests(self) -> None:
        status, response = self._post(
            {"model": "main", "input": "x", "questions": [REFUND]}, path="/v1/decisions"
        )
        self.assertEqual(status, 404)
        self.assertEqual(response["error"]["code"], "model_not_found")
        status, response = self._post(
            {"model": "d1-3B", "input": "x", "questions": [{**REFUND, "type": "bool"}]},
            path="/v1/decisions",
        )
        self.assertEqual(status, 400)
        self.assertEqual(response["error"]["param"], "questions[0].type")
        self.assertEqual(self.d1.requests, [])  # type: ignore[attr-defined]

    def test_unmappable_d1_answer_is_a_bad_gateway(self) -> None:
        self.d1.answers = {}  # type: ignore[attr-defined]
        status, _ = self._post(
            {"model": "d1-3B", "input": "x", "questions": [REFUND]}, path="/v1/decisions"
        )
        self.assertEqual(status, 502)


class DisabledDecisionsHttpTests(unittest.TestCase):
    setUp = _s1.DisabledSystem1HttpTests.setUp
    tearDown = _s1.DisabledSystem1HttpTests.tearDown
    _active_alias_ready = _s1.DisabledSystem1HttpTests._active_alias_ready
    _post = _s1.DisabledSystem1HttpTests._post

    def test_decisions_is_not_found_without_a_decision_model(self) -> None:
        status, _ = self._post(
            {"model": "d1-3B", "input": "x", "questions": [REFUND]}, path="/v1/decisions"
        )
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
