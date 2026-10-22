"""OpenAI-compatible Decisions API (``POST /v1/decisions``) on top of d1.

OpenAI's Decisions API asks typed questions about shared evidence and gets
back probabilities instead of generated text, which is what Liquid AI's d1
does on llama-server's ``/v1/systemone``.  This module translates between the
two shapes so OpenAI SDK clients can call the broker's resident decision
model unchanged:

=============  =================================  ==============================
OpenAI type    OpenAI question fields             d1 question
=============  =================================  ==============================
``predicate``  ``instructions``                   ``noul``
``choice``     ``choices[{value, description}]``  ``choice`` with ``criteria``
                                                  ``{value: description}``
``score``      ``levels[{label, description}]``   ``score`` with ``criteria``
               (lowest first)                     ``["label: description", ...]``
=============  =================================  ==============================

``input`` (a string, or user messages with ``input_text``/``input_image``
parts) becomes d1's ``state`` and ``images``.  d1 only takes data URLs, so
public ``http(s)`` image URLs are downloaded by the broker first.
"""

from __future__ import annotations

import base64
import ipaddress
import math
import socket
import time
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


QUESTION_TYPES = ("predicate", "choice", "score")
_D1_TYPES = {"predicate": "noul", "choice": "choice", "score": "score"}
MAX_QUESTIONS = 64
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_FETCH_TIMEOUT = 10.0


class DecisionRequestError(ValueError):
    """The client's request is not a valid Decisions request (HTTP 400)."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param

    def as_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": "invalid_request_error",
                "param": self.param,
                "code": None,
            }
        }


class DecisionBackendError(RuntimeError):
    """d1 answered with something the broker cannot map back (HTTP 502)."""


@dataclass(frozen=True)
class Question:
    name: str
    type: str
    instructions: str
    # choice: option values; score: level labels.  Empty for predicates.
    options: tuple[str, ...] = ()
    # The d1 criteria strings for score levels, parallel to ``options``.
    criteria: tuple[str, ...] = ()


def _required_string(record: Mapping[str, Any], key: str, param: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DecisionRequestError(f"{param} must be a non-empty string", param=param)
    return value


def _parse_options(
    record: Mapping[str, Any], key: str, label_key: str, param: str
) -> list[tuple[str, str]]:
    items = record.get(key)
    if not isinstance(items, list) or len(items) < 2:
        raise DecisionRequestError(f"{param} must list at least two entries", param=param)
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        item_param = f"{param}[{index}]"
        if not isinstance(item, Mapping):
            raise DecisionRequestError(f"{item_param} must be an object", param=item_param)
        label = _required_string(item, label_key, f"{item_param}.{label_key}")
        description = item.get("description", "")
        if not isinstance(description, str):
            raise DecisionRequestError(
                f"{item_param}.description must be a string",
                param=f"{item_param}.description",
            )
        if label in seen:
            raise DecisionRequestError(
                f"{item_param}.{label_key} {label!r} is not unique",
                param=f"{item_param}.{label_key}",
            )
        seen.add(label)
        result.append((label, description))
    return result


def parse_questions(raw: Any) -> list[Question]:
    if not isinstance(raw, list) or not raw:
        raise DecisionRequestError("questions must be a non-empty array", param="questions")
    if len(raw) > MAX_QUESTIONS:
        raise DecisionRequestError(
            f"questions may hold at most {MAX_QUESTIONS} entries", param="questions"
        )
    questions: list[Question] = []
    names: set[str] = set()
    for index, record in enumerate(raw):
        param = f"questions[{index}]"
        if not isinstance(record, Mapping):
            raise DecisionRequestError(f"{param} must be an object", param=param)
        kind = record.get("type")
        if kind not in QUESTION_TYPES:
            raise DecisionRequestError(
                f"{param}.type must be one of {', '.join(QUESTION_TYPES)}",
                param=f"{param}.type",
            )
        name = _required_string(record, "name", f"{param}.name")
        if name in names:
            raise DecisionRequestError(
                f"{param}.name {name!r} is not unique", param=f"{param}.name"
            )
        names.add(name)
        instructions = _required_string(record, "instructions", f"{param}.instructions")
        if kind == "choice":
            options = _parse_options(record, "choices", "value", f"{param}.choices")
            questions.append(
                Question(name, kind, instructions, tuple(value for value, _ in options),
                         tuple(description for _, description in options))
            )
        elif kind == "score":
            levels = _parse_options(record, "levels", "label", f"{param}.levels")
            questions.append(
                Question(
                    name,
                    kind,
                    instructions,
                    tuple(label for label, _ in levels),
                    tuple(
                        f"{label}: {description}" if description else label
                        for label, description in levels
                    ),
                )
            )
        else:
            questions.append(Question(name, kind, instructions))
    return questions


ImageFetcher = Callable[[str], str]


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            return False
    return bool(infos)


class _PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        host = urlsplit(newurl).hostname
        if urlsplit(newurl).scheme not in {"http", "https"} or not host or not _public_host(host):
            raise DecisionRequestError("image_url redirected to a non-public URL", param="input")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_IMAGE_OPENER = urllib.request.build_opener(_PublicRedirectHandler)


def fetch_image_as_data_url(url: str) -> str:
    """Download a public image URL and return it as a base64 data URL.

    Only globally routable hosts are fetched, so a client cannot use the
    broker to read loopback or LAN services.
    """

    host = urlsplit(url).hostname
    if not host or not _public_host(host):
        raise DecisionRequestError(
            "image_url must be a data URL or a public http(s) URL", param="input"
        )
    request = urllib.request.Request(url, headers={"User-Agent": "llama-broker"})
    try:
        with _IMAGE_OPENER.open(request, timeout=IMAGE_FETCH_TIMEOUT) as response:
            content_type = response.headers.get_content_type()
            data = response.read(MAX_IMAGE_BYTES + 1)
    except OSError as error:
        raise DecisionRequestError(f"could not download image: {error}", param="input") from error
    if len(data) > MAX_IMAGE_BYTES:
        raise DecisionRequestError("image is larger than 20 MiB", param="input")
    if not content_type.startswith("image/"):
        raise DecisionRequestError(
            f"image_url returned {content_type}, not an image", param="input"
        )
    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"


def _image(url: Any, param: str, fetch_image: ImageFetcher) -> str:
    if isinstance(url, Mapping):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        raise DecisionRequestError(f"{param}.image_url must be a URL string", param=param)
    if url.startswith("data:"):
        return url
    if urlsplit(url).scheme in {"http", "https"}:
        return fetch_image(url)
    raise DecisionRequestError(
        f"{param}.image_url must be a data URL or an http(s) URL", param=param
    )


def parse_input(
    raw: Any, *, fetch_image: ImageFetcher = fetch_image_as_data_url
) -> tuple[str | None, list[str]]:
    """Return d1's ``state`` text and ``images`` list for an OpenAI ``input``."""

    if isinstance(raw, str):
        if not raw.strip():
            raise DecisionRequestError("input must not be empty", param="input")
        return raw, []
    if not isinstance(raw, list) or not raw:
        raise DecisionRequestError(
            "input must be a string or a non-empty array of messages", param="input"
        )
    texts: list[str] = []
    images: list[str] = []
    for index, message in enumerate(raw):
        param = f"input[{index}]"
        if isinstance(message, str):
            texts.append(message)
            continue
        if not isinstance(message, Mapping):
            raise DecisionRequestError(f"{param} must be a message object", param=param)
        role = message.get("role", "user")
        if role != "user":
            raise DecisionRequestError(
                f"{param}.role must be 'user'", param=f"{param}.role"
            )
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            raise DecisionRequestError(
                f"{param}.content must be a string or an array of parts",
                param=f"{param}.content",
            )
        for part_index, part in enumerate(content):
            part_param = f"{param}.content[{part_index}]"
            kind = part.get("type") if isinstance(part, Mapping) else None
            if kind == "input_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif kind == "input_image":
                if part.get("file_id") is not None:
                    raise DecisionRequestError(
                        "file_id images are not supported", param=part_param
                    )
                images.append(_image(part.get("image_url"), part_param, fetch_image))
            else:
                raise DecisionRequestError(
                    f"{part_param} must be an input_text or input_image part",
                    param=part_param,
                )
    state = "\n\n".join(text for text in texts if text) or None
    if state is None and not images:
        raise DecisionRequestError("input has no text or images", param="input")
    return state, images


def systemone_request(
    state: str | None, images: Sequence[str], questions: Sequence[Question]
) -> dict[str, Any]:
    d1_questions: dict[str, Any] = {}
    for question in questions:
        record: dict[str, Any] = {
            "type": _D1_TYPES[question.type],
            "instructions": question.instructions,
        }
        if question.type == "choice":
            record["criteria"] = dict(zip(question.options, question.criteria))
        elif question.type == "score":
            record["criteria"] = list(question.criteria)
        d1_questions[question.name] = record
    request: dict[str, Any] = {"state": state, "questions": d1_questions}
    if images:
        request["images"] = list(images)
    return request


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _option_probabilities(answer: Mapping[str, Any], question: Question) -> list[float]:
    """Read d1's per-option probabilities in the question's option order.

    d1 keys them by the criteria labels; a list (positional, or of
    ``{label, probability}`` records) and level indices are accepted too.
    """

    raw = answer.get("probabilities", answer.get("probs"))
    nested = answer.get(question.type)
    if raw is None and isinstance(nested, Mapping):
        raw = nested.get("probabilities", nested.get("probs"))
    keys = [
        {option, criterion, str(index)}
        for index, (option, criterion) in enumerate(zip(question.options, question.criteria))
    ]
    found: list[float | None] = [None] * len(question.options)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            for index, accepted in enumerate(keys):
                if str(key) in accepted:
                    found[index] = _number(value)
    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                label = item.get("label", item.get("value", item.get("name")))
                value = _number(item.get("probability", item.get("prob")))
                for position, accepted in enumerate(keys):
                    if str(label) in accepted:
                        found[position] = value
            elif index < len(found):
                found[index] = _number(item)
    if any(value is None for value in found):
        raise DecisionBackendError(
            f"d1 answer for {question.name!r} lacks a probability for every option"
        )
    return [min(max(value, 0.0), 1.0) for value in found]  # type: ignore[arg-type]


def translate_answer(question: Question, answer: Any) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        raise DecisionBackendError(f"d1 returned no answer for {question.name!r}")
    if answer.get("type") == "refusal" or "refusal" in answer:
        return {"type": "refusal", "name": question.name}
    if question.type == "predicate":
        probability = _number(answer.get("noul", answer.get("probability")))
        if probability is None:
            raise DecisionBackendError(f"d1 answer for {question.name!r} has no probability")
        return {
            "type": "predicate",
            "name": question.name,
            "probability": min(max(probability, 0.0), 1.0),
        }
    probabilities = _option_probabilities(answer, question)
    confidence = _number(answer.get("confidence"))
    if confidence is None:
        confidence = max(probabilities)
    if question.type == "choice":
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "name": question.name,
            "choice": question.options[best],
            "probabilities": [
                {"value": value, "probability": probability}
                for value, probability in zip(question.options, probabilities)
            ],
            "confidence": confidence,
        }
    # OpenAI defines the score as the probability-weighted mean level index,
    # so it is computed here rather than trusting d1's own scale.
    total = sum(probabilities)
    score = sum(index * p for index, p in enumerate(probabilities)) / total if total else 0.0
    return {
        "type": "score",
        "name": question.name,
        "score": score,
        "probabilities": [
            {"value": index, "label": label, "probability": probability}
            for index, (label, probability) in enumerate(zip(question.options, probabilities))
        ],
        "confidence": confidence,
    }


def decision_response(
    model: str, questions: Sequence[Question], d1_answers: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "id": f"dec_{uuid.uuid4().hex}",
        "object": "decision",
        "created_at": int(time.time()),
        "model": model,
        "answers": [
            translate_answer(question, d1_answers.get(question.name))
            for question in questions
        ],
    }
