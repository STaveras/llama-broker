"""Keep both inherited generic autotune fixes under behavioral coverage."""
import json
from llama_broker.autotune import HttpCandidateSession


def test_reasoning_output_counts_in_json_sse_and_preserves_explicit_usage():
    session = object.__new__(HttpCandidateSession)
    for streaming in (False, True):
        part = {'content': '', 'reasoning_content': 'one two three'}
        payload = {'choices': [{('delta' if streaming else 'message'): part}]}
        measure = session._stream_measurement if streaming else session._json_measurement
        encode = lambda p: ('data: '+json.dumps(p)+'\n\ndata: [DONE]\n\n').encode() if streaming else json.dumps(p).encode()
        result = measure(encode(payload), .5)
        assert result.generated_tokens == 3
        payload['usage'] = {'completion_tokens': 7}
        assert measure(encode(payload), .5).generated_tokens == 7


def test_tool_payload_keeps_tools_without_forcing_json_grammar():
    session = HttpCandidateSession('http://127.0.0.1:1', 'fixture')
    for streaming in (False, True):
        payload = session._payload('tools-json', streaming=streaming)
        assert payload['tools']
        assert 'response_format' not in payload
