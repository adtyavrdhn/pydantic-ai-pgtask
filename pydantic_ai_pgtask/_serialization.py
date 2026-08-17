from __future__ import annotations

from collections.abc import Mapping

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelResponse

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)


def serialize_response(response: ModelResponse) -> dict[str, object]:
    dumped = _response_adapter.dump_python(response, mode='json')
    assert isinstance(dumped, dict)
    return dumped


def deserialize_response(payload: Mapping[str, object]) -> ModelResponse:
    return _response_adapter.validate_python(payload)
