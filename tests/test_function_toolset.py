from __future__ import annotations

import pytest
from pydantic_ai import FunctionToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_pgtask import PGTaskFunctionToolset

from .conftest import CheckpointStore, make_task, running_task

pytestmark = pytest.mark.anyio


def _make_toolset(calls: list[str]) -> FunctionToolset[None]:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def shout(value: str) -> str:
        calls.append(value)
        return value.upper()

    return toolset


def _run_context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt='x', messages=[])


async def test_id_passthrough() -> None:
    inner = _make_toolset([])
    toolset = PGTaskFunctionToolset(inner, step_name_prefix='a')
    assert toolset.id == inner.id


async def test_aenter_aexit() -> None:
    toolset = PGTaskFunctionToolset(_make_toolset([]), step_name_prefix='a')
    async with toolset as entered:
        assert entered is toolset


async def test_visit_and_replace_returns_self() -> None:
    toolset = PGTaskFunctionToolset(_make_toolset([]), step_name_prefix='a')

    def visitor(t: FunctionToolset[None]) -> FunctionToolset[None]:  # pragma: no cover
        return t

    assert toolset.visit_and_replace(visitor) is toolset


async def test_call_tool_without_context_passes_through() -> None:
    calls: list[str] = []
    inner = _make_toolset(calls)
    toolset = PGTaskFunctionToolset(inner, step_name_prefix='a')
    ctx = _run_context()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool('shout', {'value': 'hi'}, ctx, tools['shout'])
    assert result == 'HI'
    assert calls == ['hi']


async def test_call_tool_inside_context_is_checkpointed(store: CheckpointStore) -> None:
    calls: list[str] = []
    inner = _make_toolset(calls)
    toolset = PGTaskFunctionToolset(inner, step_name_prefix='a')
    ctx = _run_context()

    async with running_task(make_task(store)):
        tools = await toolset.get_tools(ctx)
        first = await toolset.call_tool('shout', {'value': 'hi'}, ctx, tools['shout'])

    # Retry after a simulated crash: the checkpointed result is served, so the
    # side effect happens exactly once.
    async with running_task(make_task(store)):
        replay = await toolset.call_tool('shout', {'value': 'hi'}, ctx, tools['shout'])

    assert first == replay == 'HI'
    assert calls == ['hi']
