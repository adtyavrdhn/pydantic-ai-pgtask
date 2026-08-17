from __future__ import annotations

from typing import Any, cast

import pytest
from pgtask import Task
from pydantic_ai.exceptions import UserError

from pydantic_ai_pgtask import current_context, durable, durable_task

from .conftest import CheckpointStore

pytestmark = pytest.mark.anyio


async def test_no_context_by_default() -> None:
    assert current_context() is None


async def test_durable_sets_and_resets_context(task: Task) -> None:
    async with durable(task) as ctx:
        assert current_context() is ctx
        assert ctx.task is task
    assert current_context() is None


async def test_durable_rejects_nesting(task: Task) -> None:
    async with durable(task):
        with pytest.raises(UserError, match='cannot be nested'):
            async with durable(task):
                pass  # pragma: no cover


async def test_step_occurrences_count_per_name(task: Task, store: CheckpointStore) -> None:
    async with durable(task) as ctx:

        async def one() -> int:
            return 1

        assert await ctx.step('a', one) == 1
        assert await ctx.step('a', one) == 1
        assert await ctx.step('b', one) == 1

    assert store.executions == [('a', 0), ('a', 1), ('b', 0)]


async def test_replay_serves_cached_step(task: Task, store: CheckpointStore) -> None:
    calls = {'n': 0}

    async def op() -> int:
        calls['n'] += 1
        return calls['n']

    async with durable(task) as ctx:
        assert await ctx.step('s', op) == 1

    # A retry enters a fresh context: occurrence counters restart at 0, so the
    # step resolves to the same checkpoint and the operation does not re-run.
    async with durable(task) as ctx:
        assert await ctx.step('s', op) == 1

    assert calls['n'] == 1


async def test_durable_task_wraps_handler(task: Task) -> None:
    seen: dict[str, Any] = {}

    @durable_task
    async def handler(t: Task, payload: dict[str, Any]) -> str:
        seen['ctx'] = current_context()
        seen['task'] = t
        return cast(str, payload['value'])

    result = await handler(task, {'value': 'done'})
    assert result == 'done'
    assert seen['task'] is task
    assert seen['ctx'] is not None
    assert current_context() is None
