from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

from pgtask import Task
from pydantic_ai.exceptions import UserError

StepT = TypeVar('StepT')
PayloadT = TypeVar('PayloadT')
ResultT = TypeVar('ResultT')


@dataclass
class DurableTaskContext:
    """The current pgtask task plus per-name occurrence counters for its steps.

    pgtask disambiguates repeated step names with an explicit `occurrence` argument
    instead of counting encounters itself. This context assigns occurrences in
    encounter order, so as long as steps are reached in a deterministic order a
    replayed attempt maps each step call back to the same checkpoint.
    """

    task: Task
    _occurrences: Counter[str] = field(default_factory=Counter)

    async def step(self, name: str, operation: Callable[[], Awaitable[StepT]]) -> StepT:
        occurrence = self._occurrences[name]
        self._occurrences[name] += 1
        return await self.task.step(name, operation, occurrence=occurrence)


_current_context: ContextVar[DurableTaskContext | None] = ContextVar('pydantic_ai_pgtask_context', default=None)


def current_context() -> DurableTaskContext | None:
    """Return the current durable task context, or None when not inside one."""
    return _current_context.get()


@asynccontextmanager
async def durable(task: Task) -> AsyncIterator[DurableTaskContext]:
    """Enter a durable context for `task`, making agent runs inside it durable.

    Usually you use [`durable_task`][pydantic_ai_pgtask.durable_task] instead and never
    touch this directly.
    """
    if _current_context.get() is not None:
        raise UserError('A durable pgtask context is already active; durable contexts cannot be nested.')
    ctx = DurableTaskContext(task)
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


def durable_task(
    handler: Callable[[Task, PayloadT], Awaitable[ResultT]],
) -> Callable[[Task, PayloadT], Awaitable[ResultT]]:
    """Wrap a pgtask handler so agent runs inside it are checkpointed.

    Apply it between `@tasks.task(...)` and the handler:

    ```python
    @tasks.task('analyse')
    @durable_task
    async def analyse(task: Task, payload: dict[str, Any]) -> dict[str, Any]:
        result = await agent.run(payload['prompt'])
        return {'output': result.output}
    ```
    """

    @wraps(handler)
    async def wrapper(task: Task, payload: Any) -> ResultT:
        async with durable(task):
            return await handler(task, payload)

    return wrapper
