from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pgtask import Client, Task, TaskRegistry, Worker
from pydantic_ai import Agent
from testcontainers.community.postgres import PostgresContainer

from pydantic_ai_pgtask import PGTaskDurability

from .conftest import make_model

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


def _docker_host_env() -> None:  # pragma: no cover - environment-dependent
    """testcontainers on macOS sometimes needs DOCKER_HOST pointing at the user socket."""
    if 'DOCKER_HOST' in os.environ:
        return
    home_sock = Path.home() / '.docker' / 'run' / 'docker.sock'
    if home_sock.exists():
        os.environ['DOCKER_HOST'] = f'unix://{home_sock}'


def _normalize_dsn(url: str) -> str:
    if url.startswith('postgresql+psycopg2://'):
        return 'postgresql://' + url.split('://', 1)[1]
    return url  # pragma: no cover - testcontainers always returns the psycopg2 form


@pytest.fixture(scope='session')
def postgres_container() -> Iterator[PostgresContainer]:
    _docker_host_env()
    container = PostgresContainer('postgres:16-alpine')
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope='session')
def db_dsn(postgres_container: PostgresContainer) -> str:
    return _normalize_dsn(postgres_container.get_connection_url())


async def test_agent_run_inside_worker_is_durable(db_dsn: str) -> None:
    """End to end through the public pgtask API: enqueue, work, checkpoint, complete."""
    counter = {'calls': 0}
    agent = Agent(make_model(counter), name='analyst', capabilities=[PGTaskDurability()])
    tasks = TaskRegistry(queue_name='agents')

    @tasks.task('analyse')
    async def analyse(task: Task, payload: dict[str, Any]) -> dict[str, Any]:
        result = await agent.run(payload['prompt'])
        return {'output': result.output}

    client = await Client.connect(db_dsn)
    await client.migrate()
    handle = await client.enqueue(analyse.request({'prompt': 'go'}))

    worker = Worker(db_dsn, tasks, concurrency=1, poll_interval=0.1)
    worker_run = asyncio.ensure_future(worker.run())
    try:
        result = await handle.result(timeout=30)
    finally:
        worker.shutdown()
        await worker_run

    assert result is not None
    assert result.state == 'succeeded'
    assert result.result == {'output': 'ok'}
    assert counter['calls'] == 1


async def test_replay_after_crash_serves_checkpoint(db_dsn: str) -> None:
    """The handler fails after the model call on attempt 1; attempt 2 replays the
    checkpointed response instead of re-calling the model."""
    counter = {'calls': 0}
    agent = Agent(make_model(counter), name='crashy', capabilities=[PGTaskDurability()])
    tasks = TaskRegistry(queue_name='crashes')
    attempts: list[int] = []

    @tasks.task('crash', retry_delay=0.1)
    async def crash(task: Task, payload: dict[str, Any]) -> dict[str, Any]:
        result = await agent.run('go')
        attempts.append(task.attempt)
        if task.attempt == 1:
            raise RuntimeError('simulated crash')
        return {'output': result.output}

    client = await Client.connect(db_dsn)
    await client.migrate()
    handle = await client.enqueue(crash.request({}, max_attempts=2))

    worker = Worker(db_dsn, tasks, concurrency=1, poll_interval=0.1)
    worker_run = asyncio.ensure_future(worker.run())
    try:
        result = await handle.result(timeout=30)
    finally:
        worker.shutdown()
        await worker_run

    assert result is not None
    assert result.state == 'succeeded'
    assert result.result == {'output': 'ok'}
    assert attempts == [1, 2]
    assert counter['calls'] == 1
