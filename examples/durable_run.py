"""Runnable version of the README example.

Spins up a Postgres testcontainer, migrates the pgtask schema, registers an
agent inside a durable task, enqueues a run, and drains the worker.

Run with:

    OPENAI_API_KEY=... uv run python examples/durable_run.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from pgtask import Client, Task, TaskRegistry, Worker
from pydantic_ai import Agent
from testcontainers.community.postgres import PostgresContainer

from pydantic_ai_pgtask import PGTaskDurability

tasks = TaskRegistry(queue_name='agents')
agent = Agent('openai:gpt-5.2', name='analyst', capabilities=[PGTaskDurability()])


@tasks.task('analyse')
async def analyse(task: Task, payload: dict[str, Any]) -> dict[str, Any]:
    result = await agent.run(payload['prompt'])
    return {'output': result.output}


async def main(dsn: str) -> None:
    client = await Client.connect(dsn)
    await client.migrate()
    handle = await client.enqueue(analyse.request({'prompt': 'Analyse Q3 revenue in one sentence.'}))

    worker = Worker(dsn, tasks, concurrency=1, poll_interval=0.1)
    worker_run = asyncio.ensure_future(worker.run())
    try:
        result = await handle.result(timeout=120)
    finally:
        worker.shutdown()
        await worker_run

    print(result)


if __name__ == '__main__':
    if 'DOCKER_HOST' not in os.environ:
        home_sock = Path.home() / '.docker' / 'run' / 'docker.sock'
        if home_sock.exists():
            os.environ['DOCKER_HOST'] = f'unix://{home_sock}'

    with PostgresContainer('postgres:16-alpine') as container:
        dsn = container.get_connection_url().replace('postgresql+psycopg2://', 'postgresql://')
        asyncio.run(main(dsn))
