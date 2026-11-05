"""Local tool metrics and a real MCP initialize/list/call exchange when SDK is installed."""

import asyncio
import json
from unittest.mock import Mock

import pytest
from conftest import document

from rag_service.mcp_server import LocalRuntime


def runtime_for(pipeline):
    pipeline.ingest([document("a")])
    loader = Mock(return_value=pipeline.embedder)
    runtime = LocalRuntime(pipeline.config.data_dir, "default", pipeline.store, loader)
    return runtime, loader


def test_metrics_recorded_and_returned_and_model_reused(pipeline):
    runtime, loader = runtime_for(pipeline)
    result = runtime.retrieve("backups retention", 6, 2500, None)
    again = runtime.retrieve("backups", 6, 2500, [])
    assert result.context.passages
    assert result.metrics.query_tokens == 2
    assert result.metrics.context_tokens == result.context.context_tokens
    assert result.metrics.result_tokens == pipeline.embedder.tokenizer.count(
        result.context.model_dump_json()
    )
    assert result.metrics.total_tokens == 2 + result.metrics.result_tokens
    assert result.metrics.response_time_ms >= 0
    assert result.metrics.token_counts_estimated
    assert result.metrics.generation_tokens == 0
    assert again.context.passages == []
    loader.assert_called_once()
    records = [json.loads(line) for line in runtime.metrics_file.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["request_id"] == result.metrics.request_id
    assert records[0]["response_time_ms"] == result.metrics.response_time_ms
    assert "query" not in records[0] and "passages" not in records[0]
    assert records[0]["ingestion_revision"] == 1


def test_failed_call_recorded_and_error_sanitized(pipeline):
    runtime, _ = runtime_for(pipeline)
    pipeline.embedder.fail = True
    with pytest.raises(ValueError, match="request_id=") as error:
        runtime.retrieve("private query", 6, 2500, None)
    assert "private query" not in str(error.value)
    row = json.loads(runtime.metrics_file.read_text())
    assert row["status"] == "error" and row["error_type"] == "RuntimeError"
    assert row["response_time_ms"] >= 0
    assert "private query" not in runtime.metrics_file.read_text()


def test_mcp_initialize_list_call_and_validation(pipeline):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from rag_service.mcp_server import create_server

    runtime, _ = runtime_for(pipeline)
    server = create_server(runtime)

    async def exercise():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            listing = await client.list_tools()
            assert [tool.name for tool in listing.tools] == ["retrieve_context"]
            assert listing.tools[0].outputSchema
            response = await client.call_tool("retrieve_context", {"query": "backups"})
            assert not response.isError
            assert response.structuredContent["context"]["passages"]
            assert response.structuredContent["metrics"]["query_tokens"] == 1
            invalid = await client.call_tool("retrieve_context", {"query": "x", "top_k": 101})
            assert invalid.isError
            empty = await client.call_tool(
                "retrieve_context", {"query": "backups", "document_ids": []}
            )
            assert not empty.isError
            assert empty.structuredContent["context"]["passages"] == []

    asyncio.run(exercise())


def test_stdio_subprocess_roundtrip(tmp_path):
    pytest.importorskip("mcp")
    import sys
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).with_name("mcp_fixture_server.py")), str(tmp_path)],
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            assert (await session.list_tools()).tools[0].name == "retrieve_context"
            result = await session.call_tool("retrieve_context", {"query": "backups"})
            assert not result.isError
            assert result.structuredContent["context"]["passages"][0]["citation_id"] == "S1"
            assert result.structuredContent["metrics"]["query_tokens"] == 1

    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
