import pytest


@pytest.fixture
def anyio_backend() -> str:
    # The A2A SDK and LangGraph's async API both run on asyncio.
    return "asyncio"
