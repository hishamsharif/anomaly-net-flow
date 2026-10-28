from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
import duckdb

from api.config import settings

# Async engine — used by FastAPI request handlers
engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Sync engine — used by background training threads (same SQLite file)
_sync_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite").replace(
    "postgresql+asyncpg", "postgresql+psycopg2"
)
sync_engine = create_engine(_sync_url, echo=False)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_db() -> Session:
    """Sync session for use inside background training threads."""
    return SyncSessionLocal()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_duckdb() -> duckdb.DuckDBPyConnection:
    """In-process DuckDB for fast columnar training reads."""
    return duckdb.connect(":memory:")
