"""Database client abstraction with Supabase support."""

from typing import Protocol, Optional, Any
from config.settings import settings


class DatabaseClient(Protocol):
    """Protocol for database operations - allows swapping implementations."""
    
    def insert(self, table: str, data: dict) -> dict:
        """Insert a record into a table."""
        ...
    
    def select(self, table: str, query: Optional[dict] = None) -> list:
        """Select records from a table."""
        ...
    
    def update(self, table: str, data: dict, filters: dict) -> dict:
        """Update records in a table."""
        ...
    
    def delete(self, table: str, filters: dict) -> bool:
        """Delete records from a table."""
        ...


class MockDatabaseClient:
    """In-memory mock client for MVP development."""
    
    def __init__(self):
        self._storage: dict[str, list[dict]] = {}
    
    def insert(self, table: str, data: dict) -> dict:
        """Insert a record into the in-memory storage."""
        if table not in self._storage:
            self._storage[table] = []
        
        # Add an auto-incrementing ID
        data_copy = data.copy()
        data_copy["id"] = len(self._storage[table]) + 1
        self._storage[table].append(data_copy)
        return data_copy
    
    def select(self, table: str, query: Optional[dict] = None) -> list:
        """Select records from in-memory storage."""
        items = self._storage.get(table, [])
        if query:
            return [
                item for item in items 
                if all(item.get(k) == v for k, v in query.items())
            ]
        return items
    
    def update(self, table: str, data: dict, filters: dict) -> dict:
        """Update records in in-memory storage."""
        items = self.select(table, filters)
        for item in items:
            item.update(data)
        return items[0] if items else {}
    
    def delete(self, table: str, filters: dict) -> bool:
        """Delete records from in-memory storage."""
        items = self.select(table, filters)
        for item in items:
            self._storage[table].remove(item)
        return True
    
    def clear(self, table: Optional[str] = None) -> None:
        """Clear storage for testing."""
        if table:
            self._storage[table] = []
        else:
            self._storage = {}


class SupabaseClient:
    """Real Supabase client implementation for production."""
    
    _instance: Any = None
    
    @classmethod
    def get_client(cls):
        """Get or create Supabase client singleton."""
        if not settings.SUPABASE_ENABLED:
            return None
        
        if cls._instance is None:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                raise ValueError("Supabase URL and Key required when enabled")
            
            from supabase import create_client
            cls._instance = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
        return cls._instance
    
    def insert(self, table: str, data: dict) -> dict:
        """Insert a record into Supabase."""
        client = self.get_client()
        return client.table(table).insert(data).execute().data[0]
    
    def select(self, table: str, query: Optional[dict] = None) -> list:
        """Select records from Supabase."""
        client = self.get_client()
        q = client.table(table).select("*")
        if query:
            for key, value in query.items():
                q = q.eq(key, value)
        return q.execute().data
    
    def update(self, table: str, data: dict, filters: dict) -> dict:
        """Update records in Supabase."""
        client = self.get_client()
        q = client.table(table).update(data)
        for key, value in filters.items():
            q = q.eq(key, value)
        result = q.execute().data
        return result[0] if result else {}
    
    def delete(self, table: str, filters: dict) -> bool:
        """Delete records from Supabase."""
        client = self.get_client()
        q = client.table(table).delete()
        for key, value in filters.items():
            q = q.eq(key, value)
        q.execute()
        return True


# Global mock instance for MVP
_mock_client = MockDatabaseClient()


def get_database_client() -> DatabaseClient:
    """Factory function for FastAPI dependency injection."""
    if settings.SUPABASE_ENABLED:
        return SupabaseClient()
    return _mock_client
