"""
Utilities for testing with ProximaDB

With embedded mode, most functions are no-ops since no external server is needed.
"""

import time


def check_server_health(
    rest_url: str = "http://localhost:5678", grpc_url: str = "http://localhost:5679"
) -> tuple[bool, bool]:
    """
    Check if ProximaDB servers are healthy.

    With embedded mode, always returns (True, True) since no external server is needed.
    """
    return True, True


def wait_for_server(timeout: int = 30) -> bool:
    """
    Wait for ProximaDB server to be ready.

    With embedded mode, always returns True immediately.
    """
    return True


def ensure_server_running():
    """
    No-op for embedded mode - server not needed.

    Previously this function would check for a running ProximaDB server.
    With embedded mode, the database runs in-process.
    """
    pass


def create_test_collection(
    client, name: str, dimension: int = 384, engine: str = "nova"
):
    """
    Create a test collection with error handling

    Args:
        client: ProximaDB client
        name: Collection name
        dimension: Vector dimension
        engine: Storage engine (nova or sst)
    """
    try:
        # Try to delete if exists
        try:
            client.delete_collection(name)
            time.sleep(0.5)  # Brief pause after deletion
        except:
            pass

        # Create collection
        response = client.create_collection(
            name=name, dimension=dimension, engine=engine
        )

        # Verify creation
        collections = client.list_collections()
        # Handle both dict and Collection object responses
        collection_names = []
        collection_ids = []
        for c in collections:
            if hasattr(c, "id"):
                collection_ids.append(c.id)
            if hasattr(c, "config") and hasattr(c.config, "name"):
                collection_names.append(c.config.name)
            elif hasattr(c, "name"):
                collection_names.append(c.name)
            elif isinstance(c, dict):
                collection_names.append(c.get("id", c.get("name")))
                collection_ids.append(c.get("id", c.get("name")))

        # Check both names and IDs
        if name not in collection_names and name not in collection_ids:
            # For gRPC, the name is in config.name, not collection.id
            found = False
            for c in collections:
                if (
                    hasattr(c, "config")
                    and hasattr(c.config, "name")
                    and c.config.name == name
                ):
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Collection {name} not created successfully")

        return response

    except Exception as e:
        raise RuntimeError(f"Failed to create test collection: {e}")


def cleanup_test_collections(client, prefix: str = "test_"):
    """
    Clean up all test collections

    Args:
        client: ProximaDB client
        prefix: Collection name prefix to match
    """
    try:
        collections = client.list_collections()
        for collection in collections:
            # Handle both dict and Collection object responses
            if hasattr(collection, "id"):
                name = collection.id
            elif hasattr(collection, "name"):
                name = collection.name
            elif isinstance(collection, dict):
                name = collection.get("id", collection.get("name", ""))
            else:
                continue

            if name.startswith(prefix):
                try:
                    client.delete_collection(name)
                except:
                    pass  # Ignore deletion errors
    except:
        pass  # Ignore errors during cleanup


class ServerContext:
    """
    Context manager for ensuring server is running during tests

    Usage:
        with ServerContext() as server:
            # Run tests
            pass
    """

    def __init__(self):
        self.process = None

    def __enter__(self):
        ensure_server_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Server continues running after tests
        pass
