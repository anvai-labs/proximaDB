"""ProximaDB CLI - Command Line Interface for ProximaDB operations.

This module provides a comprehensive CLI for interacting with ProximaDB,
supporting both REST and gRPC protocols.

Usage:
    proximadb --help
    proximadb collections list
    proximadb vectors insert --collection my_collection --file vectors.json
    proximadb search --collection my_collection --query "[1.0, 2.0, ...]" --top-k 10
"""

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Local imports
from proximadb_sdk.config import PortMode
from proximadb_sdk.unified_client import ProximaDBClient as UnifiedProximaDBClient

console = Console()


def _record_payload(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Normalize legacy CLI vector JSON into the SDK record write shape."""
    payload = dict(record)
    if "props" not in payload and "metadata" in payload:
        payload["props"] = payload.pop("metadata")
    return payload


def _jsonable(obj: Any) -> Any:
    """Convert SDK return values (pydantic models) into JSON-native data.

    The clients return pydantic BaseModels (Collection, SearchResult,
    BatchResult, VectorOperationResponse, ...) - NOT dicts. Rendering paths
    that call .get()/.items()/json.dumps on them crash, so every result is
    normalized here before display (review finding 2026-11-08: every read
    command exited 1 on a healthy server before this).
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _fail_if_unreachable(host: str, rest_port: int, timeout: float = 30.0) -> None:
    """Fail LOUDLY when the server is unreachable.

    The v1 client treats connection failures as a signal to activate its
    local (in-memory) fallback - silently. For a CLI that means a dead
    server renders as 'No collections found.' with exit 0 (#2469 review
    finding 5: scripts cannot distinguish empty-from-down). Probe the
    health endpoint once; if the transport is down, say so and exit
    nonzero BEFORE any command output looks like an answer. ANY HTTP
    response (even 404/405) proves reachability - only transport-level
    failures (connect refused, timeout, DNS) trigger the exit.

    Scope: the probe validates the REST/unified port (the CLI default and
    the client's URL base). In split-port mode the client dials --grpc-port
    for gRPC (ClientConfig.grpc_port), but the probe still checks only REST:
    a deployment with REST down and gRPC up exits 2.
    """
    import httpx

    try:
        httpx.get(f"http://{host}:{rest_port}/health", timeout=timeout)
    except httpx.HTTPError as e:
        click.echo(
            f"Error: ProximaDB server unreachable at {host}:{rest_port} "
            f"({type(e).__name__}). Check that the server is running and "
            "the ports are correct.",
            err=True,
        )
        sys.exit(2)


def get_client(
    host: str,
    rest_port: int,
    grpc_port: int,
    protocol: str,
    timeout: float,
) -> UnifiedProximaDBClient:
    """Create a ProximaDB client for the CLI's connection options.

    The CLI exposes separate REST/gRPC ports (legacy multi-port layout) plus a
    protocol selector; the client takes a single base URL plus ``protocol``/
    ``port_mode`` (unified mode multiplexes all protocols on one port - the
    server default since the 5678 unification).

    Reachability is probed before returning: a dead server must exit
    nonzero, never fall through to the client's silent local fallback.
    """
    _fail_if_unreachable(host, rest_port, timeout)
    mode = PortMode.MULTI if int(grpc_port) != int(rest_port) else PortMode.UNIFIED
    # In split-port mode the gRPC client must dial --grpc-port; before
    # ClientConfig.grpc_port existed it always dialled :5679 (#2487 review).
    extra = {"grpc_port": int(grpc_port)} if mode == PortMode.MULTI else {}
    return UnifiedProximaDBClient(
        url=f"http://{host}:{rest_port}",
        protocol=protocol,
        port_mode=mode,
        timeout=timeout,
        **extra,
    )


@click.group()
@click.option("--host", default="localhost", help="ProximaDB server host")
@click.option("--rest-port", default=5678, help="REST API port")
@click.option(
    "--grpc-port",
    default=5678,
    help="gRPC API port (defaults to the unified multiplexed port 5678; "
    "only differs on legacy split-port deployments)",
)
@click.option(
    "--protocol",
    type=click.Choice(["auto", "grpc", "rest"]),
    # Default REST: against the unified server, `auto` probes gRPC first and
    # a 404/UNIMPLEMENTED http2 answer (a server ANSWER, not a connection
    # error) propagates without REST fallback (TD-SDKERR-1 semantics) - the
    # stock invocation must not depend on that gamble.
    default="rest",
    help="Protocol to use (auto probes gRPC first; rest is the safe default)",
)
@click.option("--timeout", default=30.0, help="Request timeout in seconds")
@click.option("--json-output", is_flag=True, help="Output results as JSON")
@click.version_option(version="0.2.0", prog_name="ProximaDB CLI")
@click.pass_context
def cli(
    ctx: click.Context,
    host: str,
    rest_port: int,
    grpc_port: int,
    protocol: str,
    timeout: float,
    json_output: bool,
) -> None:
    """ProximaDB CLI - Cloud-native vector database for AI applications.

    Use this CLI to manage collections, insert vectors, and perform
    similarity searches against your ProximaDB server.
    """
    ctx.ensure_object(dict)
    ctx.obj["host"] = host
    ctx.obj["rest_port"] = rest_port
    ctx.obj["grpc_port"] = grpc_port
    ctx.obj["protocol"] = protocol
    ctx.obj["timeout"] = timeout
    ctx.obj["json_output"] = json_output


# --- Collection Commands ---


@cli.group()
def collections() -> None:
    """Manage ProximaDB collections."""
    pass


@collections.command("list")
@click.pass_context
def list_collections(ctx: click.Context) -> None:
    """List all collections in the database."""
    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        result = _jsonable(client.list_collections())

        if ctx.obj["json_output"]:
            click.echo(json.dumps(result, indent=2))
        else:
            if not result:
                console.print("[yellow]No collections found.[/yellow]")
                return

            table = Table(title="Collections")
            table.add_column("Name", style="cyan")
            table.add_column("Dimension", style="green")
            table.add_column("Vector Count", style="magenta")
            table.add_column("Storage Engine", style="blue")

            for coll in result:
                # Collection dumps nest identity/config under "config" and
                # counts under "stats" (the SDK model shape); fall back to
                # flat keys for any dict-shaped result.
                cfg = coll.get("config") or {}
                stats = coll.get("stats") or {}
                table.add_row(
                    cfg.get("name") or coll.get("name") or coll.get("id") or "N/A",
                    str(cfg.get("dimension") or coll.get("dimension") or "N/A"),
                    str(stats.get("vector_count") or coll.get("vector_count") or "N/A"),
                    cfg.get("storage_engine")
                    or coll.get("storage_engine")
                    or "default",
                )

            console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@collections.command("create")
@click.argument("name")
@click.option("--dimension", "-d", required=True, type=int, help="Vector dimension")
@click.option(
    "--engine",
    type=click.Choice(["sst", "viper", "nova", "swift", "raptor", "helix"]),
    default="sst",
    help="Storage engine to use",
)
@click.option("--description", help="Collection description")
@click.pass_context
def create_collection(
    ctx: click.Context,
    name: str,
    dimension: int,
    engine: str,
    description: str | None,
) -> None:
    """Create a new collection."""
    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        result = _jsonable(
            client.create_collection(
                name=name,
                dimension=dimension,
                storage_engine=engine,
                description=description,
            )
        )

        if ctx.obj["json_output"]:
            click.echo(json.dumps(result, indent=2))
        else:
            console.print(
                Panel(
                    f"[green]Collection '{name}' created successfully![/green]\n"
                    f"Dimension: {dimension}\n"
                    f"Engine: {engine}",
                    title="Success",
                )
            )
    except Exception as e:
        console.print(f"[red]Error creating collection: {e}[/red]")
        sys.exit(1)


@collections.command("delete")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_collection(ctx: click.Context, name: str, force: bool) -> None:
    """Delete a collection."""
    if not force:
        if not click.confirm(f"Are you sure you want to delete collection '{name}'?"):
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        client.delete_collection(name)

        if ctx.obj["json_output"]:
            click.echo(json.dumps({"status": "deleted", "collection": name}))
        else:
            console.print(f"[green]Collection '{name}' deleted successfully.[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting collection: {e}[/red]")
        sys.exit(1)


@collections.command("info")
@click.argument("name")
@click.pass_context
def collection_info(ctx: click.Context, name: str) -> None:
    """Get detailed information about a collection."""
    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        result = _jsonable(client.get_collection(name))

        if ctx.obj["json_output"]:
            click.echo(json.dumps(result, indent=2))
        else:
            table = Table(title=f"Collection: {name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in result.items():
                table.add_row(key, str(value))

            console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# --- Vector Commands ---


@cli.group()
def vectors() -> None:
    """Manage vectors in collections."""
    pass


@vectors.command("insert")
@click.option("--collection", "-c", required=True, help="Collection name")
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="JSON file with vectors"
)
@click.option("--vector", "-v", help="Single vector as JSON array")
@click.option(
    "--id",
    "vector_id",
    help="Vector ID (single vector; a UUID is generated if omitted)",
)
@click.option("--metadata", "-m", help="Metadata as JSON object")
@click.pass_context
def insert_vectors(
    ctx: click.Context,
    collection: str,
    file: str | None,
    vector: str | None,
    vector_id: str | None,
    metadata: str | None,
) -> None:
    """Insert vectors into a collection.

    Either provide a JSON file with vectors or a single vector.

    File format:
    [
        {"id": "vec1", "vector": [1.0, 2.0, ...], "metadata": {...}},
        ...
    ]
    """
    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        vectors_data: list[dict[str, Any]] = []

        if file:
            with open(file) as f:
                vectors_data = json.load(f)
        elif vector:
            vec = json.loads(vector)
            meta = json.loads(metadata) if metadata else {}
            record: dict[str, Any] = {"vector": vec, "metadata": meta}
            # No --id: leave the key out so the SDK mints a UUID. The old
            # literal "auto" made every ID-less insert overwrite the last.
            if vector_id:
                record["id"] = vector_id
            vectors_data = [record]
        else:
            console.print("[red]Either --file or --vector is required.[/red]")
            sys.exit(1)

        records_data = [_record_payload(record) for record in vectors_data]
        result = _jsonable(client.insert_records(collection, records_data))

        if ctx.obj["json_output"]:
            click.echo(json.dumps(result, indent=2))
        else:
            console.print(
                f"[green]Successfully inserted {len(vectors_data)} vector(s) "
                f"into '{collection}'[/green]"
            )
    except Exception as e:
        console.print(f"[red]Error inserting vectors: {e}[/red]")
        sys.exit(1)


@vectors.command("get")
@click.option("--collection", "-c", required=True, help="Collection name")
@click.argument("vector_id")
@click.pass_context
def get_vector(ctx: click.Context, collection: str, vector_id: str) -> None:
    """Get a vector by ID."""
    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        result = _jsonable(client.get_vector(collection, vector_id))

        if ctx.obj["json_output"]:
            click.echo(json.dumps(result, indent=2))
        else:
            console.print(
                Panel(
                    Syntax(json.dumps(result, indent=2), "json"),
                    title=f"Vector: {vector_id}",
                )
            )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@vectors.command("delete")
@click.option("--collection", "-c", required=True, help="Collection name")
@click.argument("vector_ids", nargs=-1)
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_vectors(
    ctx: click.Context,
    collection: str,
    vector_ids: tuple,
    force: bool,
) -> None:
    """Delete vectors by ID."""
    if not vector_ids:
        console.print("[red]At least one vector ID is required.[/red]")
        sys.exit(1)

    if not force:
        if not click.confirm(
            f"Delete {len(vector_ids)} vector(s) from '{collection}'?"
        ):
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        result = _jsonable(client.delete_vectors(collection, list(vector_ids)))

        if ctx.obj["json_output"]:
            click.echo(json.dumps(result, indent=2))
        else:
            console.print(
                f"[green]Deleted {len(vector_ids)} vector(s) from '{collection}'[/green]"
            )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# --- Search Commands ---


@cli.command("search")
@click.option("--collection", "-c", required=True, help="Collection name")
@click.option("--query", "-q", required=True, help="Query vector as JSON array")
@click.option("--top-k", "-k", default=10, help="Number of results to return")
@click.option("--filter", "-f", "filter_expr", help="Filter expression (JSON)")
# NOTE: the historical --metric option was removed (2026-11-08 review): no
# distance-metric parameter exists anywhere in the client call chain, so it
# was silently ignored - the collection's configured metric always applied.
@click.pass_context
def search(
    ctx: click.Context,
    collection: str,
    query: str,
    top_k: int,
    filter_expr: str | None,
) -> None:
    """Perform similarity search.

    Example:
        proximadb search -c my_collection -q "[1.0, 2.0, 3.0]" -k 5
    """
    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        query_vector = json.loads(query)
        # The client accepts a plain dict (or FilterBuilder) as the metadata
        # filter - no SDK-side SearchFilter model exists anymore.
        search_filter = json.loads(filter_expr) if filter_expr else None

        results = _jsonable(
            client.search(
                collection,
                query_vector,
                top_k=top_k,
                metadata_filter=search_filter,
            )
        )

        if ctx.obj["json_output"]:
            click.echo(json.dumps(results, indent=2))
        else:
            table = Table(title=f"Search Results (top {top_k})")
            table.add_column("Rank", style="dim")
            table.add_column("ID", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Metadata", style="yellow")

            for i, result in enumerate(results, 1):
                metadata_str = json.dumps(result.get("metadata", {}))[:50]
                if len(json.dumps(result.get("metadata", {}))) > 50:
                    metadata_str += "..."
                table.add_row(
                    str(i),
                    result.get("id", "N/A"),
                    f"{result.get('score', 0):.4f}",
                    metadata_str,
                )

            console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# --- Server Commands ---


@cli.group()
def server() -> None:
    """Server management commands."""
    pass


@server.command("health")
@click.pass_context
def health_check(ctx: click.Context) -> None:
    """Check server health status."""
    import httpx

    try:
        url = f"http://{ctx.obj['host']}:{ctx.obj['rest_port']}/health"
        response = httpx.get(url, timeout=ctx.obj["timeout"])

        if response.status_code == 200:
            if ctx.obj["json_output"]:
                click.echo(
                    json.dumps({"status": "healthy", "details": response.json()})
                )
            else:
                console.print("[green]Server is healthy![/green]")
                console.print(
                    Panel(
                        Syntax(json.dumps(response.json(), indent=2), "json"),
                        title="Health Details",
                    )
                )
        else:
            console.print(
                f"[yellow]Server returned status {response.status_code}[/yellow]"
            )
    except Exception as e:
        console.print(f"[red]Server unreachable: {e}[/red]")
        sys.exit(1)


@server.command("info")
@click.pass_context
def server_info(ctx: click.Context) -> None:
    """Get server information."""
    import httpx

    try:
        url = f"http://{ctx.obj['host']}:{ctx.obj['rest_port']}/api/v2/info"
        response = httpx.get(url, timeout=ctx.obj["timeout"])

        if ctx.obj["json_output"]:
            click.echo(response.text)
        else:
            data = response.json()
            table = Table(title="Server Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in data.items():
                table.add_row(key, str(value))

            console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# --- Utility Commands ---


@cli.command("benchmark")
@click.option("--collection", "-c", required=True, help="Collection name")
@click.option("--dimension", "-d", default=768, help="Vector dimension")
@click.option("--count", "-n", default=1000, help="Number of vectors to insert")
@click.option("--queries", "-q", default=100, help="Number of search queries")
@click.pass_context
def benchmark(
    ctx: click.Context,
    collection: str,
    dimension: int,
    count: int,
    queries: int,
) -> None:
    """Run a simple benchmark against the server."""
    import time

    import numpy as np

    try:
        client = get_client(
            ctx.obj["host"],
            ctx.obj["rest_port"],
            ctx.obj["grpc_port"],
            ctx.obj["protocol"],
            ctx.obj["timeout"],
        )

        console.print(f"[cyan]Running benchmark on '{collection}'...[/cyan]")

        # Create collection if needed
        try:
            client.create_collection(name=collection, dimension=dimension)
            console.print(f"  Created collection '{collection}'")
        except Exception:
            console.print(f"  Using existing collection '{collection}'")

        # Insert vectors
        console.print(f"  Inserting {count} vectors...")
        vectors = []
        for i in range(count):
            vectors.append(
                {
                    "id": f"bench_{i}",
                    "vector": np.random.randn(dimension).tolist(),
                    "metadata": {"index": i},
                }
            )

        start = time.time()
        records = [_record_payload(record) for record in vectors]
        client.insert_records(collection, records)
        insert_time = time.time() - start

        # Search
        console.print(f"  Running {queries} search queries...")
        query_times = []
        for _ in range(queries):
            query = np.random.randn(dimension).tolist()
            start = time.time()
            client.search(collection, query, top_k=10)
            query_times.append(time.time() - start)

        avg_query_time = sum(query_times) / len(query_times)

        # Results
        results = {
            "insert_total_time": f"{insert_time:.2f}s",
            "insert_vectors_per_second": f"{count / insert_time:.0f}",
            "search_avg_latency": f"{avg_query_time * 1000:.2f}ms",
            "search_qps": f"{1 / avg_query_time:.0f}",
        }

        if ctx.obj["json_output"]:
            click.echo(json.dumps(results, indent=2))
        else:
            console.print("\n[green]Benchmark Results:[/green]")
            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in results.items():
                table.add_row(key.replace("_", " ").title(), value)

            console.print(table)

    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
