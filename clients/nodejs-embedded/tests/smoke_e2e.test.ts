/**
 * Live e2e smoke test for the re-based (spec-generated transport) TypeScript
 * REST SDK (TD-126 Phase 4).
 *
 * Exercises the public SDK API — which now routes its core collection / record
 * / search wire calls through the GENERATED openapi-fetch transport
 * (src/generated/schema.ts) — against a LIVE ProximaDB server: health probe,
 * collection create -> list -> get -> delete, plus a record insert -> get
 * round-trip.
 *
 * Gated on PROXIMADB_SMOKE_URL so it never runs in the normal unit/contract
 * `npm test` (which has no server):
 *
 *   PROXIMADB_SMOKE_URL=http://127.0.0.1:5678 npx vitest run tests/smoke_e2e
 *
 * Copyright 2025 ProximaDB Contributors
 * Licensed under the Apache License, Version 2.0
 */
import { describe, expect, it } from "vitest";
import { ProximaDBClient } from "../src/client";
import { StorageEngine } from "../src/types";

const SMOKE_URL = process.env.PROXIMADB_SMOKE_URL;

const maybe = SMOKE_URL ? describe : describe.skip;

maybe("live e2e smoke (re-based generated transport)", () => {
  it("health + collection CRUD + record round-trip", async () => {
    const client = new ProximaDBClient({ url: SMOKE_URL!, maxRetries: 0 });

    // Health must be reachable.
    const ok = await client.ping();
    expect(ok).toBe(true);

    const name = `ts_smoke_${Date.now()}`;
    const dim = 4;

    // Create -> list -> get (all route through the generated transport).
    await client
      .createCollection(name)
      .dimension(dim)
      .engine(StorageEngine.SST)
      .execute();

    // listCollections must work and return an array; we do NOT assert global
    // membership of the just-created collection — like the Go pilot smoke test,
    // this is robust to a shared data dir (create -> list visibility may lag,
    // and the listing reflects all tenants' collections). Authoritative
    // visibility is checked via the direct getCollection below.
    const listed = await client.listCollections();
    expect(Array.isArray(listed)).toBe(true);

    const info = (await client.collection(name).info()) as { name?: string };
    expect(info).toBeDefined();
    expect(info.name).toBe(name);

    // Record insert -> get round-trip.
    const recordId = "rec_1";
    await client
      .collection(name)
      .insert()
      .id(recordId)
      .vector([0.1, 0.2, 0.3, 0.4])
      .meta("category", "smoke")
      .execute();

    const fetched = await client.collection(name).getVector(recordId);
    expect(fetched).not.toBeNull();
    expect((fetched as { id?: string }).id).toBe(recordId);

    // Cleanup: delete the collection.
    await client.deleteCollection(name);
  });
});
