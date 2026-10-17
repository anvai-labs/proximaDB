// SPDX-License-Identifier: Apache-2.0
// Exercise the PUBLISHED npm client (`npm install @anvailabs/proximadb-client@X.Y.Z`)
// through its CommonJS entry (`exports["."].require`): installed version,
// health, create, insert, search, delete, drop. Release evidence, part (B).
//
// Copy this file INTO the temp project that holds the registry install and run
// it from there: Node resolves a bare specifier relative to the script's own
// directory, so running it from the repo would never see the install.
//
// Usage: node npm_smoke.cjs <base_url> <expected_version> [evidence.json]
"use strict";
const fs = require("fs");
const path = require("path");

const PKG = "@anvailabs/proximadb-client";
const [, , baseUrl, want, outFile] = process.argv;

function fail(msg) {
  console.log(`::error::npm smoke (CJS) FAILED: ${msg}`);
  process.exit(1);
}

async function retry(fn, ms = 10000) {
  const deadline = Date.now() + ms;
  for (;;) {
    try {
      return await fn();
    } catch (e) {
      if (Date.now() >= deadline) throw e;
      await new Promise((r) => setTimeout(r, 500));
    }
  }
}

(async () => {
  if (!baseUrl || !want) fail("usage: npm_smoke.cjs <base_url> <version> [out]");
  // `exports` does not expose ./package.json, so read it from disk.
  const pkgJson = path.join(process.cwd(), "node_modules", PKG, "package.json");
  const installed = JSON.parse(fs.readFileSync(pkgJson, "utf8")).version;
  if (installed !== want) fail(`installed ${PKG} ${installed} != ${want}`);
  const resolved = require.resolve(PKG);
  if (!resolved.includes(path.join("node_modules", PKG))) fail(`${PKG} resolved to ${resolved}, not the registry install`);

  const { connectRest, VERSION } = require(PKG);
  // v0.4.1 exported VERSION = "1.0.0" while publishing as 0.4.1.
  if (VERSION !== want) fail(`exported VERSION ${VERSION} != ${want}`);
  const client = connectRest(baseUrl);
  const steps = [];

  const health = await client.health();
  if (health.status !== "healthy" || health.version !== want) {
    fail(`health ${JSON.stringify(health)}; expected healthy ${want}`);
  }
  steps.push("health");

  const name = "rs_js_" + Math.random().toString(16).slice(2, 14);
  await client.createCollection(name).dimension(4).metric("cosine").execute();
  steps.push("create_collection");

  const col = client.collection(name);
  const n = await col.insertVectors(
    ["a", "b", "c"],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    [{ k: "x" }, { k: "y" }, { k: "z" }],
  );
  if (n !== 3) fail(`insertVectors returned ${n}, expected 3`);
  steps.push("insert");

  await retry(async () => {
    const ids = (await col.searchVectors([1, 0, 0, 0], 3)).map((h) => h.id);
    if (ids[0] !== "a" || ids.length !== 3) throw new Error(`search returned ${JSON.stringify(ids)}`);
  });
  steps.push("search");

  await col.deleteVector("a");
  steps.push("delete");
  await retry(async () => {
    const ids = (await col.searchVectors([1, 0, 0, 0], 3)).map((h) => h.id);
    if (ids.includes("a")) throw new Error(`deleted id 'a' still returned: ${JSON.stringify(ids)}`);
  });
  steps.push("search_after_delete");

  await client.deleteCollection(name);
  steps.push("delete_collection");

  const report = { artifact: `npm:${PKG}`, entry: "require", installed_version: installed, node: process.version, resolved, steps, passed: true };
  const rendered = JSON.stringify(report, null, 2);
  if (outFile) fs.writeFileSync(outFile, rendered + "\n");
  console.log(rendered);
})().catch((e) => fail(e && e.stack ? e.stack : String(e)));
