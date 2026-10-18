// SPDX-License-Identifier: Apache-2.0
// The published npm client through its ESM entry (`exports["."].import`).
// package.json advertises an `import` condition, so an ESM consumer
// (`import { connectRest } from "@anvailabs/proximadb-client"`) resolves
// THAT file, not the CommonJS one npm_smoke.cjs exercises. A broken `import`
// target breaks every ESM user while the CJS smoke stays green -- which is why
// it is its own check. Same copy-into-the-install-dir rule as npm_smoke.cjs.
//
// Usage: node npm_esm_smoke.mjs <base_url> <expected_version>
const PKG = "@anvailabs/proximadb-client";
const [, , baseUrl, want] = process.argv;
let mod;
try {
  mod = await import(PKG);
} catch (e) {
  console.log(`::error::npm smoke (ESM) FAILED: \`import "${PKG}"\` does not load: ${e.message}`);
  process.exit(1);
}
if (mod.VERSION !== want) {
  console.log(`::error::npm smoke (ESM) FAILED: exported VERSION ${mod.VERSION} != ${want}`);
  process.exit(1);
}
if (mod.default !== mod.ProximaDBClient) {
  console.log("::error::npm smoke (ESM) FAILED: default export is not ProximaDBClient");
  process.exit(1);
}
const client = mod.connectRest(baseUrl);
const health = await client.health();
if (health.status !== "healthy" || health.version !== want) {
  console.log(`::error::npm smoke (ESM) FAILED: health ${JSON.stringify(health)}; expected healthy ${want}`);
  process.exit(1);
}
console.log(JSON.stringify({ artifact: `npm:${PKG}`, entry: "import", steps: ["import", "version", "default_export", "health"], passed: true }));
