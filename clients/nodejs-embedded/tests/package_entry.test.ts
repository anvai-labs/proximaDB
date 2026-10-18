/**
 * Package-entry gate: what `npm publish` ships must contain every file
 * package.json points a consumer at, and both module systems must load it.
 *
 * v0.4.1 shipped `exports["."].import` -> `./dist/index.mjs` while the build
 * wrote its ESM output to `dist/esm/` (with extensionless imports Node cannot
 * resolve), so `import "@anvailabs/proximadb-client"` failed for every ESM
 * consumer and the CommonJS-only tests stayed green. It also exported
 * `VERSION = "1.0.0"`. Found by scripts/release_smoke/npm_esm_smoke.mjs
 * against the published package.
 *
 * Builds the package first (self-contained: `npm test` alone is enough), then
 * asks npm itself for the tarball's file list (`npm pack --dry-run --json`).
 */
import { beforeAll, describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { VERSION } from "../src/index";

const PKG_DIR = path.resolve(__dirname, "..");
const pkg = JSON.parse(fs.readFileSync(path.join(PKG_DIR, "package.json"), "utf8"));

/** Every file path package.json routes a consumer to (exports, main, module, types). */
function entryPaths(): string[] {
  const out = new Set<string>();
  const walk = (v: unknown): void => {
    if (typeof v === "string") out.add(v);
    else if (v && typeof v === "object") Object.values(v).forEach(walk);
  };
  walk(pkg.exports);
  for (const key of ["main", "module", "types", "typings"]) {
    if (typeof pkg[key] === "string") out.add(pkg[key]);
  }
  return [...out].map((p) => p.replace(/^\.\//, ""));
}

let packed: Set<string>;

beforeAll(() => {
  execFileSync("npm", ["run", "build"], { cwd: PKG_DIR, stdio: "pipe" });
  const raw = execFileSync("npm", ["pack", "--dry-run", "--json", "--ignore-scripts"], {
    cwd: PKG_DIR,
    encoding: "utf8",
  });
  const report = JSON.parse(raw);
  packed = new Set(report[0].files.map((f: { path: string }) => f.path));
}, 180_000);

describe("published package entry points", () => {
  it("routes consumers only to files the tarball contains", () => {
    const paths = entryPaths();
    expect(paths).toContain("dist/index.mjs");
    expect(paths).toContain("dist/index.d.mts");
    const missing = paths.filter((p) => !packed.has(p));
    expect(missing, `package.json points at files npm pack does not ship: ${missing}`).toEqual([]);
  });

  it("loads through require() and import() with the same named exports", () => {
    const script = `
      const { createRequire } = await import("node:module");
      const require = createRequire(${JSON.stringify(path.join(PKG_DIR, "package.json"))});
      const cjs = require(${JSON.stringify(path.join(PKG_DIR, pkg.main))});
      const esm = await import(${JSON.stringify("file://" + path.join(PKG_DIR, "dist", "index.mjs"))});
      const names = (o) => Object.keys(o).filter((n) => n !== "default" && n !== "__esModule").sort();
      process.stdout.write(JSON.stringify({ cjs: names(cjs), esm: names(esm),
        sameClass: cjs.ProximaDBClient === esm.ProximaDBClient, defaultOk: esm.default === cjs.default }));
    `;
    const out = JSON.parse(
      execFileSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8" }),
    );
    expect(out.esm).toEqual(out.cjs);
    expect(out.esm).toContain("connectRest");
    // One implementation behind both entries: no dual-package hazard.
    expect(out.sameClass).toBe(true);
    expect(out.defaultOk).toBe(true);
  });

  it("types the ESM entry as the runtime: default import is constructable", () => {
    // A node16 ESM consumer of the BUILT package (dist/index.d.mts via the
    // `import` condition). With CommonJS types on the import path, the
    // default import is typed as the module object and `new Client()` fails.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pkg-esm-types-"));
    try {
      const nm = path.join(dir, "node_modules", "@anvailabs");
      fs.mkdirSync(nm, { recursive: true });
      fs.symlinkSync(PKG_DIR, path.join(nm, "proximadb-client"), "dir");
      fs.writeFileSync(path.join(dir, "package.json"), JSON.stringify({ type: "module" }));
      fs.writeFileSync(
        path.join(dir, "c.mts"),
        'import Client, { connectRest, VERSION, ProximaDBClient } from "@anvailabs/proximadb-client";\n' +
          "const c: ProximaDBClient = new Client();\nconst v: string = VERSION;\n" +
          'export const r = [c, v, connectRest("http://x")];\n',
      );
      fs.writeFileSync(
        path.join(dir, "tsconfig.json"),
        JSON.stringify({
          compilerOptions: { module: "node16", moduleResolution: "node16", strict: true, noEmit: true,
            skipLibCheck: true, types: [] },
          files: ["c.mts"],
        }),
      );
      const tsc = path.join(PKG_DIR, "node_modules", "typescript", "bin", "tsc");
      execFileSync(process.execPath, [tsc, "-p", dir], { encoding: "utf8", stdio: "pipe" });
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);

  it("exports VERSION equal to package.json version", () => {
    expect(VERSION).toBe(pkg.version);
  });
});
