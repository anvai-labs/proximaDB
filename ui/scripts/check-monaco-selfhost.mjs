/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Guard for TD-UICDN-1: the admin UI's Monaco runtime must resolve from our own
// origin, never the CDN default baked into @monaco-editor/loader. Checks the
// three links of that chain, in order:
//
//   1. `monaco-editor` is a real, exact-pinned dependency (so `npm audit`
//      audits the version the browser executes — the audit blind spot the TD
//      documents);
//   2. the self-hosted runtime is present in `public/vs` (copy-monaco-vs.js
//      ran, and its stamp matches the installed package);
//   3. the app still overrides the loader (`paths: { vs: '/vs' }` in
//      SqlQueryTab.tsx), and applying that override through the SHIPPED
//      loader's own config-merge logic resolves the editor script to
//      `/vs/loader.js` — proving the CDN default is unreachable at runtime.
//
// Runs as part of `prebuild`, so a build cannot ship without passing it.

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const uiRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const require = createRequire(pathToFileURL(path.join(uiRoot, 'package.json')));

function fail(message) {
  console.error(`[check-monaco-selfhost] FAIL: ${message}`);
  process.exit(1);
}

// 1. Real dependency, exact pin (a floating range could desync public/vs from
//    what the loader executes). Read the installed manifest straight from the
//    tree the copy script copies from — monaco-editor 0.57's exports map does
//    not expose ./package.json (it remaps ./* into the esm tree), so a bare
//    require('monaco-editor/package.json') fails.
const pkg = JSON.parse(fs.readFileSync(path.join(uiRoot, 'package.json'), 'utf8'));
const installedPkgPath = path.join(uiRoot, 'node_modules', 'monaco-editor', 'package.json');
if (!fs.existsSync(installedPkgPath)) {
  fail('monaco-editor is not installed under ui/node_modules — run npm install first');
}
const installedVersion = JSON.parse(fs.readFileSync(installedPkgPath, 'utf8')).version;
if (!pkg.dependencies?.['monaco-editor']) {
  fail('monaco-editor is not in ui/package.json dependencies');
}
if (pkg.dependencies['monaco-editor'] !== installedVersion) {
  fail(`monaco-editor pin ${pkg.dependencies['monaco-editor']} != installed ${installedVersion}`);
}

// 2. Self-hosted runtime present and version-stamped.
for (const f of ['loader.js', 'editor/editor.main.js']) {
  if (!fs.existsSync(path.join(uiRoot, 'public', 'vs', f))) {
    fail(`public/vs/${f} missing — run scripts/copy-monaco-vs.js (npm install) first`);
  }
}
const stampPath = path.join(uiRoot, 'public', '.monaco-vs-version');
if (!fs.existsSync(stampPath) || fs.readFileSync(stampPath, 'utf8') !== installedVersion) {
  fail(`public/.monaco-vs-version stamp missing or stale for ${installedVersion}`);
}

// 3. The app's override, straight from the call site, merged through the
//    shipped loader's own logic (`state.config = merge(state.config, config)`,
//    loader/index.js:33; script URL = `state.config.paths.vs + '/loader.js'`,
//    loader/index.js:104).
// Match an actual CALL at statement position, not a substring: a commented-out
// `// loader.config(...)` must not satisfy the check (review finding — the
// substring form was provably bypassable by commenting the line out).
const selfhostCall =
  /^\s*loader\.config\(\{\s*paths:\s*\{\s*vs:\s*'(\/vs)'\s*}\s*}\s*\)\s*;/m;
const tsx = fs.readFileSync(
  path.join(uiRoot, 'src', 'components', 'SqlQueryTab.tsx'),
  'utf8',
);
if (!selfhostCall.test(tsx)) {
  fail(
    'SqlQueryTab.tsx no longer calls `loader.config({ paths: { vs: \'/vs\' } })` ' +
      'as a statement (a commented-out override does not count)',
  );
}
const loaderLib = path.join(uiRoot, 'node_modules', '@monaco-editor/loader', 'lib', 'es');
const { default: deepMerge } = await import(
  pathToFileURL(path.join(loaderLib, 'utils', 'deepMerge.js')).href
);
const { default: defaultConfig } = await import(
  pathToFileURL(path.join(loaderLib, 'config', 'index.js')).href
);
const merged = deepMerge(defaultConfig, { paths: { vs: '/vs' } });
const loadUrl = `${merged.paths.vs}/loader.js`;
if (merged.paths.vs.includes('://') || merged.paths.vs.includes('cdn.')) {
  fail(`resolved editor URL is remote: ${loadUrl}`);
}

console.log(
  `[check-monaco-selfhost] OK: monaco-editor ${installedVersion} self-hosted; ` +
    `loader resolves ${loadUrl}`,
);
