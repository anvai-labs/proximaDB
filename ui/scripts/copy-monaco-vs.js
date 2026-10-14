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

// Self-host the Monaco editor runtime (TD-UICDN-1): copy
// `node_modules/monaco-editor/min/vs` into `public/vs` so `@monaco-editor/react`'s
// AMD loader resolves the editor from our own origin instead of the jsdelivr
// CDN. The admin UI must have zero third-party script origins (content-blocker
// and air-gapped deployments).
//
// Runs on `postinstall` and `prebuild` (npm lifecycle hooks). Idempotent: the
// copy is skipped when the installed monaco version already matches the stamp
// written next to the copied tree; a version bump or a wiped `public/vs`
// triggers a fresh copy.

const fs = require('fs');
const path = require('path');

const uiRoot = path.resolve(__dirname, '..');
const src = path.join(uiRoot, 'node_modules', 'monaco-editor', 'min', 'vs');
const dest = path.join(uiRoot, 'public', 'vs');
const stamp = path.join(uiRoot, 'public', '.monaco-vs-version');

function readInstalledVersion() {
  const pkg = path.join(uiRoot, 'node_modules', 'monaco-editor', 'package.json');
  return JSON.parse(fs.readFileSync(pkg, 'utf8')).version;
}

function copyDir(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const fromPath = path.join(from, entry.name);
    const toPath = path.join(to, entry.name);
    if (entry.isDirectory()) {
      copyDir(fromPath, toPath);
    } else {
      fs.copyFileSync(fromPath, toPath);
    }
  }
}

if (!fs.existsSync(path.join(src, 'loader.js'))) {
  console.error(
    '[copy-monaco-vs] node_modules/monaco-editor/min/vs/loader.js not found — ' +
      'run `npm install` in ui/ first (monaco-editor is a direct dependency).',
  );
  process.exit(1);
}

const version = readInstalledVersion();
if (fs.existsSync(dest) && fs.existsSync(stamp) && fs.readFileSync(stamp, 'utf8') === version) {
  process.exit(0);
}

fs.rmSync(dest, { recursive: true, force: true });
copyDir(src, dest);
fs.writeFileSync(stamp, version);
console.log(`[copy-monaco-vs] self-hosted monaco-editor ${version} -> public/vs`);
