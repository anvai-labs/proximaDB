#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Down-convert an OpenAPI 3.1 document to 3.0.3 for oapi-codegen (TD-126 Phase 2).

The published ProximaDB spec (docs/openapi/proximadb-openapi.yaml) is generated
from the utoipa-annotated axum handlers as OpenAPI 3.1.0 and is drift-gated by
Phase 1. The Go generator (oapi-codegen v2.4.x / kin-openapi 0.127) does not yet
support OpenAPI 3.1 (https://github.com/oapi-codegen/oapi-codegen/issues/373).

Rather than hand-editing the source spec (forbidden — it is generated + gated),
`make gen-go-sdk` runs this transform on a *temporary copy* immediately before
codegen. The transform is intentionally minimal and mechanical:

  * `openapi: 3.1.x`              -> `openapi: 3.0.3`
  * `type: [T, "null"]`          -> `type: T` + `nullable: true`
  * `type: ["null", T]`          -> `type: T` + `nullable: true`
  * `type: [T]`                  -> `type: T`
  * drop 3.1-only keywords that 3.0 validators reject (`const`, `examples` at
    schema scope is left alone — kin-openapi tolerates it).

This is a build-time adapter, not a spec edit: the source spec is untouched, and
the generated Go client's drift gate regenerates through this same transform, so
the generated artifact stays pinned to the published spec.
"""

import re
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - environment guard
    sys.stderr.write(
        "PyYAML is required (use the repo venv: "
        "/Users/vijaysingh/code/.venv/bin/python)\n"
    )
    sys.exit(2)


class _Yaml12CoreBoolLoader(yaml.SafeLoader):
    """SafeLoader restricted to YAML 1.2 core-schema booleans.

    PyYAML resolves scalars per YAML 1.1, where bare ``Off``/``On``/``Yes``/
    ``No`` (and ``y``/``n``) are BOOLEANS. The published spec is emitted by
    serde_yaml, which follows YAML 1.2 — there those are plain strings — so
    an enum value of ``Off`` is written unquoted and a plain ``safe_load``
    round-trips it as ``False``. That silently corrupts the down-converted
    spec: oapi-codegen emitted ``False AbacGrantEnforcement = "false"`` and
    progenitor hard-errored ("type error unexpected value type"). Narrow the
    implicit bool resolver to the YAML 1.2 core set so ``Off`` stays the
    string "Off" while ``true``/``false`` keep parsing as booleans.
    """

    pass


_Yaml12CoreBoolLoader.yaml_implicit_resolvers = {
    first_char: [
        (
            tag,
            re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
            if tag == "tag:yaml.org,2002:bool"
            else regexp,
        )
        for tag, regexp in resolvers
    ]
    for first_char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _normalize_type(node: dict) -> None:
    """Rewrite a 3.1 union `type` list into a 3.0 scalar type + nullable."""
    t = node.get("type")
    if not isinstance(t, list):
        return
    non_null = [x for x in t if x != "null"]
    has_null = any(x == "null" for x in t)
    if len(non_null) == 1:
        node["type"] = non_null[0]
    elif len(non_null) == 0:
        # `type: ["null"]` — degenerate; drop the constraint entirely.
        node.pop("type", None)
    else:
        # Multiple non-null types: 3.0 cannot express a union `type` list.
        # Fall back to the first concrete type so codegen can proceed; this
        # only affects fields the SDK treats as open `interface{}` anyway.
        node["type"] = non_null[0]
    if has_null:
        node["nullable"] = True


def _is_null_member(m) -> bool:
    """True for a 3.1 `null`-only union member, e.g. `{type: "null"}`."""
    return isinstance(m, dict) and m.get("type") == "null" and set(m) <= {"type"}


def _collapse_nullable_combinator(node: dict, key: str) -> None:
    """Collapse `oneOf`/`anyOf: [{type: null}, X]` (3.1 optional-$ref idiom) into 3.0.

    Removes the `null`-only member and marks the schema nullable. If a single
    member remains, it is inlined into the node (lifting a bare `$ref`, which 3.0
    cannot combine with sibling keywords, into a wrapping `allOf`).
    """
    members = node.get(key)
    if not isinstance(members, list):
        return
    non_null = [m for m in members if not _is_null_member(m)]
    if len(non_null) == len(members):
        return  # no null member; leave combinator alone
    node["nullable"] = True
    if len(non_null) == 1:
        node.pop(key)
        only = non_null[0]
        if isinstance(only, dict) and "$ref" in only and len(only) == 1:
            # `$ref` cannot have siblings in 3.0; wrap it in allOf.
            node["allOf"] = [{"$ref": only["$ref"]}]
        elif isinstance(only, dict):
            for k, v in only.items():
                node.setdefault(k, v)
        else:
            node[key] = [only]
    else:
        node[key] = non_null


def _walk(node):
    if isinstance(node, dict):
        for combinator in ("oneOf", "anyOf"):
            if combinator in node:
                _collapse_nullable_combinator(node, combinator)
        if "type" in node:
            _normalize_type(node)
        # 3.1 `const` -> 3.0 single-value enum.
        if "const" in node and "enum" not in node:
            node["enum"] = [node.pop("const")]
        for value in node.values():
            _walk(value)
    elif isinstance(node, list):
        for item in node:
            _walk(item)


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(
            "usage: openapi_31_to_30.py <input-3.1.yaml> <output-3.0.yaml>\n"
        )
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_Yaml12CoreBoolLoader)

    version = doc.get("openapi", "")
    if isinstance(version, str) and version.startswith("3.1"):
        doc["openapi"] = "3.0.3"

    _walk(doc)

    with open(dst, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=True, default_flow_style=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
