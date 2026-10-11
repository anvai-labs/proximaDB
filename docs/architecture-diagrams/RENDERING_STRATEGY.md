This document defines how the ProximaDB architecture atlas is rendered so it is **natively
visible on GitHub** with zero tooling, with PNG/SVG export for non-GitHub publishing.

> **Mermaid is the only source format.** All diagrams are `.mermaid` (or embedded in `.md`).
> There are **no `.puml` files** in the repository — `.gitignore` excludes `*.puml`, so PlantUML
> is not used. This keeps a single source of truth and guarantees every diagram renders on GitHub.

| Format | GitHub render | Standalone file | Best for |
|---|---|---|---|
| **Mermaid** `.mermaid` / embedded in `.md` | native (in `.md`) / via viewer | raw text | **PRIMARY — all diagrams** |
| **PNG / SVG** (exported from Mermaid) | via image embeds (`![](...)`) | yes | Embedding in non-GitHub docs, PDFs, slides |
| **ASCII art** | native (code block) | yes | Simple ideas, inline in prose, email-safe |

1. **Mermaid-first and Mermaid-only.** Every diagram is Mermaid. `atlas.md` embeds the key
   diagrams so the whole system renders on GitHub with zero install; the per-directory `.mermaid`
   files are the authoritative full-detail source (cataloged in `README.md`).
2. **Export to PNG/SVG** with the Mermaid CLI (`mmdc`) for slides/PDFs; the public Kroki service
   is a fallback when `mmdc` is not installed.
3. **ASCII art** (`ASCII_ART.md`) for the simplest one-box/one-liner ideas — native everywhere,
   no tooling, copy-paste-safe.

**Decision tree:**
1. Can a reader see it on GitHub with **zero install**? → Mermaid embedded in `atlas.md` / `README.md`.
2. Need a PNG/SVG for a slide or PDF? → export from the `.mermaid` with `render_atlas.sh`.
3. A one-off simple idea? → ASCII art inline.

```bash
bash scripts/diagrams/render_atlas.sh            # export every .mermaid to PNG+SVG
bash scripts/diagrams/render_atlas.sh --kroki    # emit Kroki-rendered .md sidecars instead
```

See `atlas.md` for the **all-Mermaid, natively-rendered single-page view** of the system.
