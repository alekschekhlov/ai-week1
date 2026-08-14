# ingest.py
import glob
import os
import re

import numpy as np

from chunk import chunk_text            # your recursive splitter (Week 3 Day 5)
from corpus import embed_corpus         # cached batch embedder (Week 3 Day 2)

HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)')
FENCE_RE = re.compile(r'^\s*(```|~~~)')     # fenced code block delimiter

# Static-site-generator noise. Markdown from Hugo/Docsy carries metadata and template
# tags that are not prose: embedding them pollutes the vectors and they leak into
# cited_text. Strip before chunking so the chunker never sees them.
FRONT_MATTER_RE = re.compile(r'^\s*---\n.*?\n---\n', re.DOTALL)   # leading YAML block
SHORTCODE_RE = re.compile(r'\{\{[<%].*?[>%]\}\}', re.DOTALL)      # Hugo {{< … >}} AND {{% … %}}
HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)            # <!-- overview -->, etc.
TITLE_RE = re.compile(r'^title:\s*(.+?)\s*$', re.MULTILINE)       # searched INSIDE front matter only


def _front_matter_title(block: str) -> str | None:
  # Deliberately not a YAML parse: front-matter titles are plain scalars, and pulling in a
  # YAML dependency to read one line is not worth it.
  m = TITLE_RE.search(block)
  return (m.group(1).strip().strip('"\'') or None) if m else None


def load_markdown(root: str):
  """Yield (doc_id, title, text). `title` is the front-matter title, or None."""
  for path in glob.glob(os.path.join(root, "**/*.md"), recursive=True):
    with open(path, encoding="utf-8") as f:
      raw = f.read()

    title = None
    fm = FRONT_MATTER_RE.match(raw)               # leading front-matter block, if any
    if fm:
      title = _front_matter_title(fm.group(0))    # read the title BEFORE discarding it
      raw = raw[fm.end():]
    raw = SHORTCODE_RE.sub('', raw)               # drop Hugo shortcodes, keep their body
    raw = HTML_COMMENT_RE.sub('', raw)            # drop authoring comments
    yield os.path.relpath(path, root), title, raw


def split_by_headers(md: str, title: str | None = None) -> list[tuple[str, str]]:
  """Return (header_path, section_text). header_path = 'H1 > H2' breadcrumb.

  `title` (from front matter) seeds the trail at level 0, so it sits above every heading
  and never gets evicted. Without it, text before the first heading has no breadcrumb at
  all — and Hugo pages usually carry their real name in front matter, not in an H1.
  """
  trail: dict[int, str] = {0: title} if title else {}
  lines_buf: list[str] = []
  cur_path = title or ""
  out: list[tuple[str, str]] = []
  in_fence = False

  def flush():
    text = "\n".join(lines_buf).strip()
    if text:
      out.append((cur_path, text))

  for line in md.splitlines():
    if FENCE_RE.match(line):
      in_fence = not in_fence     # a '# install deps' comment inside ``` is NOT a heading
      lines_buf.append(line)
      continue

    m = None if in_fence else HEADER_RE.match(line)
    if m:
      flush()
      lines_buf.clear()
      level = len(m.group(1))
      trail = {l: h for l, h in trail.items() if l < level}   # drop deeper headings
      trail[level] = m.group(2).strip()
      cur_path = " > ".join(trail[l] for l in sorted(trail))
    else:
      lines_buf.append(line)
  flush()
  return out


def _ensure_indexes(conn):
  # Built AFTER the bulk load: maintaining an HNSW graph row-by-row during insert is
  # far more expensive than building it once over the finished table.
  conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
               "ON chunks USING hnsw (embedding vector_cosine_ops)")   # dense
  conn.execute("CREATE INDEX IF NOT EXISTS chunks_fts_gin "
               "ON chunks USING gin (fts)")                            # lexical (BM25-ish)


def ingest(conn, roots: str | list[str], source: str, max_tokens: int = 512):
  # `source` is the unit of REPLACEMENT (see the DELETE below), the id prefix, and the
  # db.search() metadata filter — so every folder belonging to one source must arrive in
  # ONE call. Pass them together; calling twice with the same source wipes the first run.
  if isinstance(roots, str):
    roots = [roots]

  # 1) files -> sections -> size-bounded chunks, carrying metadata.
  #    chunk_index is the ordinal position within the run, numbered GLOBALLY — not per
  #    document and not per root. Retrieval keys on the integer PK `id`, but the schema's
  #    UNIQUE (source, chunk_index) is per source, so any narrower counter would make
  #    every file's chunk 0 collide and fail the insert.
  records = []   # (source, doc_id, section, chunk_index, content, embed_text)
  idx = 0
  for root in roots:
    # doc_id is a path relative to its OWN root, so two roots can both yield 'README.md'.
    # With several roots, qualify it by folder name to keep provenance unambiguous.
    prefix = f"{os.path.basename(os.path.normpath(root))}/" if len(roots) > 1 else ""
    for doc_id, title, md in load_markdown(root):
      for header_path, section in split_by_headers(md, title=title):
        for piece in chunk_text(section, max_tokens=max_tokens, overlap=max_tokens // 8):
          # Embed the breadcrumb-enriched text so header ancestry shapes the vector, but
          # STORE the clean piece: content is what ends up in cited_text, in the judge's
          # context and in /search output, and a "Workloads > Pods\n\n" prefix is noise there.
          embed_text = f"{header_path}\n\n{piece}" if header_path else piece
          records.append((source, prefix + doc_id, header_path, idx, piece, embed_text))
          idx += 1

  if not records:
    # Guard the DELETE below: a typo'd root must not silently wipe a good corpus.
    print(f"no markdown found under {', '.join(roots)} — nothing ingested")
    return

  # 2) batch-embed ALL chunk texts at once (cache makes re-runs free).
  #    Note this embeds r[5] (enriched), while r[4] (clean) is what gets stored.
  texts = [r[5] for r in records]
  vecs = embed_corpus(texts)["vectors"]

  # 3) replace this source's rows atomically, then bulk insert.
  #    DELETE-then-insert makes re-running idempotent; because every root for this source
  #    was collected above, the wipe-and-replace covers the complete corpus for it.
  # store `content` (clean), look the vector up by `embed_text` (the key embed_corpus used)
  rows = [(s, d, sec, i, content, np.array(vecs[embed_text]))
          for (s, d, sec, i, content, embed_text) in records]
  with conn.transaction():
    conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
    with conn.cursor() as cur:
      cur.executemany(                       # one pipelined batch, not N round-trips
        """INSERT INTO chunks (source, doc_id, section, chunk_index, content, embedding)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        rows,
      )

  _ensure_indexes(conn)
  print(f"ingested {len(records)} chunks from {', '.join(roots)} as source={source!r}")