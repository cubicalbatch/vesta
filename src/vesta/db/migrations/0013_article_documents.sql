-- Migration 0013 — per-entry document manifest for nautiluszim document-library ZIMs.
--
-- A ``kind='documents'`` archive (openZIM ``nautiluszim``) is a single-page-app
-- document library: one ``text/html`` viewer shell renders binary documents
-- (PDFs) client-side from a ``database.js`` manifest. The real content is the
-- PDFs (under ``files/``), not the viewer scaffolding. This table is the
-- resolved catalog from a ZIM entry path (the PDF) to its manifest metadata
-- (title / description / author / mimetype), so the frontend can render a
-- document library without touching ``database.js`` and the indexer can index
-- the PDFs by their manifest title.
--
-- Populated by ``zim/documents.py`` at archive registration (gated on
-- ``zims.kind = 'documents'``), parsed from ``database.js`` (never
-- scraper-specific — detection keys off the manifest's ``var DATABASE = [...]``
-- shape with ``ti``/``fp`` records). Keyed by ``(zim_id, doc_path)`` — the PDF
-- entry path — so a catalog lookup is one indexed equality read. FK-cascades
-- with the zims row; independent of the semantic index (a re-index never
-- touches it).

CREATE TABLE article_documents (
    zim_id      INTEGER NOT NULL REFERENCES zims(id) ON DELETE CASCADE,
    doc_path    TEXT    NOT NULL,   -- the ZIM entry path of the PDF, e.g. files/Water (1).pdf
    title       TEXT,                -- manifest 'ti' (manifest title wins over libzim title)
    description TEXT,                -- manifest 'dsc'
    author      TEXT,                -- manifest 'aut'
    doc_mime    TEXT,                -- the entry's true libzim mimetype, e.g. application/pdf
    UNIQUE(zim_id, doc_path)
);

CREATE INDEX article_documents_zim ON article_documents(zim_id);
