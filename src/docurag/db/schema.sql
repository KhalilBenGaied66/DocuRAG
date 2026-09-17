-- DocuRAG schema. ${dim} and ${ts_config} are rendered from validated settings (see db/pool.py).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_info (
  key   text PRIMARY KEY,
  value text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id              uuid PRIMARY KEY,
  source_uri      text NOT NULL UNIQUE,
  doc_key         text NOT NULL,                 -- logical document, stable across versions
  version         text,
  title           text NOT NULL,
  doc_type        text NOT NULL CHECK (doc_type IN ('pdf', 'kb', 'record')),
  language        text NOT NULL DEFAULT 'fr',
  acl_groups      text[] NOT NULL CHECK (cardinality(acl_groups) > 0),
  content_hash    text NOT NULL,
  metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_current      boolean NOT NULL DEFAULT true,
  embedding_model text NOT NULL,
  report          jsonb,
  ingested_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_doc_key_idx ON documents (doc_key);

CREATE TABLE IF NOT EXISTS sections (
  id          bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord         int  NOT NULL,
  path        text NOT NULL,
  content     text NOT NULL,
  tokens      int  NOT NULL,
  page_start  int,
  page_end    int
);
CREATE INDEX IF NOT EXISTS sections_document_idx ON sections (document_id);

CREATE TABLE IF NOT EXISTS chunks (
  id              bigserial PRIMARY KEY,
  document_id     uuid   NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section_id      bigint NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
  ord             int    NOT NULL,               -- position inside the section
  kind            text   NOT NULL CHECK (kind IN ('text', 'table')),
  section_path    text   NOT NULL,
  page_start      int,
  page_end        int,
  body            text   NOT NULL,               -- what the LLM and the user see
  content         text   NOT NULL,               -- body + contextual header: embedded and indexed
  content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('${ts_config}'::regconfig, content)) STORED,
  embedding       halfvec(${dim}) NOT NULL,
  embedding_model text   NOT NULL,
  acl_groups      text[] NOT NULL,               -- denormalised: filter without joining documents
  is_current      boolean NOT NULL DEFAULT true, -- denormalised from documents
  content_hash    text   NOT NULL,
  tokens          int    NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS chunks_acl_idx ON chunks USING gin (acl_groups);
CREATE INDEX IF NOT EXISTS chunks_section_idx ON chunks (section_id, ord);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- Re-ingesting unchanged text costs nothing: embeddings are cached by content hash and model.
CREATE TABLE IF NOT EXISTS embedding_cache (
  content_hash text NOT NULL,
  model        text NOT NULL,
  embedding    halfvec(${dim}) NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (content_hash, model)
);

-- A Postgres-backed job queue (FOR UPDATE SKIP LOCKED): no extra broker to run.
CREATE TABLE IF NOT EXISTS ingestion_jobs (
  id         uuid PRIMARY KEY,
  status     text NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
  spec       jsonb NOT NULL,
  attempts   int  NOT NULL DEFAULT 0,
  result     jsonb,
  error      text,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingestion_jobs_queued_idx ON ingestion_jobs (created_at)
  WHERE status = 'queued';

-- One row per answered question: quality signals, tokens and latency per stage.
CREATE TABLE IF NOT EXISTS query_log (
  id                bigserial PRIMARY KEY,
  request_id        text NOT NULL,
  user_id           text NOT NULL,
  question          text,                        -- stored only when DOCURAG_LOG_QUESTIONS=true
  question_hash     text NOT NULL,
  abstained         boolean NOT NULL,
  top_score         double precision,
  n_citations       int NOT NULL,
  invalid_citations int NOT NULL,
  prompt_version    text NOT NULL,
  chat_model        text NOT NULL,
  input_tokens      int NOT NULL,
  output_tokens     int NOT NULL,
  cached_tokens     int NOT NULL,
  timings_ms        jsonb NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS query_log_created_idx ON query_log (created_at);
