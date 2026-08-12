-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create the chunks table
create table chunks (
  id bigint primary key generated always as identity,
  content text not null,
  embedding vector(1536),
  metadata jsonb
);

-- Create a function to perform hybrid search (pgvector + BM25)
create or replace function hybrid_search (
  query_text text,
  query_embedding vector(1536),
  match_count int
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  -- Simple placeholder logic for the scaffolding
  select
    chunks.id,
    chunks.content,
    chunks.metadata,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  order by chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
