# Troubleshooting

## Bedrock Titan Embeddings V2 -- `ThrottlingException` (unresolved)

**Symptom:** `embed_text()` ([`src/rag/embedding/embedding.py`](../src/rag/embedding/embedding.py))
raises `ThrottlingException: reached max retries: 10` on `invoke_model`, even
for a single, isolated call (not just under the bulk `rag-embed` ingestion
loop). Region is `eu-central-1` ([`config.toml`](../config.toml)).

**Investigation so far:**

1. Retry config was already `adaptive`/`max_attempts=10` going in (see the
   comment above `[bedrock]` in `config.toml`) -- that wasn't the gap.
2. Checked Service Quotas (console: Service Quotas -> Amazon Bedrock, region
   `eu-central-1`). Found the actual account-level applied quota for
   Titan Text Embeddings V2 on-demand inference was **0**, against an AWS
   default of 6,000 requests/min and 300,000 tokens/min. Both rows were
   marked "Not adjustable" (can't fix via a normal quota-increase request).
3. Ruled out the request shape -- `dimensions`/`normalize` are valid,
   documented optional Titan V2 fields; a plain AWS SDK example (`inputText`
   only, `us-east-1`) has the same problem when pointed at `eu-central-1`.
4. Ruled out model availability -- `aws bedrock list-foundation-models
   --region eu-central-1` shows `amazon.titan-embed-text-v2:0` as
   `lifecycle.status: ACTIVE`. This turned out to be a red herring:
   that status is catalog-level (the model exists in the region), not
   account-specific -- it doesn't reflect whether *this account* has been
   granted on-demand access there.
5. Ruled out authentication -- `ThrottlingException` only happens after a
   request has already passed auth/authz. An auth problem would surface as
   `UnrecognizedClientException`, `ExpiredTokenException`, or
   `AccessDeniedException` instead. The Bedrock console's "Get started by
   using API Keys" banner (a bearer-token auth alternative to IAM/SigV4) is
   unrelated for the same reason -- boto3 is already authenticating fine.

**Leading hypothesis (not yet confirmed):** Bedrock model access is granted
per region, not account-wide. Titan Embeddings V2 access may have only ever
been enabled in `us-east-1` (Bedrock's most common default/getting-started
region), never explicitly requested for `eu-central-1` -- which would
produce exactly this pattern (applied quota 0, default nonzero, not
adjustable via Service Quotas because access itself, not the quota value,
is the gap).

**Status:** unresolved. Next diagnostic step was a same-code, two-region
comparison (`us-east-1` vs `eu-central-1`) to confirm the region-access
hypothesis before doing anything in the Bedrock console's Model access page
-- not yet run to completion / access not yet successfully granted in
`eu-central-1`.

**Workaround adopted:** embedding is now provider-configurable
(`config.embedding.provider`), with a local Hugging Face
`sentence-transformers` model as an alternative to Bedrock/Titan --
sidesteps Bedrock region/quota/access management for this step entirely.
Bedrock remains available and selectable once/if access is sorted out. See
[`Embedding.md`](Embedding.md).
