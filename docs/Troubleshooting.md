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

## Bedrock Claude Haiku (Converse) -- `AccessDeniedException` (unresolved, different bug)

**Symptom:** calling `ChatBedrockConverse`/`invoke_model`-via-Converse
against `config.generation.model_id`
(`eu.anthropic.claude-haiku-4-5-20251001-v1:0`) fails immediately with:

```
AccessDeniedException: User: .../erwin.rudi is not authorized to perform:
bedrock:InvokeModel on resource:
arn:aws:bedrock:eu-central-1:523397564374:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0
with an explicit deny in an identity-based policy:
arn:aws:iam::523397564374:policy/ManageOwnCredentials
```

Hit while wiring up the RAGAS synthetic-QA generator
([`Evaluation.md`](Evaluation.md)), which uses this model as the generator
LLM -- but this blocks *any* Claude Haiku generation call in the project,
including the still-missing `rag.retrieval.generation` module.

**Root cause (confirmed, not a hypothesis this time):** an **explicit deny**
statement in an IAM identity-based policy named `ManageOwnCredentials`,
attached to the `erwin.rudi` IAM user, covers `bedrock:InvokeModel` on this
specific inference-profile resource. In IAM's evaluation logic an explicit
deny always wins over any allow, anywhere else in the account (SCPs,
resource policies, other identity policies) -- so no amount of Model
access/quota configuration in the Bedrock console fixes this, unlike the
Titan issue above.

**Scope, confirmed by contrast with the Titan bug above:** this deny is
specific to this resource (or at least to Bedrock generation calls), not a
blanket `bedrock:InvokeModel` deny for the account -- Titan embed calls
against a *different* Bedrock resource path fail with `ThrottlingException`
(passed authz, failed on quota), not `AccessDeniedException`. If the deny
were account-wide, the Titan calls would have failed the same way.

**Status:** resolved -- the `ManageOwnCredentials` deny was fixed on the AWS
side. Confirmed by the error changing to a different exception type
(`ResourceNotFoundException`, see below) on the next attempt rather than
the same `AccessDeniedException`.

## Bedrock Claude Haiku (Converse) -- `ResourceNotFoundException`, Anthropic use-case form (in progress)

**Symptom:** once the IAM deny above was fixed, the same call now fails
with:

```
ResourceNotFoundException: Model use case details have not been submitted
for this account. Fill out the Anthropic use case details form before
using the model. If you have already filled out the form, try again in 15
minutes.
```

**Root cause:** a separate, one-time AWS requirement specific to Anthropic
models on Bedrock -- an account must submit a short "use case details"
form (Anthropic's own intake, surfaced through AWS) before any Claude model
can be invoked, independent of IAM permissions and Model access/quota
config. Not a bug in this project or a Bedrock quota issue -- every account
needs to do this once, the first time it uses a Claude model on Bedrock.

**Fix:** Bedrock console -> Model access (region `eu-central-1`) -> submit
the use-case-details form linked next to the Anthropic/Claude Haiku entry,
then wait ~15 minutes before retrying.

**Status:** in progress -- form submission not yet confirmed successful
(retry pending the ~15 minute wait).
