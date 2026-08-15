"""Text embedding via Amazon Bedrock (Titan Text Embeddings V2)."""

from __future__ import annotations

import json

import boto3
from botocore.config import Config as BotoConfig

from rag.config import config

client = boto3.client(
    "bedrock-runtime",
    region_name=config.bedrock.region,
    config=BotoConfig(
        retries={
            "max_attempts": config.bedrock.max_attempts,
            "mode": config.bedrock.retry_mode,
        }
    ),
)


def embed_text(text: str) -> list[float]:
    response = client.invoke_model(
        modelId=config.embedding.model_id,
        body=json.dumps(
            {
                "inputText": text,
                "dimensions": config.embedding.dimensions,
                "normalize": True,
            }
        ),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read())
    return payload["embedding"]
