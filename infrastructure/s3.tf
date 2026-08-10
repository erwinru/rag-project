// Landing zone for raw scraped source documents: the HTML cache and the
// extracted article text the scraper produces under data/. Everything here is
// re-derivable by re-running the scraper, but re-scraping costs requests
// against ml6.eu, so the contents are worth protecting.
resource "aws_s3_bucket" "raw" {
  bucket = local.raw_bucket_name

  // force_destroy stays at its default of false: `terraform destroy` fails
  // rather than silently emptying a bucket full of scraped data. Set it to
  // true deliberately and temporarily if you ever really mean to wipe it.
}

// A re-scrape overwrites objects in place. Versioning means a bad extraction
// run can be rolled back instead of having lost the previous good copy.
resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "raw" {
  bucket = aws_s3_bucket.raw.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

// ACLs are legacy. BucketOwnerEnforced disables them, leaving IAM and the
// bucket policy as the only access path.
resource "aws_s3_bucket_ownership_controls" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    // Long enough to notice and undo a bad scrape, short enough that old HTML
    // does not accumulate indefinitely.
    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  // No storage-class transitions yet: the corpus is small and gets read on
  // every re-index, so IA/Glacier would add retrieval cost for no saving.
  depends_on = [aws_s3_bucket_versioning.raw]
}

resource "aws_s3_bucket_policy" "raw" {
  bucket = aws_s3_bucket.raw.id
  policy = data.aws_iam_policy_document.raw.json

  depends_on = [aws_s3_bucket_public_access_block.raw]
}

// Deny-only. Access is granted through IAM on the principals that read and
// write the bucket, not by re-granting to the account root here.
data "aws_iam_policy_document" "raw" {
  statement {
    sid    = "DenyUnencryptedTransport"
    effect = "Deny"

    principals {
      type        = "AWS"
      identifiers = ["*"]
    }

    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.raw.arn,
      "${aws_s3_bucket.raw.arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}
