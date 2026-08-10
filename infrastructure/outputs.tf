output "raw_bucket_name" {
  description = "Name of the raw document bucket. Feed this to the scraper as its upload target."
  value       = aws_s3_bucket.raw.bucket
}

output "raw_bucket_arn" {
  description = "ARN of the raw document bucket, for IAM policies that grant access to it."
  value       = aws_s3_bucket.raw.arn
}
