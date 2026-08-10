terraform {
  // >= 1.11 for `use_lockfile` in the S3 backend. The exact version is pinned
  // in ../.terraform-version.
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}
