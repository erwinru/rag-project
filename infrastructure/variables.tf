locals {
  // Naming convention for everything in this repo:
  //     <tenant>-<project>-<purpose>-<environment>
  // e.g. erwinru-rag-project-raw-dev
  //
  // Derived rather than hand-written per resource, so a new environment only
  // needs a tfvars file and every name follows automatically. S3 bucket names
  // are globally unique across all of AWS, which is why the tenant leads.
  name_prefix = "${var.tenant}-${var.project}"

  raw_bucket_name = "${local.name_prefix}-raw-${var.environment}"

  tags = merge(
    {
      Repository     = "rag-project"
      Project        = var.project
      Customer       = var.tenant
      Tenant         = var.tenant
      Maintainer     = var.tenant
      Environment    = var.environment
      UseCase        = "ai-factory"
      ProjectPhase   = "none"
      ContactChannel = "ai-factory"
      ManagedBy      = "terraform"
    },
    var.tags,
  )
}

variable "environment" {
  type        = string
  description = "The environment this stack belongs to (dev, stage or prod)."

  validation {
    condition     = contains(["dev", "stage", "prod"], var.environment)
    error_message = "environment must be one of: dev, stage, prod."
  }
}

variable "tenant" {
  type        = string
  description = "The tenant organization that owns this infrastructure. First segment of every resource name."
}

variable "project" {
  type        = string
  description = "Project name. Second segment of every resource name."
}

variable "account_id" {
  type        = string
  description = "AWS account this is applied to. Asserted by the provider before any change."
}

variable "region" {
  type        = string
  description = "AWS region for the resources in this stack."
}

variable "tags" {
  type        = map(string)
  description = "Extra tags merged on top of the defaults."
  default     = {}
}
