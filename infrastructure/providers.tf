provider "aws" {
  region = var.region

  // Guardrail: if the shell's credentials point at a different account,
  // Terraform refuses to plan rather than building in the wrong place.
  allowed_account_ids = [var.account_id]

  default_tags {
    tags = local.tags
  }
}
