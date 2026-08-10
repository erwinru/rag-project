// Partial configuration -- bucket, key, region and locking come from
// environments/<env>/backend.tfvars via -backend-config at init time.
//
// The state bucket already exists (managed by the initial-setup-infrastructure
// repo), so unlike that repo there is no local-state bootstrap here: this can
// point at S3 from the very first init.
terraform {
  backend "s3" {}
}
