environment = "dev"
tenant      = "erwinru"
project     = "rag-project"
account_id  = "477013660852"
region      = "eu-central-1"

// Bucket names are derived in variables.tf from the three values above:
//     <tenant>-<project>-raw-<environment>  ->  erwinru-rag-project-raw-dev
