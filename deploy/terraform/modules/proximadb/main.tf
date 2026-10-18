# ProximaDB Terraform Module
# Reusable module for deploying ProximaDB on any cloud platform

terraform {
  required_version = ">= 1.0"
}

variable "name" {
  description = "Name prefix for resources"
  type        = string
  default     = "proximadb"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "instance_type" {
  description = "Instance type/size"
  type        = string
}

variable "storage_size_gb" {
  description = "Storage size in GB"
  type        = number
  default     = 100
}

variable "storage_type" {
  description = "Storage type (ssd, nvme, standard)"
  type        = string
  default     = "ssd"
}

variable "replica_count" {
  description = "Number of replicas"
  type        = number
  default     = 1
}

variable "enable_monitoring" {
  description = "Enable monitoring and metrics"
  type        = bool
  default     = true
}

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 7
}

variable "container_image" {
  description = "ProximaDB container image"
  type        = string
  default     = "vjsingh1984/proximadb:latest"
}

variable "rest_port" {
  description = "REST API port"
  type        = number
  default     = 5678
}

variable "grpc_port" {
  description = "gRPC API port"
  type        = number
  default     = 5679
}

variable "arrow_ipc_port" {
  description = "Arrow IPC port"
  type        = number
  default     = 5680
}

variable "metrics_port" {
  description = "Metrics port"
  type        = number
  default     = 9090
}

variable "storage_engine" {
  description = "Not read by the server; the engine is chosen per collection (TD-ENGSEL-1)"
  type        = string
  default     = "sst"
}

variable "log_level" {
  description = "Logging level (trace, debug, info, warn, error)"
  type        = string
  default     = "info"
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}

# Common labels for all resources
locals {
  common_labels = merge(
    {
      app         = "proximadb"
      environment = var.environment
      managed_by  = "terraform"
    },
    var.tags
  )

  # Container environment variables
  container_env = {
    RUST_LOG                     = var.log_level
    PROXIMADB_BIND_ADDRESS       = "0.0.0.0"
    PROXIMADB_REST_PORT          = tostring(var.rest_port)
    PROXIMADB_GRPC_PORT          = tostring(var.grpc_port)
    PROXIMADB_ARROW_IPC_PORT     = tostring(var.arrow_ipc_port)
    PROXIMADB_METRICS_PORT       = tostring(var.metrics_port)
    PROXIMADB_DATA_DIR           = "/data/proximadb"
    # Not read by the server (only PROXIMADB_BIND_ADDRESS is, via entrypoint.sh); kept for
    # compatibility. See TD-ENGSEL-1 and TD-PKG-3.
    PROXIMADB_DEFAULT_ENGINE     = var.storage_engine
    PROXIMADB_WAL_ENABLED        = "true"
  }
}

output "labels" {
  description = "Common labels for resources"
  value       = local.common_labels
}

output "container_env" {
  description = "Container environment variables"
  value       = local.container_env
}

output "ports" {
  description = "ProximaDB ports configuration"
  value = {
    rest       = var.rest_port
    grpc       = var.grpc_port
    arrow_ipc  = var.arrow_ipc_port
    metrics    = var.metrics_port
  }
}
