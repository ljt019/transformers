//! Text embedding pipeline for generating dense vector representations of text.
//!
//! This module provides functionality for converting text into high-dimensional
//! vectors that capture semantic meaning, useful for similarity search, clustering,
//! and other downstream tasks.
//!
//! ## Main Types
//!
//! - [`EmbeddingPipeline`] - High-level interface for text embedding
//! - [`EmbeddingPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`EmbeddingModel`] - Trait for embedding model implementations
//! - [`Qwen3EmbeddingModel`] - Qwen3-based embedding model implementation
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use transformers::pipelines::embedding_pipeline::*;
//!
//! # tokio_test::block_on(async {
//! // Create an embedding pipeline
//! let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
//!     .build()
//!     .await?;
//!
//! // Generate embeddings
//! let embeddings = pipeline.embed(&["Hello world", "How are you?"]).await?;
//! println!("Generated {} embeddings", embeddings.len());
//! # anyhow::Ok(())
//! # });
//! ```

pub mod builder;
pub mod model;
pub mod pipeline;

pub use builder::EmbeddingPipelineBuilder;
pub use model::EmbeddingModel;
pub use pipeline::EmbeddingPipeline;

pub use crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingModel;
pub use crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingSize;

pub use anyhow::Result;
