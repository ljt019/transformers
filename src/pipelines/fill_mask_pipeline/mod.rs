//! Fill-mask pipeline for predicting masked tokens in text.
//!
//! This module provides functionality for filling in masked tokens (typically `[MASK]`) 
//! in text sequences using pre-trained language models. It's useful for text completion,
//! error correction, and exploring model behavior.
//!
//! ## Main Types
//!
//! - [`FillMaskPipeline`] - High-level interface for mask filling
//! - [`FillMaskPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`FillMaskModel`] - Trait for fill-mask model implementations
//! - [`ModernBertSize`] - Available model size options
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use transformers::pipelines::fill_mask_pipeline::*;
//!
//! # tokio_test::block_on(async {
//! // Create a fill-mask pipeline
//! let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
//!     .build()
//!     .await?;
//!
//! // Fill masked tokens
//! let predictions = pipeline.predict("The capital of France is [MASK].").await?;
//! for prediction in predictions {
//!     println!("Token: {} (score: {:.3})", prediction.token, prediction.score);
//! }
//! # anyhow::Ok(())
//! # });
//! ```

pub mod builder;
pub mod fill_mask_model;
pub mod fill_mask_pipeline;

pub use builder::FillMaskPipelineBuilder;
pub use fill_mask_model::FillMaskModel;
pub use fill_mask_pipeline::FillMaskPipeline;

pub use crate::models::ModernBertSize;

pub use anyhow::Result;
