//! Zero-shot classification pipeline for classifying text without training data.
//!
//! This module provides functionality for classifying text into arbitrary categories
//! without requiring training examples for those categories. It uses natural language
//! inference to determine if a text belongs to a given category, making it very flexible
//! for dynamic classification tasks.
//!
//! ## Main Types
//!
//! - [`ZeroShotClassificationPipeline`] - High-level interface for zero-shot classification
//! - [`ZeroShotClassificationPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`ZeroShotClassificationModel`] - Trait for zero-shot classification model implementations
//! - [`ModernBertSize`] - Available model size options
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use transformers::pipelines::zero_shot_classification_pipeline::*;
//!
//! # tokio_test::block_on(async {
//! // Create a zero-shot classification pipeline
//! let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
//!     .build()
//!     .await?;
//!
//! // Classify text into arbitrary categories
//! let text = "The movie was absolutely fantastic!";
//! let labels = vec!["positive", "negative", "neutral"];
//! let results = pipeline.classify(text, &labels).await?;
//!
//! for result in results {
//!     println!("Label: {} (confidence: {:.2})", result.label, result.score);
//! }
//! # anyhow::Ok(())
//! # });
//! ```

pub mod builder;
pub mod model;
pub mod pipeline;

pub use builder::ZeroShotClassificationPipelineBuilder;
pub use model::ZeroShotClassificationModel;
pub use pipeline::ZeroShotClassificationPipeline;

pub use crate::models::ModernBertSize;

pub use anyhow::Result;
