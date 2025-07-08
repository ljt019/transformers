//! Sentiment analysis pipeline for classifying text emotional tone.
//!
//! This module provides functionality for analyzing the sentiment (positive, negative, neutral)
//! of text inputs using pre-trained transformer models. It's useful for content moderation,
//! customer feedback analysis, and social media monitoring.
//!
//! ## Main Types
//!
//! - [`SentimentAnalysisPipeline`] - High-level interface for sentiment classification  
//! - [`SentimentAnalysisPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`SentimentAnalysisModel`] - Trait for sentiment analysis model implementations
//! - [`ModernBertSize`] - Available model size options
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use transformers::pipelines::sentiment_analysis_pipeline::*;
//!
//! # tokio_test::block_on(async {
//! // Create a sentiment analysis pipeline
//! let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
//!     .build()
//!     .await?;
//!
//! // Analyze sentiment
//! let result = pipeline.predict("I love this product!").await?;
//! println!("Sentiment: {} (confidence: {:.2})", result.label, result.score);
//! # anyhow::Ok(())
//! # });
//! ```

pub mod builder;
pub mod model;
pub mod pipeline;

pub use builder::SentimentAnalysisPipelineBuilder;
pub use model::SentimentAnalysisModel;
pub use pipeline::SentimentAnalysisPipeline;

pub use crate::models::ModernBertSize;

pub use anyhow::Result;
