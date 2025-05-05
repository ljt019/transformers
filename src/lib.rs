mod generation;
mod models;
mod utils;

pub mod pipelines;

pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.7;
pub(crate) const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub(crate) const DEFAULT_REPEAT_LAST_N: usize = 64;
pub(crate) const DEFAULT_SEED: u64 = 299792458;
