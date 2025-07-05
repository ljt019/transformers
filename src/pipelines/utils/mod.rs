use candle_core::{CudaDevice, Device};

pub mod model_cache;

/// Loads a device to be used for the model. Uses CUDA by default, falling back to CPU if CUDA is not available.
pub fn load_device() -> anyhow::Result<Device> {
    match CudaDevice::new_with_stream(0) {
        Ok(cuda_device) => Ok(Device::Cuda(cuda_device)),
        Err(_) => Ok(Device::Cpu),
    }
}
