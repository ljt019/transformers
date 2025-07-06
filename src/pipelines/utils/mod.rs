use candle_core::{CudaDevice, Device};

/// Loads a device to be used for the model.
/// If `index` is `Some(i)` it will attempt to load the specified CUDA device.
/// When `None` it defaults to CUDA device 0 if available and otherwise falls back
/// to CPU.
pub fn load_device_with(index: Option<usize>) -> anyhow::Result<Device> {
    if let Some(i) = index {
        let cuda = CudaDevice::new_with_stream(i)?;
        Ok(Device::Cuda(cuda))
    } else {
        match CudaDevice::new_with_stream(0) {
            Ok(cuda_device) => Ok(Device::Cuda(cuda_device)),
            Err(_) => Ok(Device::Cpu),
        }
    }
}

/// Convenience wrapper that preserves the previous behaviour of selecting CUDA 0
/// if available and otherwise falling back to CPU.
pub fn load_device() -> anyhow::Result<Device> {
    load_device_with(None)
}
