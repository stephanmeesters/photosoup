use super::{EguiFrame, RendererError};
use ash::vk;

// Data that only changes when the window surface/swapchain changes.
// Pipelines use this to create resources sized to the current backbuffer.
pub struct SwapchainContext<'a> {
    // The Vulkan instance owns physical-device queries and instance-level objects.
    pub instance: &'a ash::Instance,
    // The logical device is the handle used for almost all GPU object creation.
    pub device: &'a ash::Device,
    // The selected adapter/GPU. Needed when choosing memory types.
    pub physical_device: vk::PhysicalDevice,
    // Graphics pipelines created for dynamic rendering declare their attachment format.
    pub color_attachment_format: vk::Format,
    // Raw images owned by the swapchain. We create views around these.
    pub swapchain_images: &'a [vk::Image],
    // Pixel size of every swapchain image.
    pub extent: vk::Extent2D,
}

// Data available before command buffers are recorded for the current frame.
pub struct FrameBeginContext<'a> {
    pub frame_index: usize,
    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub egui_frame: Option<&'a EguiFrame>,
}

// Data available while recording work that happens before dynamic rendering.
pub struct FrameContext<'a> {
    pub device: &'a ash::Device,
    // Command buffers do not execute immediately. Calls prefixed with cmd_* append GPU commands here.
    pub command_buffer: vk::CommandBuffer,
    // Index into swapchain-sized per-image resource arrays.
    pub image_index: usize,
    // The acquired image that will eventually be presented to the window.
    pub swapchain_image: vk::Image,
}

// Data available inside active dynamic rendering.
pub struct RenderingContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub extent: vk::Extent2D,
    pub egui_frame: Option<&'a EguiFrame>,
}

// Data available after queue presentation has been requested.
pub struct FrameFinishContext<'a> {
    pub frame_index: usize,
    pub egui_frame: Option<&'a EguiFrame>,
}

// A small app-level abstraction over Vulkan's explicit lifecycle. Each pass can
// own resources, record commands, and rebuild swapchain-sized resources.
pub trait Pipeline {
    // Called after a swapchain exists or is recreated. Allocate anything that
    // depends on image count, image extent, or attachment format here.
    fn on_swapchain_created(&mut self, _ctx: &SwapchainContext<'_>) -> Result<(), String> {
        Ok(())
    }

    // Called before swapchain-sized resources are destroyed.
    fn destroy_swapchain(&mut self, _device: &ash::Device) {}

    // Called once per CPU frame before acquiring/recording. Egui uses this to
    // upload texture deltas before the render command buffer is finalized.
    fn begin_frame(&mut self, _ctx: &FrameBeginContext<'_>) -> Result<(), RendererError> {
        Ok(())
    }

    // Record commands outside dynamic rendering: compute dispatch, image copies,
    // layout transitions, transfer operations, and other non-subpass work.
    fn record_before_rendering(&mut self, _ctx: &FrameContext<'_>) -> Result<(), String> {
        Ok(())
    }

    // Record draw calls that require active dynamic rendering.
    fn record_rendering(&mut self, _ctx: &RenderingContext<'_>) -> Result<(), String> {
        Ok(())
    }

    // Let passes release frame-local bookkeeping after presentation.
    fn finish_frame(&mut self, _ctx: &FrameFinishContext<'_>) {}

    // Destroy permanent resources. Swapchain-dependent resources should already
    // be gone, but implementations defensively call destroy_swapchain too.
    fn destroy(&mut self, device: &ash::Device);

    fn title(&self) -> &str;
}
