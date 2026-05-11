use super::{EguiFrame, RendererError};
use ash::vk;

pub struct SwapchainContext<'a> {
    pub instance: &'a ash::Instance,
    pub device: &'a ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub render_pass: vk::RenderPass,
    pub swapchain_images: &'a [vk::Image],
    pub extent: vk::Extent2D,
}

pub struct FrameBeginContext<'a> {
    pub frame_index: usize,
    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub egui_frame: Option<&'a EguiFrame>,
}

pub struct FrameContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub image_index: usize,
    pub swapchain_image: vk::Image,
}

pub struct RenderPassContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub extent: vk::Extent2D,
    pub egui_frame: Option<&'a EguiFrame>,
}

pub struct FrameFinishContext<'a> {
    pub frame_index: usize,
    pub egui_frame: Option<&'a EguiFrame>,
}

pub trait Pipeline {
    fn on_swapchain_created(&mut self, _ctx: &SwapchainContext<'_>) -> Result<(), String> {
        Ok(())
    }

    fn destroy_swapchain(&mut self, _device: &ash::Device) {}

    fn begin_frame(&mut self, _ctx: &FrameBeginContext<'_>) -> Result<(), RendererError> {
        Ok(())
    }

    fn record_before_render_pass(&mut self, _ctx: &FrameContext<'_>) -> Result<(), String> {
        Ok(())
    }

    fn record_render_pass(&mut self, _ctx: &RenderPassContext<'_>) -> Result<(), String> {
        Ok(())
    }

    fn finish_frame(&mut self, _ctx: &FrameFinishContext<'_>) {}

    fn destroy(&mut self, device: &ash::Device);
}
