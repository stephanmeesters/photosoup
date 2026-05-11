use super::{
    pipeline::{
        FrameBeginContext, FrameFinishContext, RenderPassContext, Pipeline, SwapchainContext,
    },
    EguiFrame, RendererError, MAX_FRAMES_IN_FLIGHT,
};
use ash::vk;
use egui_ash_renderer::{Options as EguiRendererOptions, Renderer as EguiRenderer};

pub struct EguiPipeline {
    renderer: EguiRenderer,
    pending_free: [Vec<egui::TextureId>; MAX_FRAMES_IN_FLIGHT],
}

impl EguiPipeline {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        render_pass: vk::RenderPass,
    ) -> Result<Self, String> {
        let renderer = EguiRenderer::with_default_allocator(
            instance,
            physical_device,
            device,
            render_pass,
            EguiRendererOptions {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                srgb_framebuffer: true,
                ..Default::default()
            },
        )
        .map_err(|e| format!("create_egui_renderer: {e:?}"))?;

        Ok(Self {
            renderer,
            pending_free: std::array::from_fn(|_| Vec::new()),
        })
    }

    pub fn set_render_pass(&mut self, render_pass: vk::RenderPass) -> Result<(), String> {
        self.renderer
            .set_render_pass(render_pass)
            .map_err(|e| format!("set_egui_render_pass: {e:?}"))
    }

    pub fn begin_frame(
        &mut self,
        frame_index: usize,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        frame: Option<&EguiFrame>,
    ) -> Result<(), RendererError> {
        if !self.pending_free[frame_index].is_empty() {
            self.renderer
                .free_textures(&self.pending_free[frame_index])
                .map_err(|e| RendererError::Fatal(format!("egui free_textures: {e:?}")))?;
            self.pending_free[frame_index].clear();
        }

        if let Some(frame) = frame {
            self.renderer
                .set_textures(queue, command_pool, frame.textures_delta.set.as_slice())
                .map_err(|e| RendererError::Fatal(format!("egui set_textures: {e:?}")))?;
        }

        Ok(())
    }

    pub fn record(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame: &EguiFrame,
    ) -> Result<(), String> {
        self.renderer
            .cmd_draw(
                command_buffer,
                extent,
                frame.pixels_per_point,
                frame.clipped_primitives.as_slice(),
            )
            .map_err(|e| format!("egui cmd_draw: {e:?}"))
    }
}

impl Pipeline for EguiPipeline {
    fn on_swapchain_created(&mut self, ctx: &SwapchainContext<'_>) -> Result<(), String> {
        self.set_render_pass(ctx.render_pass)
    }

    fn begin_frame(&mut self, ctx: &FrameBeginContext<'_>) -> Result<(), RendererError> {
        self.begin_frame(
            ctx.frame_index,
            ctx.graphics_queue,
            ctx.command_pool,
            ctx.egui_frame,
        )
    }

    fn record_render_pass(&mut self, ctx: &RenderPassContext<'_>) -> Result<(), String> {
        if let Some(frame) = ctx.egui_frame {
            self.record(ctx.command_buffer, ctx.extent, frame)?;
        }
        Ok(())
    }

    fn finish_frame(&mut self, ctx: &FrameFinishContext<'_>) {
        if let Some(frame) = ctx.egui_frame {
            self.pending_free[ctx.frame_index] = frame.textures_delta.free.clone();
        }
    }

    fn destroy(&mut self, _device: &ash::Device) {}
}
