use super::{
    pipeline::{
        FrameBeginContext, FrameFinishContext, Pipeline, RenderPassContext, SwapchainContext,
    }
    , RendererError, MAX_FRAMES_IN_FLIGHT,
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
}

impl Pipeline for EguiPipeline {
    fn on_swapchain_created(&mut self, ctx: &SwapchainContext<'_>) -> Result<(), String> {
        let render_pass = ctx.render_pass;
        self.renderer
            .set_render_pass(render_pass)
            .map_err(|e| format!("set_egui_render_pass: {e:?}"))
    }

    fn begin_frame(&mut self, ctx: &FrameBeginContext<'_>) -> Result<(), RendererError> {
        let frame_index = ctx.frame_index;
        let queue = ctx.graphics_queue;
        let command_pool = ctx.command_pool;
        let frame = ctx.egui_frame;
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

    fn record_render_pass(&mut self, ctx: &RenderPassContext<'_>) -> Result<(), String> {
        if let Some(frame) = ctx.egui_frame {
            let command_buffer = ctx.command_buffer;
            let extent = ctx.extent;
            self.renderer
                .cmd_draw(
                    command_buffer,
                    extent,
                    frame.pixels_per_point,
                    frame.clipped_primitives.as_slice(),
                )
                .map_err(|e| format!("egui cmd_draw: {e:?}"))?;
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
