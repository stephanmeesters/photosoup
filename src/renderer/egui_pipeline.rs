use super::{
    pipeline::{FrameBeginContext, FrameFinishContext, Pipeline, RenderingContext, SwapchainContext},
    RendererError, MAX_FRAMES_IN_FLIGHT,
};
use ash::vk;
use egui_ash_renderer::{DynamicRendering, Options as EguiRendererOptions, Renderer as EguiRenderer};

pub struct EguiPipeline {
    renderer: EguiRenderer,
    pending_free: [Vec<egui::TextureId>; MAX_FRAMES_IN_FLIGHT],
}

impl EguiPipeline {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        color_attachment_format: vk::Format,
    ) -> Result<Self, String> {
        let renderer = EguiRenderer::with_default_allocator(
            instance,
            physical_device,
            device,
            DynamicRendering {
                color_attachment_format,
                depth_attachment_format: None,
            },
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
        self.renderer
            .set_dynamic_rendering(DynamicRendering {
                color_attachment_format: ctx.color_attachment_format,
                depth_attachment_format: None,
            })
            .map_err(|e| format!("set_egui_dynamic_rendering: {e:?}"))
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

    fn record_rendering(&mut self, ctx: &RenderingContext<'_>) -> Result<(), String> {
        if let Some(frame) = ctx.egui_frame {
            let command_buffer = ctx.command_buffer;
            let extent = ctx.extent;
            let clipped_primitives =
                clamp_primitives_to_extent(frame.clipped_primitives.as_slice(), extent, frame.pixels_per_point);
            self.renderer
                .cmd_draw(
                    command_buffer,
                    extent,
                    frame.pixels_per_point,
                    clipped_primitives.as_slice(),
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

fn clamp_primitives_to_extent(
    primitives: &[egui::ClippedPrimitive],
    extent: vk::Extent2D,
    pixels_per_point: f32,
) -> Vec<egui::ClippedPrimitive> {
    let max_x = extent.width as f32 / pixels_per_point;
    let max_y = extent.height as f32 / pixels_per_point;

    primitives
        .iter()
        .filter_map(|primitive| {
            let min = egui::pos2(
                primitive.clip_rect.min.x.clamp(0.0, max_x),
                primitive.clip_rect.min.y.clamp(0.0, max_y),
            );
            let max = egui::pos2(
                primitive.clip_rect.max.x.clamp(0.0, max_x),
                primitive.clip_rect.max.y.clamp(0.0, max_y),
            );
            if min.x >= max.x || min.y >= max.y {
                return None;
            }

            let mut primitive = primitive.clone();
            primitive.clip_rect = egui::Rect::from_min_max(min, max);
            Some(primitive)
        })
        .collect()
}
