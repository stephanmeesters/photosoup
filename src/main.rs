use ash::{khr, vk};
use egui_ash_renderer::{Options as EguiRendererOptions, Renderer as EguiRenderer};
use egui_winit::State as EguiWinitState;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{ffi::CString, os::raw::c_char, time::Instant};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    egui_ctx: egui::Context,
    egui_state: Option<EguiWinitState>,
    start_time: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("photosoup - Vulkan triangle")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_resizable(true);
        let window = event_loop.create_window(attrs).expect("create window");

        let renderer = Renderer::new(&window).expect("create renderer");
        let egui_state = EguiWinitState::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );

        self.start_time = Some(Instant::now());
        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        if let Some(egui_state) = self.egui_state.as_mut() {
            let _ = egui_state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(egui_state), Some(start_time)) = (
                    self.renderer.as_mut(),
                    self.egui_state.as_mut(),
                    self.start_time,
                ) {
                    let elapsed = Instant::now().saturating_duration_since(start_time);
                    let raw_input = egui_state.take_egui_input(window);
                    let full_output = self.egui_ctx.run(raw_input, |ctx| {
                        egui::Window::new("Hello egui").show(ctx, |ui| {
                            ui.label("Hello world");
                            ui.label(format!("Running for {:.2?}", elapsed));
                        });
                    });
                    egui_state.handle_platform_output(window, full_output.platform_output);

                    let clipped_primitives = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                    let egui_frame = EguiFrame {
                        clipped_primitives,
                        textures_delta: full_output.textures_delta,
                        pixels_per_point: full_output.pixels_per_point,
                    };

                    if let Err(err) = renderer.draw_frame(Some(egui_frame)) {
                        match err {
                            RendererError::OutOfDate => renderer.recreate_swapchain(),
                            RendererError::Fatal(message) => {
                                eprintln!("{message}");
                                event_loop.exit();
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    graphics_family: u32,
    present_family: u32,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    egui_renderer: EguiRenderer,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    pending_egui_free: [Vec<egui::TextureId>; MAX_FRAMES_IN_FLIGHT],
    current_frame: usize,
    pending_extent: Option<vk::Extent2D>,
}

#[derive(Debug)]
enum RendererError {
    OutOfDate,
    Fatal(String),
}

struct EguiFrame {
    clipped_primitives: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    pixels_per_point: f32,
}

#[derive(Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl Renderer {
    fn new(window: &Window) -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| e.to_string())?;

        let app_name = CString::new("photosoup").unwrap();
        let engine_name = CString::new("photosoup").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_0);

        let display_handle = window.display_handle().map_err(|e| e.to_string())?;
        let required_extensions =
            ash_window::enumerate_required_extensions(display_handle.as_raw())
                .map_err(|e| format!("enumerate_required_extensions: {e:?}"))?;

        let layers: Vec<*const c_char> = Vec::new();
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(required_extensions)
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|e| format!("create_instance: {e:?}"))?;

        let window_handle = window.window_handle().map_err(|e| e.to_string())?;
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .map_err(|e| format!("create_surface: {e:?}"))?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let (physical_device, queue_families) =
            pick_physical_device(&instance, &surface_loader, surface)?;

        let device_extensions = [khr::swapchain::NAME.as_ptr()];
        let queue_priorities = [1.0_f32];
        let mut queue_infos = Vec::new();
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_families.graphics_family)
                .queue_priorities(&queue_priorities),
        );
        if queue_families.present_family != queue_families.graphics_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_families.present_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions);

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|e| format!("create_device: {e:?}"))?;
        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_families.present_family, 0) };
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        let egui_renderer = EguiRenderer::with_default_allocator(
            &instance,
            physical_device,
            device.clone(),
            vk::RenderPass::null(),
            EguiRendererOptions {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                srgb_framebuffer: true,
                ..Default::default()
            },
        )
        .map_err(|e| format!("create_egui_renderer: {e:?}"))?;

        let mut renderer = Self {
            _entry: entry,
            instance,
            surface_loader,
            swapchain_loader,
            surface,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            graphics_family: queue_families.graphics_family,
            present_family: queue_families.present_family,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: Vec::new(),
            swapchain_image_views: Vec::new(),
            swapchain_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            render_pass: vk::RenderPass::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            egui_renderer,
            framebuffers: Vec::new(),
            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),
            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            pending_egui_free: std::array::from_fn(|_| Vec::new()),
            current_frame: 0,
            pending_extent: None,
        };

        let size = window.inner_size();
        renderer.create_swapchain(vk::Extent2D {
            width: size.width,
            height: size.height,
        })?;
        renderer.create_render_resources()?;
        renderer
            .egui_renderer
            .set_render_pass(renderer.render_pass)
            .map_err(|e| format!("set_egui_render_pass: {e:?}"))?;
        renderer.create_sync_objects()?;
        Ok(renderer)
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.pending_extent = Some(vk::Extent2D { width, height });
        self.recreate_swapchain();
    }

    fn recreate_swapchain(&mut self) {
        let _ = unsafe { self.device.device_wait_idle() };
        self.destroy_swapchain_resources();
        let extent_hint = self.pending_extent.take().unwrap_or(self.swapchain_extent);
        if let Err(err) = self.create_swapchain(extent_hint) {
            eprintln!("{err}");
            return;
        }
        if let Err(err) = self.create_render_resources() {
            eprintln!("{err}");
            return;
        }
        if let Err(err) = self.egui_renderer.set_render_pass(self.render_pass) {
            eprintln!("set_egui_render_pass: {err:?}");
        }
    }

    fn draw_frame(&mut self, egui_frame: Option<EguiFrame>) -> Result<(), RendererError> {
        let fence = self.in_flight_fences[self.current_frame];
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| RendererError::Fatal(format!("wait_for_fences: {e:?}")))?;
            self.device
                .reset_fences(&[fence])
                .map_err(|e| RendererError::Fatal(format!("reset_fences: {e:?}")))?;
        }

        if !self.pending_egui_free[self.current_frame].is_empty() {
            self.egui_renderer
                .free_textures(&self.pending_egui_free[self.current_frame])
                .map_err(|e| RendererError::Fatal(format!("egui free_textures: {e:?}")))?;
            self.pending_egui_free[self.current_frame].clear();
        }

        let (image_index, acquire_suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        }
        .map_err(|e| match e {
            vk::Result::ERROR_OUT_OF_DATE_KHR => RendererError::OutOfDate,
            other => RendererError::Fatal(format!("acquire_next_image: {other:?}")),
        })?;

        if acquire_suboptimal {
            self.recreate_swapchain();
            return Ok(());
        }

        if let Some(frame) = egui_frame.as_ref() {
            self.egui_renderer
                .set_textures(
                    self.graphics_queue,
                    self.command_pool,
                    frame.textures_delta.set.as_slice(),
                )
                .map_err(|e| RendererError::Fatal(format!("egui set_textures: {e:?}")))?;
        }

        let command_buffer = self.command_buffers[image_index as usize];
        self.record_command_buffer(command_buffer, image_index, egui_frame.as_ref())
            .map_err(RendererError::Fatal)?;

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], fence)
                .map_err(|e| RendererError::Fatal(format!("queue_submit: {e:?}")))?;
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        if let Some(frame) = egui_frame {
            self.pending_egui_free[self.current_frame] = frame.textures_delta.free;
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        match result {
            Ok(suboptimal) if suboptimal => {
                self.recreate_swapchain();
                Ok(())
            }
            Ok(_) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(RendererError::OutOfDate),
            Err(err) => Err(RendererError::Fatal(format!("queue_present: {err:?}"))),
        }
    }

    fn record_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
        egui_frame: Option<&EguiFrame>,
    ) -> Result<(), String> {
        let _ = image_index;

        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset_command_buffer: {e:?}"))?;
        }

        let begin_info = vk::CommandBufferBeginInfo::default();
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.05, 0.06, 0.10, 1.0],
            },
        }];
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| format!("begin_command_buffer: {e:?}"))?;
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            if let Some(frame) = egui_frame {
                self.egui_renderer
                    .cmd_draw(
                        command_buffer,
                        self.swapchain_extent,
                        frame.pixels_per_point,
                        frame.clipped_primitives.as_slice(),
                    )
                    .map_err(|e| format!("egui cmd_draw: {e:?}"))?;
            }

            self.device.cmd_end_render_pass(command_buffer);
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|e| format!("end_command_buffer: {e:?}"))?;
        }

        Ok(())
    }

    fn create_swapchain(&mut self, extent_hint: vk::Extent2D) -> Result<(), String> {
        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .map_err(|e| format!("surface capabilities: {e:?}"))?;

        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
        }
        .map_err(|e| format!("surface formats: {e:?}"))?;

        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }
        .map_err(|e| format!("present modes: {e:?}"))?;

        let surface_format = formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(formats[0]);

        let present_mode = present_modes
            .iter()
            .copied()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: extent_hint.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: extent_hint.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 {
            image_count = image_count.min(surface_caps.max_image_count);
        }

        let indices = [self.graphics_family, self.present_family];
        let (image_sharing_mode, queue_family_indices) =
            if self.graphics_family == self.present_family {
                (vk::SharingMode::EXCLUSIVE, Vec::new())
            } else {
                (vk::SharingMode::CONCURRENT, indices.to_vec())
            };

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_info, None)
        }
        .map_err(|e| format!("create_swapchain: {e:?}"))?;
        let images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .map_err(|e| format!("get_swapchain_images: {e:?}"))?;

        self.swapchain = swapchain;
        self.swapchain_images = images;
        self.swapchain_format = surface_format.format;
        self.swapchain_extent = extent;
        Ok(())
    }

    fn create_render_resources(&mut self) -> Result<(), String> {
        self.create_image_views()?;
        self.create_render_pass()?;
        self.create_pipeline()?;
        self.create_framebuffers()?;
        self.create_command_pool()?;
        self.create_command_buffers()?;
        Ok(())
    }

    fn create_image_views(&mut self) -> Result<(), String> {
        self.swapchain_image_views = self
            .swapchain_images
            .iter()
            .copied()
            .map(|image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.swapchain_format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { self.device.create_image_view(&view_info, None) }
                    .map_err(|e| format!("create_image_view: {e:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    fn create_render_pass(&mut self) -> Result<(), String> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(self.swapchain_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        self.render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None) }
            .map_err(|e| format!("create_render_pass: {e:?}"))?;
        Ok(())
    }

    fn create_pipeline(&mut self) -> Result<(), String> {
        let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"));
        let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"));

        let vert_shader_module = create_shader_module(&self.device, vert_spv)?;
        let frag_shader_module = create_shader_module(&self.device, frag_spv)?;

        let entry_name = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(vert_shader_module)
                .name(&entry_name)
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(frag_shader_module)
                .name(&entry_name)
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.swapchain_extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        self.pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|e| format!("create_pipeline_layout: {e:?}"))?;

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(self.pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0);

        self.pipeline = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, e)| format!("create_graphics_pipelines: {e:?}"))?[0];

        unsafe {
            self.device.destroy_shader_module(vert_shader_module, None);
            self.device.destroy_shader_module(frag_shader_module, None);
        }
        Ok(())
    }

    fn create_framebuffers(&mut self) -> Result<(), String> {
        self.framebuffers = self
            .swapchain_image_views
            .iter()
            .copied()
            .map(|view| {
                let attachments = [view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&attachments)
                    .width(self.swapchain_extent.width)
                    .height(self.swapchain_extent.height)
                    .layers(1);
                unsafe { self.device.create_framebuffer(&framebuffer_info, None) }
                    .map_err(|e| format!("create_framebuffer: {e:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    fn create_command_pool(&mut self) -> Result<(), String> {
        if self.command_pool != vk::CommandPool::null() {
            unsafe { self.device.destroy_command_pool(self.command_pool, None) };
        }
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.command_pool = unsafe { self.device.create_command_pool(&pool_info, None) }
            .map_err(|e| format!("create_command_pool: {e:?}"))?;
        Ok(())
    }

    fn create_command_buffers(&mut self) -> Result<(), String> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(self.framebuffers.len() as u32);
        self.command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| format!("allocate_command_buffers: {e:?}"))?;
        Ok(())
    }

    fn create_sync_objects(&mut self) -> Result<(), String> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        self.image_available_semaphores.clear();
        self.render_finished_semaphores.clear();
        self.in_flight_fences.clear();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            self.image_available_semaphores.push(
                unsafe { self.device.create_semaphore(&semaphore_info, None) }
                    .map_err(|e| format!("create_semaphore: {e:?}"))?,
            );
            self.render_finished_semaphores.push(
                unsafe { self.device.create_semaphore(&semaphore_info, None) }
                    .map_err(|e| format!("create_semaphore: {e:?}"))?,
            );
            self.in_flight_fences.push(
                unsafe { self.device.create_fence(&fence_info, None) }
                    .map_err(|e| format!("create_fence: {e:?}"))?,
            );
        }
        Ok(())
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            if self.command_pool != vk::CommandPool::null() && !self.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
            }
            self.command_buffers.clear();
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
                self.render_pass = vk::RenderPass::null();
            }
            for view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.destroy_swapchain_resources();
            for i in 0..self.image_available_semaphores.len() {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, QueueFamilyIndices), String> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("enumerate_physical_devices: {e:?}"))?;

    for device in devices {
        if let Some(indices) = find_queue_families(instance, device, surface_loader, surface)? {
            let extensions = unsafe { instance.enumerate_device_extension_properties(device) }
                .map_err(|e| format!("enumerate_device_extension_properties: {e:?}"))?;
            let swapchain_supported = extensions.iter().any(|prop| {
                let name = unsafe { std::ffi::CStr::from_ptr(prop.extension_name.as_ptr()) };
                name.to_bytes() == khr::swapchain::NAME.to_bytes()
            });
            if swapchain_supported {
                return Ok((device, indices));
            }
        }
    }

    Err("no suitable Vulkan device found".to_string())
}

fn find_queue_families(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<Option<QueueFamilyIndices>, String> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut graphics_family = None;
    let mut present_family = None;

    for (index, family) in families.iter().enumerate() {
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_family = Some(index as u32);
        }
        let present_support = unsafe {
            surface_loader.get_physical_device_surface_support(device, index as u32, surface)
        }
        .map_err(|e| format!("surface_support: {e:?}"))?;
        if present_support {
            present_family = Some(index as u32);
        }
    }

    Ok(match (graphics_family, present_family) {
        (Some(graphics_family), Some(present_family)) => Some(QueueFamilyIndices {
            graphics_family,
            present_family,
        }),
        _ => None,
    })
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, String> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
        .map_err(|e| format!("read_spv: {e:?}"))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    unsafe { device.create_shader_module(&info, None) }
        .map_err(|e| format!("create_shader_module: {e:?}"))
}
