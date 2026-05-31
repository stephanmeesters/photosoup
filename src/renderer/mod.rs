mod compute_circle_pipeline;
mod compute_target;
mod egui_pipeline;
mod pipeline;
mod shader;
mod triangle_pipeline;

use ash::{khr, vk};
use compute_circle_pipeline::ComputeCirclePass;
use egui_pipeline::EguiPipeline;
use pipeline::{FrameBeginContext, FrameContext, FrameFinishContext, Pipeline, RenderingContext, SwapchainContext};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use rayon::prelude::*;
use std::{ffi::CString, os::raw::c_char};
use triangle_pipeline::TrianglePass;
use winit::window::Window;

pub(super) const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct PerFrameGoodies {
    // primary pool/buffer
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // per thread
    per_thread_goodies: Vec<PerThreadGoodies>,
}

pub struct PerThreadGoodies {
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

pub struct Renderer {
    // Entry loads Vulkan function pointers from the system loader.
    _entry: ash::Entry,
    // Instance is the root Vulkan object for physical-device and surface queries.
    instance: ash::Instance,
    // Extension loader for surface-related KHR calls.
    surface_loader: khr::surface::Instance,
    // Extension loader for swapchain-related KHR calls.
    swapchain_loader: khr::swapchain::Device,
    // Window-system surface that connects Vulkan presentation to the winit window.
    surface: vk::SurfaceKHR,
    // Selected GPU/adapter.
    physical_device: vk::PhysicalDevice,
    // Logical device used to create queues and most Vulkan objects.
    device: ash::Device,
    // Queue that executes graphics/compute/transfer commands in this app.
    graphics_queue: vk::Queue,
    // Queue used to present swapchain images to the window.
    present_queue: vk::Queue,
    graphics_family: u32,
    present_family: u32,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    pipelines: Vec<Box<dyn Pipeline + Send>>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    // Resize requests are stored until swapchain recreation can safely use them.
    pending_extent: Option<vk::Extent2D>,
    per_frame_goodies_list: Vec<Option<PerFrameGoodies>>,
}

#[derive(Debug)]
pub enum RendererError {
    OutOfDate,
    Fatal(String),
}

pub struct EguiFrame {
    pub clipped_primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

#[derive(Clone, Copy)]
pub struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, String> {
        // ash::Entry::load dynamically loads the Vulkan loader from the OS.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| e.to_string())?;

        // ApplicationInfo is mostly metadata, but api_version selects the Vulkan
        // version whose commands/features we promise to use.
        let app_name = CString::new("photosoup").unwrap();
        let engine_name = CString::new("photosoup").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        // ash-window asks the platform which instance extensions are required
        // for creating a surface for this specific display backend.
        let display_handle = window.display_handle().map_err(|e| e.to_string())?;
        let required_extensions = ash_window::enumerate_required_extensions(display_handle.as_raw())
            .map_err(|e| format!("enumerate_required_extensions: {e:?}"))?;

        // Instance creation enables platform surface extensions. No validation
        // layers are enabled here because layers is intentionally empty.
        let layers: Vec<*const c_char> = Vec::new();
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(required_extensions)
            .enabled_layer_names(&layers);

        let instance =
            unsafe { entry.create_instance(&instance_info, None) }.map_err(|e| format!("create_instance: {e:?}"))?;

        // The surface wraps the native window/display handles in a Vulkan object
        // that can be queried for presentation support.
        let window_handle = window.window_handle().map_err(|e| e.to_string())?;
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle.as_raw(), window_handle.as_raw(), None)
        }
        .map_err(|e| format!("create_surface: {e:?}"))?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        // Pick a GPU that can render and present to this surface and supports
        // VK_KHR_swapchain.
        let (physical_device, queue_families) = pick_physical_device(&instance, &surface_loader, surface)?;

        // VK_KHR_swapchain is a device extension, so it is enabled when creating
        // the logical device, not when creating the instance.
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

        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut dynamic_rendering_features);

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|e| format!("create_device: {e:?}"))?;
        // Queue handles are retrieved from the logical device; index 0 means the
        // first queue from each requested queue family.
        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_family, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_families.present_family, 0) };
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        // Build the renderer in phases: create long-lived Vulkan objects first,
        // then swapchain-sized resources that can be rebuilt on resize.
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
            pipelines: Vec::new(),
            command_pool: vk::CommandPool::null(),
            command_buffers: Vec::new(),
            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            current_frame: 0,
            pending_extent: None,
            per_frame_goodies_list: vec![None, None],
        };

        let size = window.inner_size();
        renderer.create_swapchain(vk::Extent2D {
            width: size.width,
            height: size.height,
        })?;
        renderer.create_image_views()?;
        renderer.create_pipelines()?;
        renderer.create_swapchain_dependent_resources()?;
        renderer.create_sync_objects()?;
        Ok(renderer)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        // A minimized window can report zero size. Vulkan swapchains cannot have
        // zero width/height, so wait until a real size arrives.
        if width == 0 || height == 0 {
            return;
        }
        self.pending_extent = Some(vk::Extent2D { width, height });
        self.recreate_swapchain();
    }

    pub fn recreate_swapchain(&mut self) {
        // Wait for queued work to finish before destroying resources that command
        // buffers or presentation may still reference.
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
    }

    pub fn draw_frame(&mut self, egui_frame: Option<EguiFrame>) -> Result<(), RendererError> {
        let fence = self.in_flight_fences[self.current_frame];
        unsafe {
            // This fence was attached to the last submit for current_frame. Waiting
            // prevents overwriting per-frame CPU/GPU resources still in use.
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| RendererError::Fatal(format!("wait_for_fences: {e:?}")))?;
        }

        // Acquire chooses which swapchain image we may render into. The semaphore
        // is signaled by the presentation engine when that image is ready.
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

        //////////////////
        if let Some(pfg) = self.per_frame_goodies_list[self.current_frame].take() {
            for bb in pfg.per_thread_goodies.iter() {
                unsafe { self.device.destroy_command_pool(bb.command_pool, None) };
            }
            unsafe { self.device.destroy_command_pool(pfg.command_pool, None) };
        }

        //////////////////
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { self.device.create_command_pool(&pool_info, None) }.unwrap();

        // Use one primary command buffer per swapchain image.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info) }.unwrap();
        let command_buffer = command_buffers[0];

        let mut ptg = Vec::new();
        for _ in 0..self.pipelines.len() {
            let sec_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.graphics_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let sec_command_pool = unsafe { self.device.create_command_pool(&sec_pool_info, None) }.unwrap();

            let sec_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(sec_command_pool)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(2);
            let sec_command_buffers = unsafe { self.device.allocate_command_buffers(&sec_alloc_info) }.unwrap();

            ptg.push(PerThreadGoodies {
                command_pool: sec_command_pool,
                command_buffers: sec_command_buffers,
            })
        }

        let pfg = PerFrameGoodies {
            command_pool,
            command_buffers,
            per_thread_goodies: ptg,
        };

        //////////////////

        let begin_context = FrameBeginContext {
            frame_index: self.current_frame,
            graphics_queue: self.graphics_queue,
            command_pool,
            egui_frame: egui_frame.as_ref(),
        };
        // egui only
        for pipeline in &mut self.pipelines {
            pipeline.begin_frame(&begin_context)?;
        }

        // Record all GPU commands for the acquired image before submitting them.
        let egui_frame1 = egui_frame.as_ref();
        unsafe {
            // RESET_COMMAND_BUFFER lets us reuse the per-image command buffer
            // instead of allocating a new one every frame.
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset_command_buffer: {e:?}"))
                .unwrap();
        }

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain_image_views[image_index as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);

        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            // After begin_command_buffer, cmd_* calls append GPU work into this buffer.
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| format!("begin_command_buffer: {e:?}"))
                .unwrap();
        }

        self.pipelines.par_iter_mut().enumerate().for_each(|(index, pipeline)| {

            let frame_context = FrameContext {
                device: &self.device,
                command_buffer: pfg.per_thread_goodies[index].command_buffers[0],
                image_index: image_index as usize,
                swapchain_image: self.swapchain_images[image_index as usize],
            };

            let sec = frame_context.command_buffer;
            let inheritance_info = vk::CommandBufferInheritanceInfo::default();
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .inheritance_info(&inheritance_info);
            unsafe { self.device.begin_command_buffer(sec, &begin_info).unwrap() };

            // Compute and image-copy work happens before dynamic rendering starts.
            pipeline.record_before_rendering(&frame_context).unwrap();

            unsafe { self.device.end_command_buffer(sec).unwrap() };

            let rendering_context = RenderingContext {
                device: &self.device,
                command_buffer: pfg.per_thread_goodies[index].command_buffers[1],
                extent: self.swapchain_extent,
                egui_frame: egui_frame1,
            };

            let sec = rendering_context.command_buffer;
            let color_attachment_formats = [self.swapchain_format];
            let mut inheritance_rendering_info = vk::CommandBufferInheritanceRenderingInfo::default()
                .color_attachment_formats(&color_attachment_formats)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let inheritance_info =
                vk::CommandBufferInheritanceInfo::default().push_next(&mut inheritance_rendering_info);
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
                .inheritance_info(&inheritance_info);
            unsafe { self.device.begin_command_buffer(sec, &begin_info).unwrap() };

            // Graphics pipelines and egui record draw calls while the color
            // attachment is active.
            pipeline.record_rendering(&rendering_context).unwrap();

            unsafe { self.device.end_command_buffer(sec).unwrap() };
        });

        let secondaries: Vec<vk::CommandBuffer> = pfg.per_thread_goodies.iter().map(|p| p.command_buffers[0]).collect();
        unsafe { self.device.cmd_execute_commands(command_buffer, &secondaries) };

        // Dynamic rendering binds the current swapchain image view as the
        // active color attachment without a VkRenderPass/VkFramebuffer pair.
        unsafe { self.device.cmd_begin_rendering(command_buffer, &rendering_info) };

        let secondaries: Vec<vk::CommandBuffer> = pfg.per_thread_goodies.iter().map(|p| p.command_buffers[1]).collect();
        unsafe { self.device.cmd_execute_commands(command_buffer, &secondaries) };

        unsafe { self.device.cmd_end_rendering(command_buffer) };
        Ok(()).map_err(RendererError::Fatal)?;

        // Hand the acquired swapchain image from rendering back to presentation.
        //
        // record_command_buffer leaves the image in COLOR_ATTACHMENT_OPTIMAL after
        // compute/copy work and dynamic rendering have finished writing it. The
        // presentation engine cannot read that layout, so the last recorded GPU
        // operation for the frame transitions the image to PRESENT_SRC_KHR.
        let device = &self.device;
        let image = self.swapchain_images[image_index as usize];
        let old_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let new_layout = vk::ImageLayout::PRESENT_SRC_KHR;

        // Wait for all color-attachment reads/writes from dynamic rendering to be
        // complete before the layout transition. Presentation itself is ordered by
        // the render_finished semaphore, so there is no destination access mask.
        let src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        let dst_access_mask = vk::AccessFlags::empty();
        let src_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;

        // This barrier applies only to the one 2D color image that was acquired
        // from the swapchain for this frame. Queue family ownership is unchanged.
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        unsafe {
            // Record the synchronization/layout transition into the same command
            // buffer as the rendering. It will execute on the GPU after the
            // preceding draw commands and before the command buffer completes.
            device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }

        unsafe {
            // Ending finalizes validation of the recorded commands. The buffer is
            // then ready to submit to a queue.
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|e| RendererError::Fatal(format!("end_command_buffer: {e:?}")))?;
        }

        unsafe {
            // The fence is about to be used for a new queue submission, so it must
            // be unsignaled before queue_submit attaches it.
            self.device
                .reset_fences(&[fence])
                .map_err(|e| RendererError::Fatal(format!("reset_fences: {e:?}")))?;
        }

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];
        // The command buffer starts with offscreen compute work. The acquired
        // swapchain image is first touched by the transfer stage.
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            // Submit sends the recorded command buffer to the GPU. It waits on
            // image_available, signals render_finished, and signals fence on completion.
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], fence)
                .map_err(|e| RendererError::Fatal(format!("queue_submit: {e:?}")))?;
        }

        // Present waits until rendering is complete, then hands the image to the
        // window system for display.
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let result = unsafe { self.swapchain_loader.queue_present(self.present_queue, &present_info) };

        let finish_context = FrameFinishContext {
            frame_index: self.current_frame,
            egui_frame: egui_frame.as_ref(),
        };
        // egui only
        for pipeline in &mut self.pipelines {
            pipeline.finish_frame(&finish_context);
        }

        self.per_frame_goodies_list[self.current_frame] = Some(pfg);

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

    fn create_swapchain(&mut self, extent_hint: vk::Extent2D) -> Result<(), String> {
        // Surface capabilities tell us image counts, transforms, extents, and
        // usages supported by this window/system combination.
        let surface_caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .map_err(|e| format!("surface capabilities: {e:?}"))?;

        // Surface formats define the color format and color space of presented images.
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
        }
        .map_err(|e| format!("surface formats: {e:?}"))?;

        // Present modes control pacing: FIFO is vsync-like and always available;
        // MAILBOX is low-latency triple buffering when supported.
        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }
        .map_err(|e| format!("present modes: {e:?}"))?;

        // The compute pass copies into the swapchain before dynamic rendering
        // uses the image as a color attachment.
        let required_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;
        if !surface_caps.supported_usage_flags.contains(required_usage) {
            return Err("surface does not support color attachment + transfer destination images".to_string());
        }

        // Prefer a common sRGB format for correct presentation gamma, otherwise
        // use the first format the surface reports.
        let surface_format = formats
            .iter()
            .copied()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or(formats[0]);

        // Prefer MAILBOX for lower latency; FIFO is guaranteed by Vulkan.
        let present_mode = present_modes
            .iter()
            .copied()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        // let present_mode = vk::PresentModeKHR::FIFO;

        // Some platforms dictate the swapchain extent. Others let the app choose
        // within min/max bounds, so clamp the window size hint.
        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: extent_hint
                    .width
                    .clamp(surface_caps.min_image_extent.width, surface_caps.max_image_extent.width),
                height: extent_hint.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        // Request one more image than the minimum so rendering can proceed while
        // another image is queued for presentation, capped by max_image_count.
        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 {
            image_count = image_count.min(surface_caps.max_image_count);
        }

        // EXCLUSIVE is faster when graphics and present use the same queue family.
        // CONCURRENT is simpler when ownership must span two different families.
        let indices = [self.graphics_family, self.present_family];
        let (image_sharing_mode, queue_family_indices) = if self.graphics_family == self.present_family {
            (vk::SharingMode::EXCLUSIVE, Vec::new())
        } else {
            (vk::SharingMode::CONCURRENT, indices.to_vec())
        };

        // Swapchain creation allocates the presentation images owned by the
        // window system. We get their handles afterward with get_swapchain_images.
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain = unsafe { self.swapchain_loader.create_swapchain(&swapchain_info, None) }
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
        // These resources depend on swapchain format, extent, image count, or
        // swapchain image count, so they are rebuilt together.
        self.create_image_views()?;
        self.create_swapchain_dependent_resources()
    }

    fn create_pipelines(&mut self) -> Result<(), String> {
        // Pipeline order is the frame order: compute background, triangle overlay,
        // then egui overlay.
        let compute_circle_pass =
            ComputeCirclePass::new(&self.device).map_err(|e| format!("create_compute_circle_pipeline: {e}"))?;

        let egui_pipeline = EguiPipeline::new(
            &self.instance,
            self.physical_device,
            self.device.clone(),
            self.swapchain_format,
        )
        .map_err(|e| format!("create_egui_pipeline: {e}"))?;

        self.pipelines = vec![
            Box::new(compute_circle_pass),
            Box::new(TrianglePass::default()),
            Box::new(egui_pipeline),
        ];
        Ok(())
    }

    fn create_swapchain_dependent_resources(&mut self) -> Result<(), String> {
        // Give each pipeline the handles it needs to allocate resources tied to
        // the current swapchain.
        let swapchain_context = SwapchainContext {
            instance: &self.instance,
            device: &self.device,
            physical_device: self.physical_device,
            color_attachment_format: self.swapchain_format,
            swapchain_images: &self.swapchain_images,
            extent: self.swapchain_extent,
        };
        for pipeline in &mut self.pipelines {
            pipeline.on_swapchain_created(&swapchain_context)?;
        }
        // self.create_command_pool()?;
        // self.create_command_buffers()?;
        self.create_render_finished_semaphores()?;
        Ok(())
    }

    fn create_image_views(&mut self) -> Result<(), String> {
        // Swapchain images are raw images; image views describe how they are read
        // or written as 2D color images.
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

    // fn create_command_pool(&mut self) -> Result<(), String> {
    //     if self.command_pool != vk::CommandPool::null() {
    //         unsafe { self.device.destroy_command_pool(self.command_pool, None) };
    //     }
    //     // Command buffers allocated from this pool can be submitted to the
    //     // graphics queue family. RESET_COMMAND_BUFFER permits per-frame reuse.
    //     let pool_info = vk::CommandPoolCreateInfo::default()
    //         .queue_family_index(self.graphics_family)
    //         .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    //     self.command_pool = unsafe { self.device.create_command_pool(&pool_info, None) }
    //         .map_err(|e| format!("create_command_pool: {e:?}"))?;
    //     Ok(())
    // }

    // fn create_command_buffers(&mut self) -> Result<(), String> {
    //     // Use one primary command buffer per swapchain image.
    //     let alloc_info = vk::CommandBufferAllocateInfo::default()
    //         .command_pool(self.command_pool)
    //         .level(vk::CommandBufferLevel::PRIMARY)
    //         .command_buffer_count(self.swapchain_images.len() as u32);
    //     self.command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info) }
    //         .map_err(|e| format!("allocate_command_buffers: {e:?}"))?;
    //     Ok(())
    // }

    fn create_sync_objects(&mut self) -> Result<(), String> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // Start fences signaled so the first frame does not wait forever for a
        // submission that has not happened yet.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        self.image_available_semaphores.clear();
        self.in_flight_fences.clear();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            self.image_available_semaphores.push(
                unsafe { self.device.create_semaphore(&semaphore_info, None) }
                    .map_err(|e| format!("create_semaphore: {e:?}"))?,
            );
            self.in_flight_fences.push(
                unsafe { self.device.create_fence(&fence_info, None) }.map_err(|e| format!("create_fence: {e:?}"))?,
            );
        }
        Ok(())
    }

    fn create_render_finished_semaphores(&mut self) -> Result<(), String> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        self.destroy_render_finished_semaphores();

        // These are indexed by swapchain image because presentation waits on the
        // image-specific render completion signal.
        for _ in 0..self.swapchain_images.len() {
            self.render_finished_semaphores.push(
                unsafe { self.device.create_semaphore(&semaphore_info, None) }
                    .map_err(|e| format!("create_render_finished_semaphore: {e:?}"))?,
            );
        }
        Ok(())
    }

    fn destroy_render_finished_semaphores(&mut self) {
        unsafe {
            // Semaphores have no Rust owner; every created Vulkan handle must be
            // explicitly destroyed when no queue can still reference it.
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            // Destroy in reverse dependency order: objects that reference the
            // swapchain images go away before the swapchain itself.
            self.destroy_render_finished_semaphores();
            if self.command_pool != vk::CommandPool::null() && !self.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
            }
            self.command_buffers.clear();
            for pipeline in &mut self.pipelines {
                pipeline.destroy_swapchain(&self.device);
            }
            for view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            // Final idle wait makes teardown simple: no submitted command buffer
            // can still be using resources we are about to destroy.
            let _ = self.device.device_wait_idle();
            self.destroy_swapchain_resources();
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }

            for ff in self.per_frame_goodies_list.iter() {
                if let Some(pfg) = ff {
                    for bb in pfg.per_thread_goodies.iter() {
                        self.device.destroy_command_pool(bb.command_pool, None);
                    }
                    self.device.destroy_command_pool(pfg.command_pool, None);
                }
            }
            // if self.command_pool != vk::CommandPool::null() {
            //     self.device.destroy_command_pool(self.command_pool, None);
            // }
            for pipeline in &mut self.pipelines {
                pipeline.destroy(&self.device);
            }
            self.pipelines.clear();
            // Device must outlive all device-created objects. Surface and instance
            // are instance-level objects, so they are destroyed afterward.
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
    // Enumerate installed Vulkan-capable adapters and pick the first one that
    // has queue families and swapchain support for this surface.
    let devices =
        unsafe { instance.enumerate_physical_devices() }.map_err(|e| format!("enumerate_physical_devices: {e:?}"))?;

    for device in devices {
        if let Some(indices) = find_queue_families(instance, device, surface_loader, surface)? {
            if physical_device_supports_required_capabilities(instance, device)? {
                return Ok((device, indices));
            }
        }
    }

    Err("no suitable Vulkan device found".to_string())
}

fn physical_device_supports_required_capabilities(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
) -> Result<bool, String> {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    if properties.api_version < vk::API_VERSION_1_3 {
        return Ok(false);
    }

    // Swapchain support is mandatory because the app presents to a window.
    let extensions = unsafe { instance.enumerate_device_extension_properties(device) }
        .map_err(|e| format!("enumerate_device_extension_properties: {e:?}"))?;
    let swapchain_supported = extensions.iter().any(|prop| {
        let name = unsafe { std::ffi::CStr::from_ptr(prop.extension_name.as_ptr()) };
        name.to_bytes() == khr::swapchain::NAME.to_bytes()
    });
    if !swapchain_supported {
        return Ok(false);
    }

    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut dynamic_rendering_features);
    unsafe {
        instance.get_physical_device_features2(device, &mut features);
    }
    if dynamic_rendering_features.dynamic_rendering == vk::FALSE {
        return Ok(false);
    }

    // The HLSL compute pass writes packed RGBA bytes to an r32ui storage image,
    // then copies the size-compatible pixels to the swapchain.
    let compute_target_format = unsafe { instance.get_physical_device_format_properties(device, vk::Format::R32_UINT) };
    let required_format_features = vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::TRANSFER_SRC;
    if !compute_target_format
        .optimal_tiling_features
        .contains(required_format_features)
    {
        return Ok(false);
    }

    Ok(true)
}

fn find_queue_families(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<Option<QueueFamilyIndices>, String> {
    // Queue families describe which operations queues can perform. Presentation
    // support is surface-specific, so it must be queried separately.
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut graphics_family = None;
    let mut present_family = None;

    for (index, family) in families.iter().enumerate() {
        // The app records graphics, compute, and transfer work into one command
        // buffer submitted to this queue family.
        let required_queue_flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        if family.queue_flags.contains(required_queue_flags) {
            graphics_family = Some(index as u32);
        }
        // Present support means this queue family can hand images to this window surface.
        let present_support =
            unsafe { surface_loader.get_physical_device_surface_support(device, index as u32, surface) }
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
