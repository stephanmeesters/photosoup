use ash::vk;

pub struct ComputeTargets {
    // There is one offscreen image per swapchain image.
    targets: Vec<ComputeTarget>,
    // All targets match the current swapchain size.
    extent: vk::Extent2D,
}

impl ComputeTargets {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        target_count: usize,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        // Swapchains can have 2+ images. Mirroring that count avoids reusing an
        // offscreen image while its paired swapchain image is still in flight.
        let targets = (0..target_count)
            .map(|_| ComputeTarget::new(instance, device, physical_device, extent))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { targets, extent })
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn image_views(&self) -> impl Iterator<Item = vk::ImageView> + '_ {
        self.targets.iter().map(|target| target.image_view)
    }

    pub fn image(&self, image_index: usize) -> Result<vk::Image, String> {
        self.targets
            .get(image_index)
            .map(|target| target.image)
            .ok_or_else(|| format!("missing compute target for image {image_index}"))
    }

    pub fn record_prepare_for_compute(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<(), String> {
        // The compute shader needs the image in GENERAL layout with write access.
        // UNDEFINED is OK here because the previous contents do not matter: the
        // shader overwrites the image every frame.
        transition_image(
            device,
            command_buffer,
            self.image(image_index)?,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        Ok(())
    }

    pub fn record_copy_to_swapchain(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        swapchain_image: vk::Image,
    ) -> Result<(), String> {
        let target_image = self.image(image_index)?;

        // Make shader writes visible to transfer reads and switch the offscreen
        // image into the layout required by cmd_copy_image as a source.
        transition_image(
            device,
            command_buffer,
            target_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        );

        // The swapchain image was just acquired. Its old contents are irrelevant
        // because we overwrite it with the compute output before rendering overlays.
        transition_image(
            device,
            command_buffer,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        // Copy the whole 2D color image from the compute target into the acquired
        // swapchain image. Both images must already be in transfer-compatible layouts.
        let copy_region = vk::ImageCopy::default()
            .src_subresource(color_subresource_layers())
            .dst_subresource(color_subresource_layers())
            .extent(vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            });

        unsafe {
            // This records the copy; the GPU performs it later when the command
            // buffer is submitted to the queue.
            device.cmd_copy_image(
                command_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }

        // Dynamic rendering will use the swapchain image as a color attachment,
        // so make the transfer write visible to color attachment work.
        transition_image(
            device,
            command_buffer,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        Ok(())
    }

    pub fn destroy(self, device: &ash::Device) {
        for target in self.targets {
            target.destroy(device);
        }
    }
}

struct ComputeTarget {
    // Raw image storage the compute shader writes into.
    image: vk::Image,
    // Image views describe how shaders/descriptors interpret image subresources.
    image_view: vk::ImageView,
    // Device-local allocation bound to image.
    memory: vk::DeviceMemory,
}

impl ComputeTarget {
    fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        // STORAGE lets the compute shader write the image. TRANSFER_SRC lets us
        // copy it into the swapchain after compute finishes.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_UINT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None) }
            .map_err(|e| format!("create_compute_target_image: {e:?}"))?;
        // Vulkan image creation does not allocate memory. Query requirements,
        // choose a compatible memory type, allocate, then bind.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type_index = find_memory_type(
            instance,
            physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| format!("allocate_compute_target_memory: {e:?}"))?;

        unsafe {
            // Binding attaches the allocation to the image handle. Offset 0 is
            // valid because the whole allocation was sized for this image.
            device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| format!("bind_compute_target_memory: {e:?}"))?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R32_UINT)
            .subresource_range(color_subresource_range());
        let image_view = unsafe { device.create_image_view(&view_info, None) }
            .map_err(|e| format!("create_compute_target_image_view: {e:?}"))?;

        Ok(Self {
            image,
            image_view,
            memory,
        })
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    // type_filter is a bitmask from Vulkan saying which memory types can back
    // this resource. We additionally require DEVICE_LOCAL for fast GPU access.
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..memory_properties.memory_type_count {
        let supported = (type_filter & (1 << i)) != 0;
        let has_properties = memory_properties.memory_types[i as usize]
            .property_flags
            .contains(properties);
        if supported && has_properties {
            return Ok(i);
        }
    }

    Err("no suitable memory type for compute target image".to_string())
}

fn color_subresource_layers() -> vk::ImageSubresourceLayers {
    // ImageCopy uses layers to identify the color aspect/layer of source and destination.
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    // Barriers and image views use ranges to identify which mip levels/layers they affect.
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
) {
    // A pipeline barrier is both a layout transition and a memory dependency:
    // it orders earlier writes/reads before later reads/writes across pipeline stages.
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_subresource_range())
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    unsafe {
        // This records the barrier into the command buffer. The source/destination
        // stage masks define where the GPU must wait and where work may continue.
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
}
