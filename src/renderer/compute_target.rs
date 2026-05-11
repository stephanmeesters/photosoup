use ash::vk;

pub struct ComputeTargets {
    targets: Vec<ComputeTarget>,
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

        let copy_region = vk::ImageCopy::default()
            .src_subresource(color_subresource_layers())
            .dst_subresource(color_subresource_layers())
            .extent(vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_image(
                command_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }

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
    image: vk::Image,
    image_view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl ComputeTarget {
    fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
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
            device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| format!("bind_compute_target_memory: {e:?}"))?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
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
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

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
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
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
