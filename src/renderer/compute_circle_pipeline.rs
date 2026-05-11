use ash::vk;
use std::ffi::CString;

pub struct ComputeCirclePipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    targets: Vec<ComputeTarget>,
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

impl ComputeCirclePipeline {
    pub fn new(device: &ash::Device) -> Result<Self, String> {
        let descriptor_set_layout = create_descriptor_set_layout(device)?;
        let pipeline_layout = create_pipeline_layout(device, descriptor_set_layout)?;
        let pipeline = create_pipeline(device, pipeline_layout)?;

        Ok(Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            targets: Vec::new(),
        })
    }

    pub fn recreate_targets(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        target_count: usize,
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.destroy_targets(device);

        self.targets = (0..target_count)
            .map(|_| ComputeTarget::new(instance, device, physical_device, extent))
            .collect::<Result<Vec<_>, _>>()?;

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(self.targets.len() as u32);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.targets.len() as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));

        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("create_compute_descriptor_pool: {e:?}"))?;

        let layouts = vec![self.descriptor_set_layout; self.targets.len()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| format!("allocate_compute_descriptor_sets: {e:?}"))?;

        for (&descriptor_set, target) in self.descriptor_sets.iter().zip(&self.targets) {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(target.image_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info));

            unsafe {
                device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }
        }

        Ok(())
    }

    pub fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        swapchain_image: vk::Image,
        image_index: usize,
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        let descriptor_set = *self
            .descriptor_sets
            .get(image_index)
            .ok_or_else(|| format!("missing compute descriptor set for image {image_index}"))?;
        let target = self
            .targets
            .get(image_index)
            .ok_or_else(|| format!("missing compute target for image {image_index}"))?;

        transition_image(
            device,
            command_buffer,
            target.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            device.cmd_dispatch(
                command_buffer,
                extent.width.div_ceil(16),
                extent.height.div_ceil(16),
                1,
            );
        }

        transition_image(
            device,
            command_buffer,
            target.image,
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
                width: extent.width,
                height: extent.height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_image(
                command_buffer,
                target.image,
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

    pub fn destroy_targets(&mut self, device: &ash::Device) {
        self.descriptor_sets.clear();
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            self.descriptor_pool = vk::DescriptorPool::null();
        }
        for target in self.targets.drain(..) {
            target.destroy(device);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.destroy_targets(device);
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
                self.descriptor_set_layout = vk::DescriptorSetLayout::null();
            }
        }
    }
}

impl Drop for ComputeCirclePipeline {
    fn drop(&mut self) {
        debug_assert_eq!(self.descriptor_pool, vk::DescriptorPool::null());
        debug_assert_eq!(self.pipeline, vk::Pipeline::null());
        debug_assert_eq!(self.pipeline_layout, vk::PipelineLayout::null());
        debug_assert_eq!(self.descriptor_set_layout, vk::DescriptorSetLayout::null());
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, String> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);
    let info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));

    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("create_compute_descriptor_set_layout: {e:?}"))
}

fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let layouts = [descriptor_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

    unsafe { device.create_pipeline_layout(&info, None) }
        .map_err(|e| format!("create_compute_pipeline_layout: {e:?}"))
}

fn create_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, String> {
    let shader_module = create_shader_module(
        device,
        include_bytes!(concat!(env!("OUT_DIR"), "/circle.comp.spv")),
    )?;
    let entry_name = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .module(shader_module)
        .name(&entry_name)
        .stage(vk::ShaderStageFlags::COMPUTE);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let result = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    }
    .map_err(|(_, e)| format!("create_compute_pipeline: {e:?}"))
    .map(|pipelines| pipelines[0]);

    unsafe {
        device.destroy_shader_module(shader_module, None);
    }

    result
}

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule, String> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
        .map_err(|e| format!("read_spv: {e:?}"))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);

    unsafe { device.create_shader_module(&info, None) }
        .map_err(|e| format!("create_compute_shader_module: {e:?}"))
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
