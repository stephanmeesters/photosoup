use ash::vk;
use std::ffi::CString;

use super::{
    compute_target::ComputeTargets,
    pipeline::{FrameContext, Pipeline, SwapchainContext},
};

pub struct ComputeCirclePass {
    pipeline: ComputeCirclePipeline,
    targets: Option<ComputeTargets>,
}

impl ComputeCirclePass {
    pub fn new(device: &ash::Device) -> Result<Self, String> {
        Ok(Self {
            pipeline: ComputeCirclePipeline::new(device)?,
            targets: None,
        })
    }
}

impl Pipeline for ComputeCirclePass {
    fn on_swapchain_created(&mut self, ctx: &SwapchainContext<'_>) -> Result<(), String> {
        let targets = ComputeTargets::new(
            ctx.instance,
            ctx.device,
            ctx.physical_device,
            ctx.swapchain_images.len(),
            ctx.extent,
        )?;
        self.pipeline.recreate_descriptors(ctx.device, &targets)?;
        self.targets = Some(targets);
        Ok(())
    }

    fn destroy_swapchain(&mut self, device: &ash::Device) {
        self.pipeline.destroy_descriptors(device);
        if let Some(targets) = self.targets.take() {
            targets.destroy(device);
        }
    }

    fn record_before_render_pass(&mut self, ctx: &FrameContext<'_>) -> Result<(), String> {
        let targets = self
            .targets
            .as_ref()
            .ok_or_else(|| "missing compute targets".to_string())?;
        self.pipeline
            .record(ctx.device, ctx.command_buffer, targets, ctx.image_index)?;
        targets.record_copy_to_swapchain(
            ctx.device,
            ctx.command_buffer,
            ctx.image_index,
            ctx.swapchain_image,
        )
    }

    fn destroy(&mut self, device: &ash::Device) {
        self.destroy_swapchain(device);
        self.pipeline.destroy(device);
    }
}

pub struct ComputeCirclePipeline {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
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
        })
    }

    pub fn recreate_descriptors(
        &mut self,
        device: &ash::Device,
        targets: &ComputeTargets,
    ) -> Result<(), String> {
        self.destroy_descriptors(device);

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(targets.len() as u32);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(targets.len() as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));

        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("create_compute_descriptor_pool: {e:?}"))?;

        let layouts = vec![self.descriptor_set_layout; targets.len()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| format!("allocate_compute_descriptor_sets: {e:?}"))?;

        for (&descriptor_set, image_view) in self.descriptor_sets.iter().zip(targets.image_views())
        {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(image_view)
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
        targets: &ComputeTargets,
        image_index: usize,
    ) -> Result<(), String> {
        let descriptor_set = *self
            .descriptor_sets
            .get(image_index)
            .ok_or_else(|| format!("missing compute descriptor set for image {image_index}"))?;

        targets.record_prepare_for_compute(device, command_buffer, image_index)?;
        let extent = targets.extent();

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

        Ok(())
    }

    pub fn destroy_descriptors(&mut self, device: &ash::Device) {
        self.descriptor_sets.clear();
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            self.descriptor_pool = vk::DescriptorPool::null();
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.destroy_descriptors(device);
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
