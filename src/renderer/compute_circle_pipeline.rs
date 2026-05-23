use ash::vk;
use std::ffi::CString;

use super::{
    compute_target::ComputeTargets,
    pipeline::{FrameContext, Pipeline, SwapchainContext},
    shader::Shader,
};

pub struct ComputeCirclePass {
    // Permanent compute pipeline state: shader, descriptor layout, and pipeline layout.
    pipeline: ComputeCirclePipeline,
    // Swapchain-sized storage images that the compute shader writes into.
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
        // One compute target per swapchain image keeps each acquired image paired
        // with a matching offscreen image of the same size.
        let targets = ComputeTargets::new(
            ctx.instance,
            ctx.device,
            ctx.physical_device,
            ctx.swapchain_images.len(),
            ctx.extent,
        )?;
        // Descriptor sets contain the image views that the compute shader writes.
        // They depend on the target images, so they are rebuilt with the swapchain.
        self.pipeline.recreate_descriptors(ctx.device, &targets)?;
        self.targets = Some(targets);
        Ok(())
    }

    fn destroy_swapchain(&mut self, device: &ash::Device) {
        // Descriptor sets reference target image views, so destroy descriptors
        // before destroying the images they point at.
        self.pipeline.destroy_descriptors(device);
        if let Some(targets) = self.targets.take() {
            targets.destroy(device);
        }
    }

    fn record_before_rendering(&mut self, ctx: &FrameContext<'_>) -> Result<(), String> {
        let targets = self
            .targets
            .as_ref()
            .ok_or_else(|| "missing compute targets".to_string())?;
        // The compute pass runs before graphics rendering. It writes a full-screen
        // image, then copies that image into the acquired swapchain image.
        self.pipeline
            .record(ctx.device, ctx.command_buffer, targets, ctx.image_index)?;
        targets.record_copy_to_swapchain(ctx.device, ctx.command_buffer, ctx.image_index, ctx.swapchain_image)
    }

    fn destroy(&mut self, device: &ash::Device) {
        self.destroy_swapchain(device);
        self.pipeline.destroy(device);
    }
}

pub struct ComputeCirclePipeline {
    // Describes the set/binding slots the compute shader can access.
    descriptor_set_layout: vk::DescriptorSetLayout,
    // Reflected binding metadata kept so descriptor pools/updates match the shader.
    descriptor_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    // Connects descriptor set layouts and push constants to the pipeline.
    pipeline_layout: vk::PipelineLayout,
    // Opaque Vulkan compute pipeline object.
    pipeline: vk::Pipeline,
    // Pool that owns descriptor set allocation storage.
    descriptor_pool: vk::DescriptorPool,
    // One descriptor set per swapchain image/compute target.
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl ComputeCirclePipeline {
    pub fn new(device: &ash::Device) -> Result<Self, String> {
        let shader = Shader::load("shaders/circle.cs.hlsl")?;
        // Build the descriptor layout from SPIR-V reflection instead of manually
        // duplicating HLSL register bindings in Rust.
        let (descriptor_set_layout, descriptor_bindings) = create_descriptor_set_layout(device, &shader)?;
        let pipeline_layout = create_pipeline_layout(device, descriptor_set_layout)?;
        let pipeline = create_pipeline(device, pipeline_layout, &shader)?;

        Ok(Self {
            descriptor_set_layout,
            descriptor_bindings,
            pipeline_layout,
            pipeline,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
        })
    }

    pub fn recreate_descriptors(&mut self, device: &ash::Device, targets: &ComputeTargets) -> Result<(), String> {
        self.destroy_descriptors(device);

        // Descriptor pools need capacities per descriptor type, not just a total
        // set count. Multiply each binding by target count because every target
        // gets its own descriptor set.
        let pool_sizes = self
            .descriptor_bindings
            .iter()
            .map(|binding| {
                vk::DescriptorPoolSize::default()
                    .ty(binding.descriptor_type)
                    .descriptor_count(binding.descriptor_count * targets.len() as u32)
            })
            .collect::<Vec<_>>();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(targets.len() as u32)
            .pool_sizes(&pool_sizes);

        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("create_compute_descriptor_pool: {e:?}"))?;

        // Allocate identical set layouts, one set per swapchain/target image.
        let layouts = vec![self.descriptor_set_layout; targets.len()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        self.descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| format!("allocate_compute_descriptor_sets: {e:?}"))?;

        let image_binding = self
            .descriptor_bindings
            .iter()
            .find(|binding| binding.descriptor_type == vk::DescriptorType::STORAGE_IMAGE)
            .ok_or_else(|| "circle compute shader has no reflected storage image".to_string())?;

        for (&descriptor_set, image_view) in self.descriptor_sets.iter().zip(targets.image_views()) {
            // A storage image descriptor gives the shader write access to an image view.
            // GENERAL is the layout storage images use for unordered shader writes.
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(image_binding.binding)
                .descriptor_type(image_binding.descriptor_type)
                .image_info(std::slice::from_ref(&image_info));

            unsafe {
                // Descriptor updates happen on the CPU and bake image handles into
                // the descriptor set. They are not command-buffer operations.
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

        // Put the target image into a layout and access state that the compute
        // shader can legally write to.
        targets.record_prepare_for_compute(device, command_buffer, image_index)?;
        let extent = targets.extent();

        unsafe {
            // Select the compute pipeline for subsequent compute commands.
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            // Bind the descriptor set containing this frame's storage image.
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            // Dispatch workgroups. The shader local size is 16x16, so div_ceil
            // covers the whole image even when the extent is not divisible by 16.
            device.cmd_dispatch(command_buffer, extent.width.div_ceil(16), extent.height.div_ceil(16), 1);
        }

        Ok(())
    }

    pub fn destroy_descriptors(&mut self, device: &ash::Device) {
        // Descriptor sets are implicitly freed when their pool is destroyed.
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

fn create_descriptor_set_layout(
    device: &ash::Device,
    shader: &Shader,
) -> Result<(vk::DescriptorSetLayout, Vec<vk::DescriptorSetLayoutBinding<'static>>), String> {
    // This layout tells Vulkan which resources set 0 contains. Pipeline creation
    // will fail if the shader's descriptors are incompatible with this layout.
    let bindings = shader.descriptor_set_layout_bindings(0, vk::ShaderStageFlags::COMPUTE)?;
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    let layout = unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("create_compute_descriptor_set_layout: {e:?}"))?;

    Ok((layout, bindings))
}

fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, String> {
    let layouts = [descriptor_set_layout];
    // Pipeline layout is the complete interface between command buffers and the
    // shader: descriptor set layouts plus push-constant ranges.
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

    unsafe { device.create_pipeline_layout(&info, None) }.map_err(|e| format!("create_compute_pipeline_layout: {e:?}"))
}

fn create_pipeline(
    device: &ash::Device,
    pipeline_layout: vk::PipelineLayout,
    shader: &Shader,
) -> Result<vk::Pipeline, String> {
    // The Rust dispatch math assumes a 16x16x1 local size. Check the compiled
    // shader so an HLSL edit cannot silently desync CPU and GPU assumptions.
    let group_size = shader
        .compute_group_size()?
        .ok_or_else(|| "circle compute shader is missing local size".to_string())?;
    if group_size != (16, 16, 1) {
        return Err(format!(
            "circle compute shader local size must be (16, 16, 1), got {group_size:?}"
        ));
    }

    let shader_module = shader.module(device)?;
    let entry_name = CString::new("main").unwrap();
    // A compute pipeline has one shader stage: the compute shader entry point.
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .module(shader_module)
        .name(&entry_name)
        .stage(vk::ShaderStageFlags::COMPUTE);
    // Pipeline creation compiles/links the shader with the declared resource
    // interface into a GPU-executable pipeline object.
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let result =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None) }
            .map_err(|(_, e)| format!("create_compute_pipeline: {e:?}"))
            .map(|pipelines| pipelines[0]);

    unsafe {
        // The pipeline keeps what it needs internally; the temporary shader module
        // can be destroyed after pipeline creation.
        device.destroy_shader_module(shader_module, None);
    }

    result
}
