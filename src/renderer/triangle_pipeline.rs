use ash::vk;
use std::ffi::CString;

use super::{
    pipeline::{Pipeline, RenderingContext, SwapchainContext},
    shader::Shader,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct TriangleVertex {
    // Matches location 0 in triangle.vert.hlsl.
    position: [f32; 2],
    // Matches location 1 in triangle.vert.hlsl.
    color: [f32; 4],
}

const TRIANGLE_VERTICES: [TriangleVertex; 3] = [
    TriangleVertex {
        position: [0.5, -0.45],
        color: [0.2, 0.4, 1.0, 1.0],
    },
    TriangleVertex {
        position: [0.0, 0.55],
        color: [0.1, 0.9, 0.3, 1.0],
    },
    TriangleVertex {
        position: [-0.5, -0.45],
        color: [1.0, 0.2, 0.1, 1.0],
    },
];

#[derive(Default)]
pub struct TrianglePass {
    // Graphics pipelines depend on the swapchain format and extent, so this is
    // created/destroyed with the swapchain.
    pipeline: Option<TrianglePipeline>,
}

impl Pipeline for TrianglePass {
    fn on_swapchain_created(&mut self, ctx: &SwapchainContext<'_>) -> Result<(), String> {
        self.pipeline = Some(TrianglePipeline::new(
            ctx.instance,
            ctx.device,
            ctx.physical_device,
            ctx.color_attachment_format,
            ctx.extent,
        )?);
        Ok(())
    }

    fn destroy_swapchain(&mut self, device: &ash::Device) {
        if let Some(mut pipeline) = self.pipeline.take() {
            pipeline.destroy(device);
        }
    }

    fn record_rendering(&mut self, ctx: &RenderingContext<'_>) -> Result<(), String> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            pipeline.record(ctx.device, ctx.command_buffer);
        }
        Ok(())
    }

    fn destroy(&mut self, device: &ash::Device) {
        self.destroy_swapchain(device);
    }
}

pub struct TrianglePipeline {
    // Describes descriptor sets/push constants. The triangle shader uses neither,
    // but Vulkan still requires a pipeline layout object.
    layout: vk::PipelineLayout,
    // Opaque graphics pipeline containing shaders plus fixed-function state.
    pipeline: vk::Pipeline,
    // GPU buffer that stores the three TriangleVertex values.
    vertex_buffer: vk::Buffer,
    // Host-visible memory allocation bound to vertex_buffer.
    vertex_buffer_memory: vk::DeviceMemory,
}

impl TrianglePipeline {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        color_attachment_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        let vert_shader = Shader::load("shaders/triangle.vert.hlsl")?;
        let frag_shader = Shader::load("shaders/triangle.frag.hlsl")?;

        // Shader modules are temporary inputs to pipeline creation.
        let vert_shader_module = vert_shader.module(device)?;
        let frag_shader_module = frag_shader.module(device).map_err(|e| {
            unsafe {
                device.destroy_shader_module(vert_shader_module, None);
            }
            e
        })?;
        let (vertex_buffer, vertex_buffer_memory) =
            create_vertex_buffer(instance, device, physical_device)?;

        // Build the final graphics pipeline using the shader modules and vertex buffer.
        let result = Self::create(
            device,
            color_attachment_format,
            extent,
            vert_shader_module,
            frag_shader_module,
            vertex_buffer,
            vertex_buffer_memory,
        );

        unsafe {
            // Once the pipeline is created, Vulkan no longer needs the shader modules.
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        if result.is_err() {
            unsafe {
                device.destroy_buffer(vertex_buffer, None);
                device.free_memory(vertex_buffer_memory, None);
            }
        }

        result
    }

    fn create(
        device: &ash::Device,
        color_attachment_format: vk::Format,
        extent: vk::Extent2D,
        vert_shader_module: vk::ShaderModule,
        frag_shader_module: vk::ShaderModule,
        vertex_buffer: vk::Buffer,
        vertex_buffer_memory: vk::DeviceMemory,
    ) -> Result<Self, String> {
        let entry_name = CString::new("main").unwrap();
        // A graphics pipeline has one stage per shader. The entry name must match
        // the HLSL function compiled into SPIR-V.
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

        // Binding 0 says each vertex comes from one tightly packed TriangleVertex
        // and advances once per vertex.
        let vertex_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<TriangleVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        // Attribute descriptions map bytes in TriangleVertex to shader locations.
        let vertex_attributes = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(size_of::<[f32; 2]>() as u32),
        ];
        // Vertex input state connects vertex buffers to shader inputs.
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding))
            .vertex_attribute_descriptions(&vertex_attributes);
        // TRIANGLE_LIST consumes vertices in groups of three. Primitive restart
        // only matters for strip topologies, so it is disabled.
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        // Viewport maps normalized device coordinates to framebuffer pixels.
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        // Scissor clips rendering to the whole framebuffer.
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));
        // Rasterizer turns post-vertex triangles into fragments.
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);
        // No MSAA here: every pixel has one color sample.
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Allow all RGBA channels to be written and disable blending, so fragment
        // shader output replaces the existing color where the triangle draws.
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
        // Empty pipeline layout because this shader has no descriptors/push constants.
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .map_err(|e| format!("create_pipeline_layout: {e:?}"))?;

        // Dynamic rendering pipelines declare the attachment formats they will
        // render into instead of baking in a VkRenderPass/subpass.
        let color_attachment_formats = [color_attachment_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            // Pipeline creation may compile/link GPU-specific machine code. A
            // pipeline cache could speed this up, but this app passes null.
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, e)| {
            unsafe {
                device.destroy_pipeline_layout(layout, None);
            }
            format!("create_graphics_pipelines: {e:?}")
        })?[0];

        Ok(Self {
            layout,
            pipeline,
            vertex_buffer,
            vertex_buffer_memory,
        })
    }

    pub fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0];
        unsafe {
            // Select this graphics pipeline for later draw commands.
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            // Bind vertex buffer slot 0 so the vertex shader receives TriangleVertex data.
            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            // Draw 3 vertices, 1 instance, starting at vertex 0 and instance 0.
            device.cmd_draw(command_buffer, TRIANGLE_VERTICES.len() as u32, 1, 0, 0);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.layout, None);
                self.layout = vk::PipelineLayout::null();
            }
            if self.vertex_buffer != vk::Buffer::null() {
                device.destroy_buffer(self.vertex_buffer, None);
                self.vertex_buffer = vk::Buffer::null();
            }
            if self.vertex_buffer_memory != vk::DeviceMemory::null() {
                device.free_memory(self.vertex_buffer_memory, None);
                self.vertex_buffer_memory = vk::DeviceMemory::null();
            }
        }
    }
}

impl Drop for TrianglePipeline {
    fn drop(&mut self) {
        debug_assert_eq!(self.pipeline, vk::Pipeline::null());
        debug_assert_eq!(self.layout, vk::PipelineLayout::null());
        debug_assert_eq!(self.vertex_buffer, vk::Buffer::null());
        debug_assert_eq!(self.vertex_buffer_memory, vk::DeviceMemory::null());
    }
}

fn create_vertex_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let buffer_size = size_of_val(&TRIANGLE_VERTICES) as vk::DeviceSize;
    // VERTEX_BUFFER means this buffer can be bound with cmd_bind_vertex_buffers.
    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| format!("create_triangle_vertex_buffer: {e:?}"))?;
    // Buffers also need separately allocated memory in Vulkan.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(
        instance,
        physical_device,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .map_err(|e| {
        unsafe {
            device.destroy_buffer(buffer, None);
        }
        e
    })?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
        unsafe {
            device.destroy_buffer(buffer, None);
        }
        format!("allocate_triangle_vertex_buffer_memory: {e:?}")
    })?;

    unsafe {
        // Attach the allocation to the buffer so the buffer has real backing memory.
        device.bind_buffer_memory(buffer, memory, 0).map_err(|e| {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
            format!("bind_triangle_vertex_buffer_memory: {e:?}")
        })?;
        // Map exposes GPU memory to the CPU. HOST_COHERENT means writes become
        // visible to the GPU without an explicit flush.
        let data = device
            .map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            .map_err(|e| {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                format!("map_triangle_vertex_buffer_memory: {e:?}")
            })?;
        // Copy the Rust vertex array bytes into the mapped Vulkan allocation.
        std::ptr::copy_nonoverlapping(
            TRIANGLE_VERTICES.as_ptr().cast::<u8>(),
            data.cast::<u8>(),
            buffer_size as usize,
        );
        // Unmapping ends CPU access. The buffer still owns the uploaded bytes.
        device.unmap_memory(memory);
    }

    Ok((buffer, memory))
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    // Vulkan returns a bitmask of compatible memory types for the buffer. We pick
    // one whose properties allow CPU mapping and coherent visibility.
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

    Err("no suitable memory type for triangle vertex buffer".to_string())
}
