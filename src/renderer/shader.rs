use ash::vk;
use rspirv_reflect::{BindingCount, DescriptorType, Reflection};
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/shader_index.rs"));

pub struct Shader {
    spirv: &'static [u8],
}

impl Shader {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let path = normalize_path(path);
        let spirv = SHADERS
            .iter()
            .find_map(|(shader_path, spirv)| (*shader_path == path).then_some(*spirv))
            .ok_or_else(|| format!("missing compiled shader artifact for {path}"))?;

        Ok(Self { spirv })
    }

    pub fn module(&self, device: &ash::Device) -> Result<vk::ShaderModule, String> {
        create_shader_module(device, self.spirv)
    }

    pub fn descriptor_set_layout_bindings(
        &self,
        set: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Result<Vec<vk::DescriptorSetLayoutBinding<'static>>, String> {
        descriptor_set_layout_bindings(self.spirv, set, stage_flags)
    }

    pub fn compute_group_size(&self) -> Result<Option<(u32, u32, u32)>, String> {
        compute_group_size(self.spirv)
    }
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

pub fn create_shader_module(
    device: &ash::Device,
    bytes: &[u8],
) -> Result<vk::ShaderModule, String> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
        .map_err(|e| format!("read_spv: {e:?}"))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);

    unsafe { device.create_shader_module(&info, None) }
        .map_err(|e| format!("create_shader_module: {e:?}"))
}

pub fn descriptor_set_layout_bindings(
    bytes: &[u8],
    set: u32,
    stage_flags: vk::ShaderStageFlags,
) -> Result<Vec<vk::DescriptorSetLayoutBinding<'static>>, String> {
    let reflection = Reflection::new_from_spirv(bytes)
        .map_err(|e| format!("reflect_shader_descriptors: {e}"))?;
    let descriptor_sets = reflection
        .get_descriptor_sets()
        .map_err(|e| format!("reflect_shader_descriptor_sets: {e}"))?;
    let Some(descriptor_set) = descriptor_sets.get(&set) else {
        return Ok(Vec::new());
    };

    descriptor_set
        .iter()
        .map(|(&binding, info)| {
            Ok(vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(to_vk_descriptor_type(info.ty)?)
                .descriptor_count(binding_count(&info.binding_count)?)
                .stage_flags(stage_flags))
        })
        .collect()
}

pub fn compute_group_size(bytes: &[u8]) -> Result<Option<(u32, u32, u32)>, String> {
    let reflection =
        Reflection::new_from_spirv(bytes).map_err(|e| format!("reflect_compute_shader: {e}"))?;
    Ok(reflection.get_compute_group_size())
}

fn to_vk_descriptor_type(ty: DescriptorType) -> Result<vk::DescriptorType, String> {
    match ty {
        DescriptorType::SAMPLER => Ok(vk::DescriptorType::SAMPLER),
        DescriptorType::COMBINED_IMAGE_SAMPLER => Ok(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        DescriptorType::SAMPLED_IMAGE => Ok(vk::DescriptorType::SAMPLED_IMAGE),
        DescriptorType::STORAGE_IMAGE => Ok(vk::DescriptorType::STORAGE_IMAGE),
        DescriptorType::UNIFORM_TEXEL_BUFFER => Ok(vk::DescriptorType::UNIFORM_TEXEL_BUFFER),
        DescriptorType::STORAGE_TEXEL_BUFFER => Ok(vk::DescriptorType::STORAGE_TEXEL_BUFFER),
        DescriptorType::UNIFORM_BUFFER => Ok(vk::DescriptorType::UNIFORM_BUFFER),
        DescriptorType::STORAGE_BUFFER => Ok(vk::DescriptorType::STORAGE_BUFFER),
        DescriptorType::UNIFORM_BUFFER_DYNAMIC => Ok(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC),
        DescriptorType::STORAGE_BUFFER_DYNAMIC => Ok(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC),
        DescriptorType::INPUT_ATTACHMENT => Ok(vk::DescriptorType::INPUT_ATTACHMENT),
        other => Err(format!("unsupported reflected descriptor type: {other:?}")),
    }
}

fn binding_count(count: &BindingCount) -> Result<u32, String> {
    match count {
        BindingCount::One => Ok(1),
        BindingCount::StaticSized(count) => (*count)
            .try_into()
            .map_err(|_| format!("descriptor binding count is too large: {count}")),
        BindingCount::Unbounded => Err("unbounded descriptor arrays are unsupported".to_string()),
    }
}
