use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    println!("cargo:rerun-if-changed=shaders");

    let shader_out_dir = out_dir.join("shaders");
    fs::create_dir_all(&shader_out_dir).expect("create shader output dir");

    let compiler = shaderc::Compiler::new().expect("shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("shaderc options");
    options.set_source_language(shaderc::SourceLanguage::HLSL);
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as u32,
    );

    let shaders = discover_shaders(Path::new("shaders")).expect("discover shaders");
    let mut entries = Vec::new();
    for shader in shaders {
        println!("cargo:rerun-if-changed={}", shader.display());
        let output_path = shader_artifact_path(&shader_out_dir, &shader);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).expect("create shader artifact parent dir");
        }
        if shader_needs_compile(&shader, &output_path).expect("check shader artifact freshness") {
            println!("compiling shader {}", shader.display());
            compile_shader(&compiler, &options, &shader, &output_path);
        }

        entries.push((normalize_path(&shader), output_path));
    }

    write_shader_index(&out_dir.join("shader_index.rs"), &entries).expect("write shader index");
}

fn discover_shaders(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut shaders = Vec::new();
    discover_shaders_inner(dir, &mut shaders)?;
    shaders.sort();
    Ok(shaders)
}

fn discover_shaders_inner(dir: &Path, shaders: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            discover_shaders_inner(&path, shaders)?;
        } else if shader_kind(&path).is_some() {
            shaders.push(path);
        }
    }
    Ok(())
}

fn shader_kind(path: &Path) -> Option<shaderc::ShaderKind> {
    match path.file_name().and_then(|file_name| file_name.to_str()) {
        Some(file_name) if file_name.ends_with(".vert.hlsl") => Some(shaderc::ShaderKind::Vertex),
        Some(file_name) if file_name.ends_with(".frag.hlsl") => Some(shaderc::ShaderKind::Fragment),
        Some(file_name) if file_name.ends_with(".comp.hlsl") => Some(shaderc::ShaderKind::Compute),
        _ => None,
    }
}

fn shader_artifact_path(shader_out_dir: &Path, shader: &Path) -> PathBuf {
    let output_name = format!("{}.spv", shader.strip_prefix("shaders").unwrap().display());
    shader_out_dir.join(output_name)
}

fn shader_needs_compile(source_path: &Path, output_path: &Path) -> Result<bool, std::io::Error> {
    let Ok(output_metadata) = fs::metadata(output_path) else {
        return Ok(true);
    };

    let source_modified = fs::metadata(source_path)?.modified()?;
    let output_modified = output_metadata.modified()?;
    Ok(source_modified > output_modified)
}

fn compile_shader(
    compiler: &shaderc::Compiler,
    options: &shaderc::CompileOptions<'_>,
    source_path: &Path,
    output_path: &Path,
) {
    let kind = shader_kind(source_path).unwrap_or_else(|| {
        panic!(
            "unsupported shader filename for {}; expected .vert.hlsl, .frag.hlsl, or .comp.hlsl",
            source_path.display()
        )
    });
    let source = fs::read_to_string(source_path).expect("shader source");
    let artifact = compiler
        .compile_into_spirv(
            &source,
            kind,
            &source_path.to_string_lossy(),
            "main",
            Some(options),
        )
        .unwrap_or_else(|err| panic!("{}: {err}", source_path.display()));
    fs::write(output_path, artifact.as_binary_u8()).expect("write shader artifact");
}

fn normalize_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn write_shader_index(path: &Path, entries: &[(String, PathBuf)]) -> Result<(), std::io::Error> {
    let mut source = String::from("pub static SHADERS: &[(&str, &[u8])] = &[\n");
    for (shader_path, artifact_path) in entries {
        source.push_str("    (");
        source.push_str(&format!("{shader_path:?}"));
        source.push_str(", include_bytes!(");
        source.push_str(&format!("{:?}", artifact_path.display().to_string()));
        source.push_str(")),\n");
    }
    source.push_str("];\n");

    if fs::read_to_string(path).is_ok_and(|existing| existing == source) {
        return Ok(());
    }

    fs::write(path, source)
}
