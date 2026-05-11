use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=shaders/triangle.vert");
    println!("cargo:rerun-if-changed=shaders/triangle.frag");
    println!("cargo:rerun-if-changed=shaders/circle.comp");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let compiler = shaderc::Compiler::new().expect("shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("shaderc options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as u32,
    );

    compile(
        &compiler,
        &options,
        "shaders/triangle.vert",
        shaderc::ShaderKind::Vertex,
        out_dir.join("triangle.vert.spv"),
    );
    compile(
        &compiler,
        &options,
        "shaders/triangle.frag",
        shaderc::ShaderKind::Fragment,
        out_dir.join("triangle.frag.spv"),
    );
    compile(
        &compiler,
        &options,
        "shaders/circle.comp",
        shaderc::ShaderKind::Compute,
        out_dir.join("circle.comp.spv"),
    );
}

fn compile(
    compiler: &shaderc::Compiler,
    options: &shaderc::CompileOptions<'_>,
    source_path: &str,
    kind: shaderc::ShaderKind,
    output_path: PathBuf,
) {
    let source = fs::read_to_string(source_path).expect("shader source");
    let artifact = compiler
        .compile_into_spirv(&source, kind, source_path, "main", Some(options))
        .unwrap_or_else(|err| panic!("{source_path}: {err}"));
    fs::write(output_path, artifact.as_binary_u8()).expect("write shader");
}
