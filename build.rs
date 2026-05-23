use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=DXC");

    let shader_out_dir = out_dir.join("shaders");
    fs::create_dir_all(&shader_out_dir).expect("create shader output dir");
    let dxc = env::var_os("DXC").unwrap_or_else(|| OsString::from("dxc"));

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
            compile_shader(&dxc, &shader, &output_path);
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
        } else if shader_profile(&path).is_some() {
            shaders.push(path);
        }
    }
    Ok(())
}

fn shader_profile(path: &Path) -> Option<&'static str> {
    match path.file_name().and_then(|file_name| file_name.to_str()) {
        Some(file_name) if file_name.ends_with(".vert.hlsl") => Some("vs_6_6"),
        Some(file_name) if file_name.ends_with(".frag.hlsl") => Some("ps_6_6"),
        Some(file_name) if file_name.ends_with(".cs.hlsl") => Some("cs_6_6"),
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

fn compile_shader(dxc: &OsString, source_path: &Path, output_path: &Path) {
    let profile = shader_profile(source_path).unwrap_or_else(|| {
        panic!(
            "unsupported shader filename for {}; expected .vert.hlsl, .frag.hlsl, or .cs.hlsl",
            source_path.display()
        )
    });
    let output = Command::new(dxc)
        .arg("-spirv")
        .arg("-T")
        .arg(profile)
        .arg("-E")
        .arg("main")
        .arg("-Fo")
        .arg(output_path)
        .arg(source_path)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run dxc for {}: {err}. Install dxc or set DXC to its path",
                source_path.display()
            )
        });

    if !output.status.success() {
        panic!(
            "dxc failed for {}\nstdout:\n{}\nstderr:\n{}",
            source_path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
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
