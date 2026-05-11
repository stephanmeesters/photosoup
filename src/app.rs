use crate::{
    renderer::{EguiFrame, Renderer, RendererError},
    ui::{self, UiState},
};
use egui_winit::State as EguiWinitState;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};

#[derive(Default)]
pub struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    egui_ctx: egui::Context,
    egui_state: Option<EguiWinitState>,
    start_time: Option<Instant>,
    last_frame_time: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("photosoup - Vulkan triangle")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .with_resizable(true);
        let window = event_loop.create_window(attrs).expect("create window");

        let renderer = Renderer::new(&window).expect("create renderer");
        let egui_state = EguiWinitState::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );

        self.start_time = Some(Instant::now());
        self.last_frame_time = None;
        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        if let Some(egui_state) = self.egui_state.as_mut() {
            let _ = egui_state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(egui_state), Some(start_time)) = (
                    self.renderer.as_mut(),
                    self.egui_state.as_mut(),
                    self.start_time,
                ) {
                    let now = Instant::now();
                    let frame_dt = self
                        .last_frame_time
                        .replace(now)
                        .map(|previous| now.saturating_duration_since(previous))
                        .unwrap_or_default();
                    let fps = if frame_dt.is_zero() {
                        0.0
                    } else {
                        1.0 / frame_dt.as_secs_f32()
                    };
                    let elapsed = now.saturating_duration_since(start_time);
                    let raw_input = egui_state.take_egui_input(window);
                    let ui_state = UiState { elapsed, fps };
                    let full_output = self.egui_ctx.run(raw_input, |ctx| {
                        ui::show(ctx, &ui_state);
                    });
                    egui_state.handle_platform_output(window, full_output.platform_output);

                    let clipped_primitives = self
                        .egui_ctx
                        .tessellate(full_output.shapes, full_output.pixels_per_point);

                    let egui_frame = EguiFrame {
                        clipped_primitives,
                        textures_delta: full_output.textures_delta,
                        pixels_per_point: full_output.pixels_per_point,
                    };

                    if let Err(err) = renderer.draw_frame(Some(egui_frame)) {
                        match err {
                            RendererError::OutOfDate => renderer.recreate_swapchain(),
                            RendererError::Fatal(message) => {
                                eprintln!("{message}");
                                event_loop.exit();
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
