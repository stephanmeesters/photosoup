use std::time::Duration;

pub struct UiState {
    pub elapsed: Duration,
    pub fps: f32,
}

pub fn show(ctx: &egui::Context, state: &UiState) {
    egui::Window::new("Hello egui").show(ctx, |ui| {
        ui.label("Hello world");
        ui.horizontal(|ui| {
            ui.label(format!("Running for {:.2?}", state.elapsed));
            ui.label(format!("FPS: {:.0}", state.fps));
        });
    });
}
