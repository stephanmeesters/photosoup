use std::time::Duration;

pub struct UiState {
    pub elapsed: Duration,
    pub fps: f32,
}

pub fn show(ctx: &egui::Context, state: &UiState) {
    egui::Window::new("Hello egui")
        .resizable(true)
        .default_size([340.0, 180.0])
        .show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                ui.set_min_size(ui.available_size());
                ui.label("Hello world");
                ui.horizontal(|ui| {
                    ui.label(format!("Running for {:.2?}", state.elapsed));
                    ui.label(format!("FPS: {:.0}", state.fps));
                });
            });
        });
}
