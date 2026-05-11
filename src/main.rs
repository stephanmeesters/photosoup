use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = photosoup::app::App::default();
    event_loop.run_app(&mut app).expect("run app");
}
