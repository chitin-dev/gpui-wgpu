//! A testing harness built around one app and the windows opened on it.
//!
//! [`TestApp`] is the third of GPUI's test harnesses and the simplest to write
//! a UI test against. [`TestAppContext`](crate::TestAppContext) is the oldest
//! and the most general: it exists to serve `#[gpui::test]`, and its update
//! entry points do not flush effects, so a test that calls `update` twice in a
//! row has to know which of the two flushes and which does not.
//! [`VisualTestContext`](crate::VisualTestContext) narrows that to a single
//! window. `TestApp` sits beside them: it owns the app outright, flushes after
//! every `update`, and hands back a window handle the test can drive directly,
//! so the common shape — build a view, poke it, draw, assert on what was
//! painted — reads as those four steps and nothing else.
//!
//! # Example
//! ```ignore
//! #[test]
//! fn test_my_view() {
//!     let mut app = TestApp::new();
//!
//!     let mut window = app.open_window(|window, cx| {
//!         MyView::new(window, cx)
//!     });
//!
//!     window.update(|view, window, cx| {
//!         view.do_something(cx);
//!     });
//!
//!     // Check rendered state
//!     window.read(|view, cx| {
//!         assert_eq!(view.count, 1);
//!     });
//! }
//! ```

use crate::{
    AnyWindowHandle, App, AppCell, AppContext as _, AssetSource, AsyncApp, BackgroundExecutor,
    BorrowAppContext as _, Bounds, ClipboardItem, Context, Entity, ForegroundExecutor, Global,
    InputEvent, Keystroke, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    Platform, PlatformTextSystem, Point, Render, ScrollDelta, ScrollWheelEvent, Size, Task,
    TestDispatcher,
    TestPlatform, TestWindow, TextSystem, TouchPhase, Window, WindowBounds, WindowHandle,
    WindowOptions, app::GpuiMode,
};
use rand::{SeedableRng, rngs::StdRng};
use std::{future::Future, rc::Rc, sync::Arc, time::Duration};

/// A test application: one app, its platform, and the windows opened on it.
///
/// Unlike [`TestAppContext`](crate::TestAppContext), `TestApp` automatically
/// flushes effects after each update and provides simpler window management.
/// It is not created for you by `#[gpui::test]`; a test that wants it builds
/// one, because the harness it replaces is chosen per test rather than by the
/// macro.
pub struct TestApp {
    app: Rc<AppCell>,
    platform: Rc<TestPlatform>,
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    /// Kept alive for its side effects only.
    ///
    /// The executors hold their own `Arc` of the same dispatcher, so nothing
    /// here needs it; but the dispatcher owns the test clock, and dropping it
    /// early while tasks still hold a reference to it would split the clock in
    /// two. Holding one here ties its lifetime to the harness.
    #[allow(dead_code)]
    dispatcher: TestDispatcher,
    text_system: Arc<TextSystem>,
}

impl TestApp {
    /// Create a new test application with the default (noop) text backend.
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Create a new test application with a specific random seed.
    ///
    /// The seed drives everything on the platform that is random — display
    /// uuids and the like — so a test that needs a particular draw is
    /// reproducible rather than lucky.
    pub fn with_seed(seed: u64) -> Self {
        Self::build(seed, None, Arc::new(()))
    }

    /// Create a new test application with a custom text system for real font shaping.
    ///
    /// The default backend measures nothing: every glyph advances zero, so text
    /// lays out as if it were empty and an assertion about where a word lands
    /// is meaningless. Supplying a [`PlatformTextSystem`] here is what makes
    /// text layout testable at all, and supplying a *deterministic* one is what
    /// makes the assertion portable across machines with different fonts
    /// installed.
    pub fn with_text_system(text_system: Arc<dyn PlatformTextSystem>) -> Self {
        Self::build(0, Some(text_system), Arc::new(()))
    }

    /// Create a new test application with a custom text system and asset source.
    ///
    /// Needed when a view under test resolves assets — icons, embedded fonts —
    /// since the default source resolves nothing and the view would render as
    /// though the file were missing.
    pub fn with_text_system_and_assets(
        text_system: Arc<dyn PlatformTextSystem>,
        asset_source: Arc<dyn AssetSource>,
    ) -> Self {
        Self::build(0, Some(text_system), asset_source)
    }

    fn build(
        seed: u64,
        platform_text_system: Option<Arc<dyn PlatformTextSystem>>,
        asset_source: Arc<dyn AssetSource>,
    ) -> Self {
        let dispatcher = TestDispatcher::new(StdRng::seed_from_u64(seed));
        let arc_dispatcher = Arc::new(dispatcher.clone());
        let background_executor = BackgroundExecutor::new(arc_dispatcher.clone());
        let foreground_executor = ForegroundExecutor::new(arc_dispatcher);
        let platform = match platform_text_system.clone() {
            Some(text_system) => TestPlatform::with_text_system(
                background_executor.clone(),
                foreground_executor.clone(),
                text_system,
            ),
            None => TestPlatform::new(background_executor.clone(), foreground_executor.clone()),
        };
        let http_client = http_client::FakeHttpClient::with_404_response();
        // Built from the same backend the platform got, never from the
        // platform's own copy: a window reaches fonts through the text system,
        // so if the two disagreed a test would shape with one set of metrics
        // and measure with another.
        let text_system = Arc::new(TextSystem::new(
            platform_text_system.unwrap_or_else(|| platform.text_system()),
        ));

        let app = App::new_app(platform.clone(), asset_source, http_client);
        app.borrow_mut().mode = GpuiMode::test();

        Self {
            app,
            platform,
            background_executor,
            foreground_executor,
            dispatcher,
            text_system,
        }
    }

    /// Run a closure with mutable access to the App context.
    ///
    /// Automatically runs until parked after the closure completes, so effects
    /// the closure scheduled — entity notifications, window refreshes — have
    /// been delivered by the time this returns and the next statement sees a
    /// settled app.
    pub fn update<R>(&mut self, f: impl FnOnce(&mut App) -> R) -> R {
        let result = {
            let mut app = self.app.borrow_mut();
            app.update(f)
        };
        self.run_until_parked();
        result
    }

    /// Run a closure with read-only access to the App context.
    pub fn read<R>(&self, f: impl FnOnce(&App) -> R) -> R {
        let app = self.app.borrow();
        f(&app)
    }

    /// Create a new entity in the app.
    pub fn new_entity<T: 'static>(
        &mut self,
        build: impl FnOnce(&mut Context<T>) -> T,
    ) -> Entity<T> {
        self.update(|cx| cx.new(build))
    }

    /// Update an entity.
    pub fn update_entity<T: 'static, R>(
        &mut self,
        entity: &Entity<T>,
        f: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> R {
        self.update(|cx| entity.update(cx, f))
    }

    /// Read an entity.
    pub fn read_entity<T: 'static, R>(
        &self,
        entity: &Entity<T>,
        f: impl FnOnce(&T, &App) -> R,
    ) -> R {
        self.read(|cx| f(entity.read(cx), cx))
    }

    /// Open a test window with the given root view, using maximized bounds.
    ///
    /// Maximized rather than some fixed size so that the window fills the
    /// simulated display: a test asserting on layout should not silently be
    /// asserting on a viewport it happened to pick.
    pub fn open_window<V: Render + 'static>(
        &mut self,
        build_view: impl FnOnce(&mut Window, &mut Context<V>) -> V,
    ) -> TestAppWindow<V> {
        let bounds = self.read(|cx| Bounds::maximized(None, cx));
        let handle = self.update(|cx| {
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |window, cx| cx.new(|cx| build_view(window, cx)),
            )
            .expect("failed to open test window")
        });

        TestAppWindow {
            handle,
            app: self.app.clone(),
            background_executor: self.background_executor.clone(),
        }
    }

    /// Open a test window with specific options.
    ///
    /// The escape hatch from [`open_window`](Self::open_window)'s maximized
    /// bounds, for a test that needs a particular size or an undecorated
    /// window.
    pub fn open_window_with_options<V: Render + 'static>(
        &mut self,
        options: WindowOptions,
        build_view: impl FnOnce(&mut Window, &mut Context<V>) -> V,
    ) -> TestAppWindow<V> {
        let handle = self.update(|cx| {
            cx.open_window(options, |window, cx| cx.new(|cx| build_view(window, cx)))
                .expect("failed to open test window")
        });

        TestAppWindow {
            handle,
            app: self.app.clone(),
            background_executor: self.background_executor.clone(),
        }
    }

    /// Run pending tasks until there's nothing left to do.
    pub fn run_until_parked(&self) {
        self.background_executor.run_until_parked();
    }

    /// Advance the simulated clock by the given duration.
    ///
    /// Time moves only when a test asks it to, so anything on a timer — a
    /// debounce, an animation, a retry — advances by exactly as much as the
    /// test intends rather than by however long the machine took.
    pub fn advance_clock(&self, duration: Duration) {
        self.background_executor.advance_clock(duration);
    }

    /// Spawn a future on the foreground executor.
    pub fn spawn<Fut, R>(&self, f: impl FnOnce(AsyncApp) -> Fut) -> Task<R>
    where
        Fut: Future<Output = R> + 'static,
        R: 'static,
    {
        self.foreground_executor.spawn(f(self.to_async()))
    }

    /// Spawn a future on the background executor.
    pub fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R>
    where
        R: Send + 'static,
    {
        self.background_executor.spawn(future)
    }

    /// Get an async handle to the app.
    pub fn to_async(&self) -> AsyncApp {
        AsyncApp {
            app: Rc::downgrade(&self.app),
            background_executor: self.background_executor.clone(),
            foreground_executor: self.foreground_executor.clone(),
        }
    }

    /// Get the background executor.
    pub fn background_executor(&self) -> &BackgroundExecutor {
        &self.background_executor
    }

    /// Get the foreground executor.
    pub fn foreground_executor(&self) -> &ForegroundExecutor {
        &self.foreground_executor
    }

    /// Get the text system.
    pub fn text_system(&self) -> &Arc<TextSystem> {
        &self.text_system
    }

    /// Check if a global of the given type exists.
    pub fn has_global<G: Global>(&self) -> bool {
        self.read(|cx| cx.has_global::<G>())
    }

    /// Set a global value.
    pub fn set_global<G: Global>(&mut self, global: G) {
        self.update(|cx| cx.set_global(global));
    }

    /// Read a global value.
    pub fn read_global<G: Global, R>(&self, f: impl FnOnce(&G, &App) -> R) -> R {
        self.read(|cx| f(cx.global(), cx))
    }

    /// Update a global value.
    pub fn update_global<G: Global, R>(&mut self, f: impl FnOnce(&mut G, &mut App) -> R) -> R {
        self.update(|cx| cx.update_global(f))
    }

    // Platform simulation methods

    /// Write text to the simulated clipboard.
    pub fn write_to_clipboard(&self, item: ClipboardItem) {
        self.platform.write_to_clipboard(item);
    }

    /// Read from the simulated clipboard.
    pub fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.platform.read_from_clipboard()
    }

    /// Get URLs that have been opened via `cx.open_url()`.
    pub fn opened_url(&self) -> Option<String> {
        self.platform.opened_url.borrow().clone()
    }

    /// Check if a file path prompt is pending.
    pub fn did_prompt_for_new_path(&self) -> bool {
        self.platform.did_prompt_for_new_path()
    }

    /// Simulate answering a path selection dialog.
    pub fn simulate_new_path_selection(
        &self,
        select: impl FnOnce(&std::path::Path) -> Option<std::path::PathBuf>,
    ) {
        self.platform.simulate_new_path_selection(select);
    }

    /// Check if a prompt dialog is pending.
    pub fn has_pending_prompt(&self) -> bool {
        self.platform.has_pending_prompt()
    }

    /// Simulate answering a prompt dialog.
    pub fn simulate_prompt_answer(&self, button: &str) {
        self.platform.simulate_prompt_answer(button);
    }

    /// Get all open windows.
    pub fn windows(&self) -> Vec<AnyWindowHandle> {
        self.read(|cx| cx.windows())
    }
}

impl Default for TestApp {
    fn default() -> Self {
        Self::new()
    }
}

/// A test window with inspection and simulation capabilities.
///
/// Held by value rather than through the app, so a test can keep one open
/// across several assertions and drive it directly; cloning it produces
/// another handle onto the same window, not a second window.
pub struct TestAppWindow<V> {
    handle: WindowHandle<V>,
    app: Rc<AppCell>,
    background_executor: BackgroundExecutor,
}

impl<V: 'static + Render> TestAppWindow<V> {
    /// Get the window handle.
    pub fn handle(&self) -> WindowHandle<V> {
        self.handle
    }

    /// Get the root view entity.
    pub fn root(&self) -> Entity<V> {
        let mut app = self.app.borrow_mut();
        let any_handle: AnyWindowHandle = self.handle.into();
        app.update_window(any_handle, |root_view, _, _| {
            root_view
                .downcast::<V>()
                .ok()
                .expect("root view type mismatch")
        })
        .expect("window not found")
    }

    /// Update the root view, then run until parked.
    ///
    /// The closure receives the window as well as the view, because much of
    /// what a test wants to poke at — text styles, rem size, the element
    /// arena — lives on the window and is only reachable while one is in hand.
    pub fn update<R>(&mut self, f: impl FnOnce(&mut V, &mut Window, &mut Context<V>) -> R) -> R {
        let result = {
            let mut app = self.app.borrow_mut();
            let any_handle: AnyWindowHandle = self.handle.into();
            app.update_window(any_handle, |root_view, window, cx| {
                let view = root_view
                    .downcast::<V>()
                    .ok()
                    .expect("root view type mismatch");
                view.update(cx, |view, cx| f(view, window, cx))
            })
            .expect("window not found")
        };
        self.background_executor.run_until_parked();
        result
    }

    /// Read the root view.
    pub fn read<R>(&self, f: impl FnOnce(&V, &App) -> R) -> R {
        let app = self.app.borrow();
        let view = app
            .windows
            .get(self.handle.window_id())
            .and_then(|window| window.as_ref())
            .and_then(|window| window.root.clone())
            .and_then(|root| root.downcast::<V>().ok())
            .expect("window or root view not found");
        f(view.read(&app), &app)
    }

    /// Get the window title, as the root view last set it.
    pub fn title(&self) -> Option<String> {
        self.test_window().0.lock().title.clone()
    }

    /// Simulate a keystroke, e.g. `"ctrl-shift-p"`.
    pub fn simulate_keystroke(&mut self, keystroke: &str) {
        let keystroke = Keystroke::parse(keystroke).expect("invalid keystroke");
        {
            let mut app = self.app.borrow_mut();
            let any_handle: AnyWindowHandle = self.handle.into();
            app.update_window(any_handle, |_, window, cx| {
                window.dispatch_keystroke(keystroke, cx);
            })
            .expect("window not found");
        }
        self.background_executor.run_until_parked();
    }

    /// Simulate multiple keystrokes (space-separated).
    pub fn simulate_keystrokes(&mut self, keystrokes: &str) {
        for keystroke in keystrokes.split(' ') {
            self.simulate_keystroke(keystroke);
        }
    }

    /// Simulate typing text, one character per keystroke.
    pub fn simulate_input(&mut self, input: &str) {
        for char in input.chars() {
            self.simulate_keystroke(&char.to_string());
        }
    }

    /// Simulate a mouse move.
    pub fn simulate_mouse_move(&mut self, position: Point<Pixels>) {
        self.simulate_event(MouseMoveEvent {
            position,
            modifiers: Default::default(),
            pressed_button: None,
        });
    }

    /// Simulate a mouse down event.
    pub fn simulate_mouse_down(&mut self, position: Point<Pixels>, button: MouseButton) {
        self.simulate_event(MouseDownEvent {
            position,
            button,
            modifiers: Default::default(),
            click_count: 1,
            first_mouse: false,
        });
    }

    /// Simulate a mouse up event.
    pub fn simulate_mouse_up(&mut self, position: Point<Pixels>, button: MouseButton) {
        self.simulate_event(MouseUpEvent {
            position,
            button,
            modifiers: Default::default(),
            click_count: 1,
        });
    }

    /// Simulate a click at the given position.
    pub fn simulate_click(&mut self, position: Point<Pixels>, button: MouseButton) {
        self.simulate_mouse_down(position, button);
        self.simulate_mouse_up(position, button);
    }

    /// Simulate a scroll event.
    pub fn simulate_scroll(&mut self, position: Point<Pixels>, delta: Point<Pixels>) {
        self.simulate_event(ScrollWheelEvent {
            position,
            delta: ScrollDelta::Pixels(delta),
            modifiers: Default::default(),
            touch_phase: TouchPhase::Moved,
        });
    }

    /// Simulate an input event.
    pub fn simulate_event<E: InputEvent>(&mut self, event: E) {
        let platform_input = event.to_platform_input();
        {
            let mut app = self.app.borrow_mut();
            let any_handle: AnyWindowHandle = self.handle.into();
            app.update_window(any_handle, |_, window, cx| {
                window.dispatch_event(platform_input, cx);
            })
            .expect("window not found");
        }
        self.background_executor.run_until_parked();
    }

    /// Simulate resizing the window.
    ///
    /// The resize callback re-enters the app — it is how the window learns its
    /// new size — so the harness must not still be holding a borrow when it
    /// fires. The window is cloned out first, leaving the app free.
    pub fn simulate_resize(&mut self, size: Size<Pixels>) {
        let mut test_window = self.test_window();
        test_window.simulate_resize(size);
        self.background_executor.run_until_parked();
    }

    /// Simulate the window moving to a display with a different scale factor.
    ///
    /// Delivered as a resize, since that is how the platform reports it: the
    /// window keeps its logical size, and everything measured in device pixels
    /// has to be redone. As with [`simulate_resize`](Self::simulate_resize),
    /// the borrow is released before the callback runs.
    pub fn simulate_scale_factor_change(&mut self, scale_factor: f32) {
        let mut test_window = self.test_window();
        test_window.simulate_scale_factor_change(scale_factor);
        self.background_executor.run_until_parked();
    }

    /// Force a redraw of the window.
    pub fn draw(&mut self) {
        let mut app = self.app.borrow_mut();
        let any_handle: AnyWindowHandle = self.handle.into();
        app.update_window(any_handle, |_, window, cx| {
            window.draw(cx).clear();
        })
        .expect("window not found");
    }

    /// The platform window backing this handle, cloned so the app borrow can
    /// be dropped before anything that re-enters the app is called.
    fn test_window(&self) -> TestWindow {
        let mut app = self.app.borrow_mut();
        let any_handle: AnyWindowHandle = self.handle.into();
        app.windows
            .get_mut(any_handle.window_id())
            .and_then(|window| window.as_deref_mut())
            .and_then(|window| window.platform_window.as_test())
            .cloned()
            .expect("test window not found")
    }
}

impl<V> Clone for TestAppWindow<V> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
            app: self.app.clone(),
            background_executor: self.background_executor.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Empty, FocusHandle, Focusable, div, prelude::*};

    struct Counter {
        count: usize,
        focus_handle: FocusHandle,
    }

    impl Counter {
        fn new(_window: &mut Window, cx: &mut Context<Self>) -> Self {
            let focus_handle = cx.focus_handle();
            Self {
                count: 0,
                focus_handle,
            }
        }

        fn increment(&mut self, _cx: &mut Context<Self>) {
            self.count += 1;
        }
    }

    impl Focusable for Counter {
        fn focus_handle(&self, _cx: &App) -> FocusHandle {
            self.focus_handle.clone()
        }
    }

    impl Render for Counter {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().child(format!("Count: {}", self.count))
        }
    }

    #[test]
    fn updates_are_flushed_by_the_time_update_returns() {
        let mut app = TestApp::new();

        let mut window = app.open_window(Counter::new);

        window.update(|counter, _window, cx| {
            counter.increment(cx);
        });

        window.read(|counter, _| {
            assert_eq!(counter.count, 1);
        });
    }

    #[test]
    fn drawing_a_window_is_a_no_op_for_an_unchanged_view() {
        let mut app = TestApp::new();
        let mut window = app.open_window(|_, _| Empty);

        // A frame is produced, and the arena it allocated into is cleared, so
        // a second draw is safe rather than tripping the arena's checks.
        window.draw();
        window.draw();
    }

    #[test]
    fn simulated_scale_factor_change_reaches_the_window() {
        let mut app = TestApp::new();
        let mut window = app.open_window(|_, _| Empty);
        let viewport_size = window.update(|_, window, _| {
            assert_eq!(window.scale_factor(), 2.0);
            window.viewport_size()
        });

        window.simulate_scale_factor_change(1.0);

        window.update(|_, window, _| {
            assert_eq!(window.scale_factor(), 1.0);
            assert_eq!(window.viewport_size(), viewport_size);
        });
    }

    #[test]
    fn entities_can_be_created_and_updated_outside_a_window() {
        let mut app = TestApp::new();

        let entity = app.new_entity(|cx| Counter {
            count: 42,
            focus_handle: cx.focus_handle(),
        });

        app.read_entity(&entity, |counter, _| {
            assert_eq!(counter.count, 42);
        });

        app.update_entity(&entity, |counter, _cx| {
            counter.count += 1;
        });

        app.read_entity(&entity, |counter, _| {
            assert_eq!(counter.count, 43);
        });
    }

    #[test]
    fn globals_are_visible_to_the_harness() {
        let mut app = TestApp::new();

        struct MyGlobal(String);
        impl Global for MyGlobal {}

        assert!(!app.has_global::<MyGlobal>());

        app.set_global(MyGlobal("hello".into()));

        assert!(app.has_global::<MyGlobal>());

        app.read_global::<MyGlobal, _>(|global, _| {
            assert_eq!(global.0, "hello");
        });

        app.update_global::<MyGlobal, _>(|global, _| {
            global.0 = "world".into();
        });

        app.read_global::<MyGlobal, _>(|global, _| {
            assert_eq!(global.0, "world");
        });
    }

    #[test]
    fn opened_windows_are_listed_by_the_harness() {
        let mut app = TestApp::new();
        assert!(app.windows().is_empty());

        let window = app.open_window(|_, _| Empty);
        assert_eq!(app.windows(), vec![window.handle().into()]);
    }
}
