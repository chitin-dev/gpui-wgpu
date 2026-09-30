//! Frame timing for observers that live outside the frame loop.
//!
//! Everything here serves one question: *how long did the frames this window
//! just drew take, and how much work was coalesced into each one?* A frame
//! counter, a frame-time chart, an FPS overlay — each is a reader of that
//! answer, and none of them should have to be wired into the render path to
//! get it.
//!
//! The shape is a process-wide ring buffer of recent [`FrameEvent`]s that
//! survives independently of any reader. Writers are the render loop, which
//! appends a [`FrameEvent::Draw`] per drawn frame and a [`FrameEvent::Present`]
//! per frame handed to the platform; readers are [`FrameTimingCollector`]s,
//! each holding a cursor into that buffer and taking only the entries recorded
//! since it last looked. A reader can therefore be created at any point in the
//! process' life and never has to be registered anywhere first.
//!
//! Two properties make it safe to leave on:
//!
//! - **Recording is free when nobody is watching.** [`record_frame_event`]
//!   starts with one relaxed atomic load and returns; the ring buffer is only
//!   touched while [`trace_enabled`] is true. An application that never turns
//!   tracing on pays a load per frame and nothing else.
//! - **Reading never perturbs the frame.** A collector copies out what has
//!   accumulated and does no work in the render path; the overlay that
//!   displays the numbers cannot become the reason the numbers are bad.
//!
//! The switch is process-wide rather than per-collector because the buffer is:
//! one atomic gates every writer, and turning it off clears what was recorded
//! so a later reader cannot be handed frames from a period nobody was
//! measuring.
//!
//! This is deliberately a different instrument from the `flamegraph` feature's
//! capture engine. That one is an opt-in session — explicitly started, paying
//! for serialized spans and a binary trace export — and is the right tool for
//! asking where time went *inside* a frame. This module answers how long the
//! frame was, which has to be answerable cheaply and at all times, so it keeps
//! no spans and holds no session.

use crate::{WindowId, time_ext::Instant};
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

/// Timing for a single drawn window frame.
#[derive(Debug, Copy, Clone)]
pub struct FrameTiming {
    /// The window that was drawn.
    pub window_id: WindowId,
    /// When the frame first became dirty — its first invalidation. `None` when
    /// the frame was drawn before anything marked it dirty, which is how the
    /// very first frame of a window arrives.
    pub dirty_at: Option<Instant>,
    /// Number of invalidations coalesced into this frame.
    ///
    /// A number well above one is the shape of a redraw storm: work asked for
    /// several frames' worth of updates but the platform could only hand over
    /// one frame to carry them.
    pub invalidations: u64,
    /// When [`Window::draw`](crate::Window::draw) started.
    pub draw_start: Instant,
    /// When [`Window::draw`](crate::Window::draw) finished.
    pub draw_end: Instant,
}

impl FrameTiming {
    /// Time spent inside `Window::draw`.
    ///
    /// This is the framework's own cost for the frame, measured from the
    /// inside. It excludes the platform submission that follows, and it
    /// excludes everything the application did between frames, so it is the
    /// number to read when asking what a frame costs rather than how long the
    /// application took to decide to draw one.
    pub fn draw_duration(&self) -> Duration {
        self.draw_end.duration_since(self.draw_start)
    }

    /// Time from the frame's first invalidation to the end of its draw, if the
    /// first invalidation was observed.
    ///
    /// The other half of the frame: [`draw_duration`](Self::draw_duration) is
    /// what drawing cost, and this is how long the work waited to be drawn.
    /// Latency a user feels is the sum of the two.
    pub fn dirty_to_draw_duration(&self) -> Option<Duration> {
        self.dirty_at
            .map(|dirty_at| self.draw_end.duration_since(dirty_at))
    }
}

/// Work spent submitting a window frame to the platform.
#[derive(Debug, Copy, Clone)]
pub struct PresentTiming {
    /// The window whose frame was submitted.
    pub window_id: WindowId,
    /// When the platform submission began.
    pub present_start: Instant,
    /// When the platform submission completed.
    ///
    /// This, not the draw, is what a frame rate counts: a drawn frame that was
    /// never presented did not reach the display.
    pub present_end: Instant,
}

impl PresentTiming {
    /// Time spent submitting the frame to the platform.
    pub fn present_duration(&self) -> Duration {
        self.present_end.duration_since(self.present_start)
    }
}

/// A frame event recorded by the render loop.
///
/// Each drawn frame produces a [`Draw`](Self::Draw); each frame actually handed
/// to the platform produces a [`Present`](Self::Present). A frame that was
/// drawn but not presented — the window was occluded, or another draw
/// superseded it — has the former and not the latter, which is exactly the
/// distinction a frame-rate reader needs.
#[derive(Debug, Copy, Clone)]
pub enum FrameEvent {
    /// A window frame was drawn.
    Draw(FrameTiming),
    /// A newly drawn window frame was presented.
    Present(PresentTiming),
}

/// Whether frame events are currently being retained.
pub fn trace_enabled() -> bool {
    TRACE_ENABLED.load(Ordering::Relaxed)
}

/// Enables or disables frame-event recording, returning whether the setting
/// changed.
///
/// `false` means the switch was already where the caller asked it to be, which
/// is how a caller that shares the switch with others discovers whether it
/// owns it: turning tracing on and being told `false` means somebody else had
/// already turned it on, and that caller is the one who will turn it off.
///
/// Turning tracing off clears the ring buffer. Recorded frames are only ever
/// read by a collector that was alive while they were drawn, so a later reader
/// cannot be handed a period nobody was measuring — the alternative would be
/// an FPS overlay whose first reading covers the time before it was opened.
pub fn set_trace_enabled(enabled: bool) -> bool {
    match TRACE_ENABLED.compare_exchange(!enabled, enabled, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            if !enabled {
                clear_frame_timings();
            }
            true
        }
        Err(_) => false,
    }
}

/// Bytes of frame events to retain before the oldest are evicted.
///
/// Deliberately generous: a reader polls once per rendered frame and takes
/// everything new, so the buffer only has to hold the frames drawn between two
/// polls. Sizing it at sixteen mebibytes means a reader that stalls for
/// seconds — the overlay was hidden, the process was descheduled — still finds
/// the interval it missed rather than a hole.
const MAX_FRAME_TIMINGS: usize = (16 * 1024 * 1024) / core::mem::size_of::<FrameEvent>();

struct FrameTimings {
    timings: VecDeque<FrameEvent>,
    /// Total events ever pushed, including those since evicted. Readers keep a
    /// cursor into this rather than an index into `timings`, so eviction from
    /// the front does not make an old cursor point at the wrong entries.
    total_pushed: u64,
}

/// The ring buffer, behind a spin lock.
///
/// A spin lock rather than a parking one because every critical section is a
/// handful of instructions — one push, or one copy-out — and the writer is the
/// render loop, which must never be descheduled on behalf of a reader that is
/// merely drawing a number on screen.
static FRAME_TIMINGS: spin::Mutex<FrameTimings> = spin::Mutex::new(FrameTimings {
    timings: VecDeque::new(),
    total_pushed: 0,
});

static TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

fn clear_frame_timings() {
    let mut frames = FRAME_TIMINGS.lock();
    frames.timings.clear();
    frames.timings.shrink_to_fit();
    frames.total_pushed = 0;
}

/// Records a frame event.
///
/// No-op unless frame tracing is enabled via [`set_trace_enabled`], so the
/// render loop can call this unconditionally.
pub fn record_frame_event(event: FrameEvent) {
    if !trace_enabled() {
        return;
    }
    std::hint::cold_path(); // optimize for when profiling is off

    let mut frames = FRAME_TIMINGS.lock();
    if frames.timings.len() >= MAX_FRAME_TIMINGS {
        frames.timings.pop_front();
    }
    frames.timings.push_back(event);
    frames.total_pushed += 1;
}

/// Drains frame events recorded after this collector was created, tracking a
/// cursor so each call to [`Self::collect_unseen`] returns only new entries.
///
/// One collector per reader. Two readers sharing one would each see only the
/// frames the other had not already taken.
pub struct FrameTimingCollector {
    cursor: u64,
}

impl Default for FrameTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimingCollector {
    /// Creates a collector that only sees frame events recorded from this point on.
    ///
    /// Frames already in the buffer are skipped rather than returned: they
    /// were drawn before this reader existed, and reporting them would date
    /// the reader's first measurement to a time it was not running.
    pub fn new() -> Self {
        Self {
            cursor: FRAME_TIMINGS.lock().total_pushed,
        }
    }

    /// Returns frame events recorded since the previous call (or since the
    /// collector was created). If the ring buffer wrapped around since the
    /// previous poll, the evicted entries are lost.
    pub fn collect_unseen(&mut self) -> Vec<FrameEvent> {
        let frames = FRAME_TIMINGS.lock();
        let buffer_len = frames.timings.len() as u64;
        let buffer_start = frames.total_pushed.saturating_sub(buffer_len);
        let skip = self.cursor.saturating_sub(buffer_start) as usize;
        let unseen = frames
            .timings
            .iter()
            .skip(skip.min(frames.timings.len()))
            .copied()
            .collect();
        self.cursor = frames.total_pushed;
        unseen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes the tests that touch the process-wide switch.
    ///
    /// `TRACE_ENABLED` and the ring buffer are global by design, so two tests
    /// exercising them at once would see each other's frames and each other's
    /// switch flips. The guard restores tracing to off and empties the buffer
    /// however the test ends, including by panic, so one failure does not
    /// leave the switch on for the rest.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    struct TraceTestGuard(MutexGuard<'static, ()>);

    impl TraceTestGuard {
        fn new() -> Self {
            let guard = TEST_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            set_trace_enabled(false);
            Self(guard)
        }
    }

    impl Drop for TraceTestGuard {
        fn drop(&mut self) {
            set_trace_enabled(false);
        }
    }

    fn draw_event(window_id: WindowId, invalidations: u64) -> FrameEvent {
        let draw_start = Instant::now();
        FrameEvent::Draw(FrameTiming {
            window_id,
            dirty_at: Some(draw_start),
            invalidations,
            draw_start,
            draw_end: draw_start + Duration::from_millis(4),
        })
    }

    /// The names an observer outside this crate writes, spelled the way it
    /// writes them. `gpui` is this crate's own name for itself, so naming the
    /// types by their public paths here is what proves the module and the
    /// crate-root re-export both resolve, and that they resolve to the same
    /// types rather than to lookalikes.
    #[test]
    fn the_frame_types_are_nameable_by_their_public_paths() {
        fn draw(_: gpui::FrameTiming) {}
        fn present(_: gpui::profiler::PresentTiming) {}
        let _ = draw as fn(gpui::profiler::FrameTiming);
        let _ = present as fn(gpui::PresentTiming);
        let _ = gpui::set_trace_enabled as fn(bool) -> bool;
        let _ = gpui::profiler::set_trace_enabled as fn(bool) -> bool;
        let _ = gpui::trace_enabled as fn() -> bool;
        let _ = gpui::profiler::trace_enabled as fn() -> bool;
        let _: fn() -> gpui::FrameTimingCollector = gpui::profiler::FrameTimingCollector::new;
    }

    #[test]
    fn events_are_not_recorded_while_tracing_is_off() {
        let _guard = TraceTestGuard::new();
        let mut collector = FrameTimingCollector::new();

        record_frame_event(draw_event(WindowId::from(1), 1));

        assert!(collector.collect_unseen().is_empty());
    }

    #[test]
    fn a_collector_sees_only_events_recorded_after_it_was_created() {
        let _guard = TraceTestGuard::new();
        set_trace_enabled(true);

        record_frame_event(draw_event(WindowId::from(1), 1));
        let mut collector = FrameTimingCollector::new();
        record_frame_event(draw_event(WindowId::from(2), 3));

        let unseen = collector.collect_unseen();
        let [FrameEvent::Draw(timing)] = unseen.as_slice() else {
            panic!("expected exactly one draw event, got {unseen:?}");
        };
        assert_eq!(timing.window_id, WindowId::from(2));
        assert_eq!(timing.invalidations, 3);
    }

    #[test]
    fn a_collector_drains_rather_than_repeats() {
        let _guard = TraceTestGuard::new();
        set_trace_enabled(true);
        let mut collector = FrameTimingCollector::new();

        record_frame_event(draw_event(WindowId::from(1), 1));
        assert_eq!(collector.collect_unseen().len(), 1);
        assert!(
            collector.collect_unseen().is_empty(),
            "the second poll must not re-report the frame the first took"
        );
    }

    #[test]
    fn disabling_tracing_clears_what_was_recorded() {
        let _guard = TraceTestGuard::new();
        set_trace_enabled(true);
        let mut collector = FrameTimingCollector::new();
        record_frame_event(draw_event(WindowId::from(1), 1));

        set_trace_enabled(false);
        set_trace_enabled(true);

        assert!(
            collector.collect_unseen().is_empty(),
            "frames from before the switch was cleared must not resurface"
        );
    }

    #[test]
    fn the_switch_reports_whether_it_changed() {
        let _guard = TraceTestGuard::new();

        // Turns on: the value was false, so this is the change.
        assert!(set_trace_enabled(true));
        // Already on, so a caller asking to turn it on is told it does not own
        // the switch.
        assert!(!set_trace_enabled(true));
        assert!(trace_enabled());

        // Turns off: the value was true, so this is the change.
        assert!(set_trace_enabled(false));
        assert!(!set_trace_enabled(false));
        assert!(!trace_enabled());
    }

    #[test]
    fn draw_duration_is_measured_from_draw_start_to_draw_end() {
        let draw_start = Instant::now();
        let timing = FrameTiming {
            window_id: WindowId::from(7),
            dirty_at: Some(draw_start - Duration::from_millis(6)),
            invalidations: 2,
            draw_start,
            draw_end: draw_start + Duration::from_millis(4),
        };

        assert_eq!(timing.draw_duration(), Duration::from_millis(4));
        assert_eq!(
            timing.dirty_to_draw_duration(),
            Some(Duration::from_millis(10))
        );
    }

    #[test]
    fn a_frame_with_no_observed_invalidation_has_no_dirty_to_draw_duration() {
        let draw_start = Instant::now();
        let timing = FrameTiming {
            window_id: WindowId::from(7),
            dirty_at: None,
            invalidations: 0,
            draw_start,
            draw_end: draw_start,
        };

        assert_eq!(timing.dirty_to_draw_duration(), None);
    }
}
