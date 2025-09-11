Alright—straight up: this run is not client-ready. The pipeline executes end-to-end (JSON/CSV/XLSX are produced), but the analytics are internally inconsistent and look algorithmic/bug-driven rather than signal-driven. Here’s why, with receipts, and what to fix next.

What’s good (salvageable)

End-to-end export works: you’ve got a structured JSON plus Metrics/Phases/Events tables (CSV/XLSX).

Time alignment exists (timestamps, frames are emitted) and instrument detections carry non-zero confidences.

The run captures lots of events (so your IO/looping is wired).

What breaks trust (evidence)

Global metadata is inconsistent

video_duration is ~1073.33s at 30 FPS, but total_frames is 0 (it should be ~32,200).

Phases are degenerate

Every phase is the same class (LABRAL_MOBILIZATION) in 2-second blocks repeating every 10 seconds (0–2s, 10–12s, 20–22s, …). That’s a periodic pattern, not learned behavior. Examples: 0–2s, 10–12s, 20–22s.

Late video shows the same 2-sec blocks—again at 10-sec cadence (e.g., 910–912s, 920–922s, 930–932s).

Reported total labral-mobilization time is 216s (108 blocks × 2s), i.e., only ~20% coverage of a ~17.9-min case; everything else is missing.

Idle/active time is not credible

Phases show idle_time: 0.0 across the board, and roll-up shows total_idle_time: 0.0. Real surgeries aren’t 0% idle.

Bleeding events look synthetic

Summary says 108 bleeding events; individual events are 1-second long at regular timestamps (10-sec cadence). Example event at 1070–1071s, “Moderate,” no intervention.

Instrument timeline is periodic and incomplete

Only InstrumentType.GRASPER “entry” events appear, and they fire at regular 15-second intervals (e.g., 555s, 570s, 585s, 600s; later 765s, 780s…). There’s no evidence of “exit” or other tools.

Metrics block is broken

The metrics object reports zeros where real numbers are known and even a negative transition count (-1). That’s a post-processing bug.

Key clinical channels are empty

Anchors, anatomical structures, and custom events are empty; implants = 0; time-to-first-suture = 0. That contradicts a realistic labral repair flow.

Bottom line: a surgeon or exec will not trust these analytics. The outputs look like periodic windowing artifacts and default fallbacks—not meaningful detection.

What to fix next (enterprise-grade, in order)

Timebase & frame accounting

Compute and persist total_frames = round(video_duration * fps); ensure all events have valid end_frame/end_time and non-negative durations; add unit tests that fail on None, negatives, or phase_transition_count < 0. (This alone will catch half the issues.)

De-dup & smoothing

Stop fixed-cadence firing. Use non-overlapping windows for detectors; apply sequence models/filters (e.g., HMM/CRF or majority-vote smoothing) so phases don’t flip every 10s and instruments don’t “enter” every 15s with no exits.

Phase taxonomy & coverage

Predict the full labral-repair flow (diagnostic arthroscopy, glenoid prep, labral mobilization, anchor placement, suture passage, suture tensioning, final inspection). Enforce a coverage sanity check: Σ phase_durations / total_procedure_time should be ≥70–80% or mark “Unknown”.

Event validation rules

Bleeding: require min duration, hysteresis on start/stop, and temporal NMS to merge adjacent 1-second pops. Record confidence intervals and “intervention required” only if an intervention tool/event is actually detected within a short window.

Instrument lifecycle

Track enter and exit. Compute instrument_changes as tool-type switches over time (not 0). Enforce temporal consistency (a tool can’t “enter” every 15s without an “exit”).

Metrics aggregator correctness

Compute total_active_time, total_idle_time, phase_transition_count ≥ 0, efficiency_score from real aggregates (not defaults). Add schema validation (pydantic) + post-compute invariants (no negative counts; sums coherent with duration).

QA against ground truth

Run per-class precision/recall/F1 on phases, and event-level P/R at ±2s tolerance for bleeding, instrument events, and anchors. Produce a confusion matrix and a per-class report before you show any case to a client.

Client-facing deliverable shape

One-page summary (KPIs), plus an interactive timeline (phases, instruments, bleeding) and thumbnails around events. Export the same to PDF. KPIs a clinician cares about: total/phase durations, time-to-first-suture, instrument changes, bleeding count & longest episode, idle %.

What a “valuable” single-case report should show

Timeline coverage: ≥80% of the video classified into expected phases (not a single phase repeated).

Bleeding: few, consolidated episodes with sensible durations and timestamps, not periodic 1-sec pops.

Instruments: realistic lifecycles and changes (e.g., grasper ↔ shaver ↔ suture passer), not a single tool “entering” on a schedule.

Metrics: no zeros/negatives; numbers align (e.g., total_active + total_idle ≈ total_duration).

Anchors/sutures: counts and timings present when those steps appear.