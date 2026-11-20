# Changelog

All notable changes to `awvision` are recorded here.

## [Unreleased]

### Changed

- **awvision**: the default vision-model order is now `bonsai2-27b-5090` then the
  `bonsai2-27b` alias; `gemma4-12b` is dropped. Measured 2026-11-20 on three UI crops:
  the 27B labelled roles (text/button) correctly where the 4B got them wrong, for 1-2 s
  more per crop; gemma4-12b answers with an empty message when tools are offered and
  underperforms as a vision model. Re-measured that day on one SAVE-button crop:
  gemma4-12b returned HTTP 503 no_backend with an empty `choices[0].message.content`.
  `sight`'s recorded default model name follows the same choice.

### Added

- **awvision**: `sight.capture_screen_origin(out)` reports the captured monitor's real
  origin alongside the frame (via `mss` when installed; the Pillow primary monitor falls
  back to `(0,0)`), so a capture on a second monitor is localized in desktop pixels.

## [0.3.3] - 2026-11-19

### Added

- **awvision**: the pinned vision resolver (one `/v1/models` probe, locked, refuses the
  plain-http mesh fallback) so every crop of a screen answers from the same
  scheduler-routed model.
- **awvision**: `sight.capture_screen` and `sight.capture` for live frames, plus the
  frame signature/diff helpers `awscreen` uses to skip an unchanged screen.

### Fixed

- **awvision**: `resolve_pinned` keeps `AWVISION_URL`/config as written and never falls
  back to an unverified endpoint.
