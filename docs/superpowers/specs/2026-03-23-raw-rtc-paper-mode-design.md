# Raw RTC Paper Mode Design

## Goal

Keep the current `raw_prefix` RTC behavior as a stable legacy baseline, and add a separate raw RTC mode that more closely matches the RTC paper's overlap-focused soft-masking behavior.

## Why

Recent sync experiments suggest the model's intra-chunk smoothness is already decent. That shifts the optimization target back to the original RTC problem: reducing discontinuity at chunk handoff without replacing the model's own chunk dynamics.

The current `raw_prefix` implementation already applies prefix-weighted guidance, but it still differs from the paper in two important ways:

- it commonly uses a manually truncated guidance horizon
- it defaults to a linear decay schedule instead of the paper-style soft mask

## Scope

This design covers:

- a new paper-style raw RTC mode
- CLI/config wiring for selecting it cleanly
- tests that preserve legacy behavior while validating the new semantics

This design does not cover:

- removing or changing the existing `raw_prefix` behavior
- further B1 work
- training-side changes

## Approach

### Legacy mode

Keep `raw_prefix` unchanged.

- It continues to use `rtc_execution_horizon` as the mask decay endpoint.
- It continues to respect the explicit `rtc_schedule` value exactly as today.

### Paper mode

Add a new `raw_prefix_paper` guidance mode.

- It still performs raw-space RTC guidance through the existing vector-Jacobian-product path.
- Its mask endpoint is derived from the available raw overlap, not from a short manually truncated prefix.
- Its default schedule should be paper-oriented, using exponential decay unless the user explicitly overrides the schedule.

This gives us a clean A/B:

- `raw_prefix`: current engineering baseline
- `raw_prefix_paper`: more faithful paper-style raw RTC

## Behavior Details

For `raw_prefix_paper`, the prefix weights should behave like:

- weight `1` on the frozen region caused by inferred delay
- soft decay over the remaining overlap region
- `0` beyond the overlap

The overlap endpoint should be computed from the real raw leftover length available to the guidance function after padding/clamping, so the mode naturally tracks how much prior chunk remains relevant.

## CLI and Config

Expose the new mode through the existing RTC CLI.

- `--rtc_guidance_mode raw_prefix_paper`

Also add a schedule option that avoids forcing users to remember different defaults per mode:

- `--rtc_schedule auto`

With `auto`:

- `raw_prefix` resolves to `linear`
- `raw_prefix_paper` resolves to `exp`
- `executed_prefix_b1` keeps its current behavior unaffected

Explicit values like `linear` or `exp` should still override `auto`.

## Testing Strategy

Add focused tests that prove:

- legacy `raw_prefix` outputs are unchanged
- `raw_prefix_paper` uses a longer overlap-aware mask than legacy when `rtc_execution_horizon` would otherwise truncate guidance
- CLI/config wiring accepts the new mode and schedule option

## Expected Outcome

After this change, you can compare two raw RTC variants cleanly:

- the existing short-prefix engineering variant
- a more paper-faithful overlap-guided variant

That should make future trajectory-smoothness sweeps much easier to interpret.
