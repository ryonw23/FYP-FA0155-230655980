# Controlled pose-corruption evaluation

This development-only utility accepts previously extracted **raw, unsmoothed**
landmark frames. It does not rerun MediaPipe and is not imported by the
production analysis path. Every scenario is applied independently to a deep
copy, then both the clean control and corrupted copy are passed to the existing
`build_kinematic_self_calibration` function.

Run the small knee matrix against a clean source frame:

```bash
python -m powerlifting_coach.evaluation.pose_corruption \
  extraction-log.json corruption-results.json \
  --side left --frame 120 --csv corruption-summary.csv
```

The input may be a raw frame array, an object with a `frames` array, or an
extraction log with `landmark_time_series`. JSON retains the complete scenario,
coordinate change records, clean/corrupted profiles, multi-signal comparisons,
and reacquisition evidence. The optional CSV is a compact dissertation-oriented
table. Library callers can define scenarios for shoulder, hip, knee, or ankle
and choose horizontal, vertical, or diagonal displacement.

Offsets equal `magnitude_body_scale` multiplied by the clean control profile's
median linked shoulder-to-ankle scale. Diagonal offsets preserve that total
distance by dividing it equally between the axes. Levels such as 0.05, 0.10,
0.25, and 0.50 are experimental perturbation severities only; they are not
biomechanical quantities or production anomaly thresholds.

Rank percentile is the empirical percentage of finite robust deviations in the
corrupted run's corresponding signal distribution that are less than or equal
to the injected observation. Top 1%, 5%, and 10% fields are descriptive views of
that rank, not decisions. Signals remain separate rather than being combined
into a score. A zero-MAD or unavailable diagnostic remains `null`, faithfully
preserving the existing calibration limitation.

Synthetic results can show whether the method responds predictably to known
trajectory changes. They do **not** establish MediaPipe accuracy, real-world
corruption frequency, biomechanical ground truth, or coaching accuracy. Known
real-video failures remain separate observational case studies.

## Deadlift body-scale normalisation

Compare the production whole-video linked-chain scale with the same scale from
only the detected setup window, without pose re-extraction:

```bash
python -m powerlifting_coach.evaluation.deadlift_scale \
  extraction-log.json [more-extraction-logs.json ...] \
  --json deadlift-scale-results.json --csv deadlift-scale-summary.csv
```

Supply the JSON extraction log downloaded from the normal UI for each deadlift.
It must contain the unmodified `landmark_time_series` array (including each
entry's `landmarks` object), plus `fps`; retain `frame_width` and `frame_height`
when present. The CSV is the concise comparison table and JSON retains scale
sample counts, framewise coordination values, fixed-threshold sensitivity
labels, and the isolated coordinate-transformation check.

This exploratory comparison cannot establish which normalisation is more
accurate or stable without independent reference labels or repeated comparable
recordings. Transformation invariance does not establish tracking or
biomechanical accuracy. Results from previously inspected clips are exploratory.
