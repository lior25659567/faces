# Bent-over row references

Authentic start and contracted-end photographs of the same barbell-row demonstration.
Downloaded from [Free Exercise DB](https://github.com/yuhonas/free-exercise-db) at commit
`f00c92c7dcf1216a928a52c3706c7ce8e2f71ed5`; exact URLs and SHA-256 hashes are in `sources.json`.
The upstream repository describes its JSON and imagery as a public-domain dataset and supplies
the Unlicense (copied in `UPSTREAM-LICENSE.md`). Its README credits `wrkout/exercises.json`;
the dataset does not name the original photographer. These are development reference inputs,
not generated examples or proof of anatomical ground truth.

- `start.jpg`: bar hanging below the shoulders, elbows extended, torso hinged and knees slightly bent.
- `end.jpg`: bar drawn toward the lower torso, elbows behind the body.
- Both photographs are **three-quarter/front views**, not side views. Keep that view tag in Studio.
  They show the bar and both hands well, but perspective makes flat side-angle measurements unsuitable.
- There is no middle photograph. Intermediate motion must be constrained/interpolated and validated;
  it is not observed ground truth.
- The source instructions emphasize a stationary torso, a slight knee bend, and controlled lowering.
  The source's instruction to keep the head up is descriptive of this example, not a required solver target.

Do not substitute the downloaded reverse-grip or rear-delt-row variants for this exercise: they use
different grips, pull targets, and camera angles.
