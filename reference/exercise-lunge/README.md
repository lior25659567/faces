# Walking-lunge sequence

The current automatic source is the denser [26-frame observed one-step bundle](one-step/README.md).
Its fitted movement and independent contact replay passed, so it can build one forward step that
plays once. This directory retains the older 12-frame selection and its genuine missing-handoff
blocker for historical runs; it has not been relabeled as a complete sequence.

These 12 photographs are decoded frames from one fixed-camera recording of a
person performing **dumbbell walking lunges**. The athlete takes alternating
forward steps toward the camera. They provide a standing start, both lower
positions, and actual transitions, rather than two lower positions that would
make a reversed, stationary cycle.

## Attribution and license

Walking lunges, [wger exercise 206, video 33](https://wger.de/api/v2/exerciseinfo/206/),
by **Goulart**, is licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
The JPEG files in this directory are cropped frames derived from that video and
are distributed under the same license. The changes are video decoding, JPEG
encoding, and one identical rectangular crop; no pose edits or generated pixels
were applied. See [sources.json](sources.json) for the original video URL,
provider metadata, attribution, original-file checksum, and license record.

## Frame selection and reproduction

[sequence.json](sequence.json) lists the zero-based source frame index and actual
presentation timestamp of every image, along with its SHA-256 checksum. All
frames use the same 600 × 900 crop at x=650, y=150 within the 1920 × 1080 source.
This preserves the observed pixel movement and scale change.

Download the URL in `sources.json` and verify its SHA-256 before extraction:

```text
3d83e21334e0119fc831609c7581af02ec30642f1ebedf514d951ca25fd4d154
```

For example, frame 120 becomes `lunge-00.jpg` using FFmpeg:

```sh
ffmpeg -hide_banner -loglevel error -i walking-lunge.MOV \
  -vf 'select=eq(n\,120),crop=600:900:650:150' \
  -frames:v 1 -q:v 2 lunge-00.jpg
```

Repeat with the recorded `sourceFrameIndex` for each output file. FFmpeg's JPEG
encoder version can affect encoded checksums; the committed images and hashes
identify the reviewed bytes.

## Use limits

The camera is frontal and has no metric calibration. Forward travel is
foreshortened and the athlete grows in the image. Any depth, floor contacts, or
root travel inferred from these frames remains an approximation to validate
against the rendered movement. The rear knee and swing foot sometimes overlap.

The frames must remain in chronological order. They are not a pair of poses to
play forward and backward, and they must not be presented as a calibrated side
view. `autoBuild` remains false until the stepping builder passes its checks.
The legacy registry remains available for old runs. New unattended reference retrieval selects
the denser one-step bundle. An unchanged automatic 12-frame import can be retained beside the
new frames: an explicit source selection interprets the old sequence as context for that run,
without changing its files, tags or history. Manually edited inputs stop that upgrade for review.

The initial standing frame is at 2.001667 seconds. The athlete continues stepping
after the second lower position, so this selection has **no final standing
frame** and is not a closed cycle. At 7.005 seconds the trunk is upright while
the right foot passes the planted left foot; the selection ends at the next
right heel approach. A renderer must handle the end of playback explicitly.

Each frame records the visibly observed supporting foot and per-foot contact
states. `uncertain` marks a hidden foot or an ambiguous touchdown; it must not
silently become a fixed toe/floor contact. All 12 frames were readable by the
Studio pose detector, but rear-leg landmarks at the lower positions can have low
visibility. A readable body does not establish accurate depth or hidden joints.

Studio's bundled-reference registry exposes these reviewed files with their
original video URL, presentation timestamps, contact annotations, and attribution.
Loading a frame verifies its committed checksum locally and does not download the
video again. Callers cannot substitute a path or change a frame's source metadata.
The start, first lower position, and second lower position are comparison anchors;
the remaining frames are motion samples. The second lower position is not a
standing finish or the end of the source sequence. The registry also keeps
`autoBuild: false`, so this addition alone cannot start an unsupported animation.

The sequence reader preserves all 12 samples separately from the three comparison
anchors. A conservative root solver recovers travel only while a visibly planted
foot connects adjacent samples. This recording fails that requirement between
samples 1 and 2: the left support becomes uncertain while the right foot changes
from airborne to planted. Hip-relative pose landmarks do not establish forward
root travel across that unobserved handoff. The builder reports this boundary
instead of fixing an occluded foot, extrapolating travel, or making an in-place
forward/reverse cycle. A denser, clearly observed contact transition or calibrated
motion recording is required before unattended rendering can be enabled.
