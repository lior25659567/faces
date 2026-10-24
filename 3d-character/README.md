# 3D character

The character the exercise animations are rendered with, and the drawings it is made from.

## What to do next

Send **`mascot-tpose-SEND-THIS-TO-TRIPO.png`** to Tripo (or any image-to-3D tool) and drop the
`.glb` it returns into this folder as `mascot-tpose.glb`. Then:

    cd ~/Developer/faces
    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python pipeline/blender/animate.py -- \
        --exercise "Dumbbell Lateral Raise" \
        --model 3d-character/mascot-tpose.glb \
        --out /tmp/raise

Everything downstream already works and re-checks itself on each run: the model is measured,
rigged, bound and verified before a frame is rendered, and the binding report prints how much of
each arm belongs to the arm bones.

## The files

| file | what it is |
| --- | --- |
| `mascot-tpose-SEND-THIS-TO-TRIPO.png` | The drawing to generate the next model from. Arms held straight out, so the armpits are open background. |
| `mascot-arms-down.glb` | The first Tripo model. Rigs and animates, but see the problem below. |
| `reference-drawing-front.png` | The house mascot, front. What every other drawn avatar is matched against. |
| `reference-drawing-back.png` | The same, from behind. |

## Why the T-pose matters

`mascot-arms-down.glb` was generated from a drawing with the arms hanging, so the arm is **fused
to the torso at the armpit**. Measured on the model itself: the silhouette only separates into
arm, torso, arm at 58% of the figure's height, while the shoulder joint sits at 63%. That leaves
roughly five centimetres of shared geometry which has to stretch across the whole opening when
the arm lifts, and it renders as a fan of long thin triangles under each arm.

No amount of weighting repairs that — the geometry to stretch is simply not there. It has to be
fixed in the model, which is why characters meant for animation are built in a T-pose or A-pose.

The replacement drawing was checked rather than assumed: scanning its silhouette across the arm
line finds **three separate pieces** — left arm, torso, right arm, with real background between
them. The arms-down version measures one.

## A caveat for the new model

A T-pose puts the shoulder at the top of its range in the REST pose, so a rigged arm will raise
*from* horizontal rather than *to* it. The pose solver handles this — it aims bones at absolute
directions taken from the form notes rather than applying offsets to wherever the rest pose sat —
but the first still you render from the new model will look different before the animation runs.

The thighs still touch at the crotch in this drawing. Fine for anything driven by the arms; a
version with the feet further apart is worth generating before attempting squats or lunges.
