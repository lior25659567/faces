#!/usr/bin/env python
"""Prove that every angle in the pose file reached the figure, one entry at a time.

    python3 pipeline/pose_audit.py

This is the check that was missing when 40 of the file's 85 angles were being dropped in silence.
`pose_model.solve` now refuses a `(joint, measure)` pair it has no handler for, which is the fix;
this is the independent confirmation, and it deliberately does NOT ask the same question. Asking
"is every pair in the file declared in `APPLIES`?" would only re-run the solver's own test and
agree with it — the adjacent measurement, the kind that reported this project's mesh binding 100%
clean through five failed fixes. A pair can be declared and still never consulted for one side, or
consulted and thrown away.

So the question here is the literal one: for each phase, which angle ENTRIES did the solver
actually read? `_lookup` is the single door every angle comes through, so it is wrapped for the
duration of one solve and asked to name what it handed back. An entry is accounted for if

    applied      some lookup returned it, so it is on the figure, or
    superseded   it lost to a sibling marked `"preferred": true` — the squat's high-bar torso and
                 the pushdown's alternative start, which are choices the file records and the
                 figure cannot hold both of

and is a DROP otherwise. Dropped angles are the failure; the audit exits non-zero.

Being independent of the solver's internals is also what lets this file measure the old code: run
it against a copy of a previous `pose_model.py` and it reports that version's drops in the same
columns, which is how the before-and-after numbers in the task report were obtained rather than
reasoned about.
"""

import math

import pose_model

# How far behind vertical a thigh may point before it is a complaint, in degrees. Not zero, because
# a thigh built by three chained rotations lands on vertical to within floating-point noise and not
# exactly on it.
BEHIND = 0.5


def solve_watched(phase):
    """Solve one phase and return the skeleton plus the id of every entry `_lookup` handed back.

    Identity, not equality: two entries of the same phase can carry identical fields (the squat's
    two torso angles differ only in degrees, and a careless `in` test on dicts would call one of
    them consumed because the other was).
    """
    read = []
    real = pose_model._lookup

    def watched(angles, joint, side, measures):
        found = real(angles, joint, side, measures)
        if found is not None:
            read.append(id(found))
        return found

    pose_model._lookup = watched
    try:
        return pose_model.solve(phase), read
    finally:
        pose_model._lookup = real


def verdicts(phase, read):
    """One verdict per angle of the phase, in file order."""
    out = []
    for angle in phase["angles"]:
        if id(angle) in read:
            out.append((angle, "applied"))
            continue
        # A group is one joint and one side; the measures inside it can differ, as the pushdown's
        # start elbow shows (included 90 against flexion 105).
        winner = [other for other in phase["angles"]
                  if other is not angle
                  and other["joint"] == angle["joint"] and other["side"] == angle["side"]
                  and other.get("preferred") and id(other) in read]
        out.append((angle, "superseded" if winner else "DROPPED"))
    return out


def thighs_are_under_the_figure(phase, skeleton):
    """Complaints about a thigh pointing behind vertical with nothing in the data putting it there.

    The leg's equivalent of `sides_are_sane`, and it exists for the same kind of reason: the
    bent-over row solved with its legs straight out behind it and floating, because an unstated hip
    angle let the thigh inherit the torso's lean. The angle accounting could not see it — every
    angle in that phase was applied, and the figure was still impossible. A lifter who leans over
    the bar keeps the thighs under them or falls over.

    Measured in the body's own frame, so that a figure lying down is judged by where its legs point
    along itself rather than by where they point in the room. A phase that states a hip angle for
    that side is exempt: if the source says the thigh goes back, it goes back.
    """
    pitch = pose_model.ORIENTATIONS.get(phase.get("orientation", "standing"), 0.0)
    complaints = []
    for side in ("left", "right"):
        hip, knee = skeleton.at(f"hip.{side}"), skeleton.at(f"knee.{side}")
        along = pose_model._rotate((knee[0] - hip[0], knee[1] - hip[1], knee[2] - hip[2]),
                                   (1.0, 0.0, 0.0), -pitch)
        forward = math.degrees(math.atan2(along[1], -along[2]))
        if forward >= -BEHIND:
            continue
        if [a for a in phase["angles"] if a["joint"] == "hip" and a["side"] in (side, "both")]:
            continue
        complaints.append(f"the {side} thigh points {-forward:.1f} degrees behind vertical and no "
                          f"hip angle in this phase puts it there")
    return complaints


def inert_entries(phase):
    """Angles that change nothing: solve the phase again without each one and see if it matters.

    Being read is not the same as having an effect, and this file would be lying by omission if it
    only proved the first. The bench press is the case that forced this: its finish states a
    shoulder abduction of 75 degrees, `_lookup` hands that entry over, the accounting above counts
    it applied — and the figure does not move a thousandth of a millimetre, because the abduction
    axis is the world's +Y and a supine lifter's arm lies along it.

    Not a failure, because plenty of entries are inert on purpose: an elbow "included 180" and an
    ankle "included 90" both state the rest pose, and the two superseded alternatives are inert by
    definition. It is a list to read. An entry in it whose number is nowhere near a rest value is
    a measure that is not reaching the figure.
    """
    inert = []
    for index, angle in enumerate(phase["angles"]):
        without = dict(phase, angles=[a for i, a in enumerate(phase["angles"]) if i != index])
        try:
            pruned = pose_model.solve(without)
        except ValueError:
            continue
        whole = pose_model.solve(phase)
        if all(max(abs(x - y) for x, y in zip(whole.at(name), pruned.at(name))) < 1e-9
               for name in whole.joints):
            inert.append(angle)
    return inert


def nothing_is_below_the_floor(skeleton):
    """Complaints about joints underground.

    The third check of the same family as `sides_are_sane` and the thigh one above, and the same
    kind of claim: not that a particular exercise looks right, but that no body does this. It is
    enforced in `ground()` and asserted here, because an invariant held in one function and
    checked nowhere is an invariant until somebody edits that function.

    Against the solver as it stands this is silent. Against the one from before the clamp it names
    the plank's prone start, where six joints sat up to 0.254 below the floor.
    """
    under = [(name, joint.position[2]) for name, joint in skeleton.joints.items()
             if joint.position[2] < -1e-6]
    if not under:
        return []
    worst = min(under, key=lambda pair: pair[1])
    return [f"{len(under)} joint(s) are below the floor, the lowest being {worst[0]} at "
            f"z={worst[1]:+.3f}"]


# How far a figure's root may drift along the floor between its own two phases before it counts as
# walking. One centimetre: tight enough to have caught the 23 mm the bench press used to slide,
# loose enough to ignore a contact set rounding differently by a fraction of a millimetre.
DRIFT = 0.01


def horizontal_drift(exercise):
    """Complaints about a figure that walks between its own two keyframes.

    `ground` answers two separate questions — what holds a figure UP, and where it sits in the
    GROUND PLANE — and it used to answer the second with the first. The plank slid 0.220 along the
    mat between its phases and the leg press slid 0.198 out of its seat. Neither was visible in a
    still, and both are obvious the moment the two frames are played in sequence, which is where
    the staging agent found them.

    Measured at the pelvis, because that is the figure's root: limbs are supposed to move between
    keyframes and the body is not. An exercise that genuinely travels says so in its own data, in a
    `travels` field quoting the sentence that says so, so this check carries no list of exercise
    names to go stale when the list changes.
    """
    phases = {phase["phase"]: phase for phase in exercise["phases"]}
    if "start" not in phases or "finish" not in phases:
        return []
    start = pose_model.solve(phases["start"]).at("pelvis")
    finish = pose_model.solve(phases["finish"]).at("pelvis")
    drift = math.dist(start[:2], finish[:2])
    if drift <= DRIFT or exercise.get("travels_horizontally"):
        return []
    return [f"the pelvis travels {drift * 1000:.0f} mm along the floor between the two phases, and "
            f"nothing in this exercise's data says it travels"]


def vertical_drift(exercise):
    """Complaints about a figure that rises or sinks when its exercise says it should not.

    The twin of the check above, and kept SEPARATE from it on purpose. The two axes are different
    claims about an exercise: a plank rises and must not slide, a squat does both, a leg press does
    neither. One field covering both would have meant declaring the plank a traveller for its
    lift-off, which would then have blinded the horizontal check to the 0.220 scoot that started
    all of this — in exactly the exercise it bit.

    Measured at the pelvis for the same reason: limbs move between keyframes, bodies do not, unless
    the movement is the body moving and the data says so.
    """
    phases = {phase["phase"]: phase for phase in exercise["phases"]}
    if "start" not in phases or "finish" not in phases:
        return []
    start = pose_model.solve(phases["start"]).at("pelvis")[2]
    finish = pose_model.solve(phases["finish"]).at("pelvis")[2]
    if abs(finish - start) <= DRIFT or exercise.get("travels_vertically"):
        return []
    return [f"the pelvis {'rises' if finish > start else 'sinks'} {abs(finish - start) * 1000:.0f} "
            f"mm between the two phases, and nothing in this exercise's data says it moves "
            f"vertically"]


def audit(data):
    """Walk every exercise and phase. Returns the rows, the drops and the outright failures."""
    rows, dropped, failed = [], [], []
    for exercise in data["exercises"]:
        for phase in exercise["phases"]:
            where = (exercise["name"], phase["phase"])
            try:
                skeleton, read = solve_watched(phase)
            except ValueError as problem:
                failed.append((where, str(problem)))
                rows.append((where, len(phase["angles"]), None, None, None, "REFUSED"))
                continue
            counts = {"applied": 0, "superseded": 0, "DROPPED": 0}
            for angle, verdict in verdicts(phase, read):
                counts[verdict] += 1
                if verdict == "DROPPED":
                    dropped.append((where, angle))
            crossed = pose_model.sides_are_sane(skeleton)
            behind = thighs_are_under_the_figure(phase, skeleton)
            sunk = nothing_is_below_the_floor(skeleton)
            for complaint in crossed + behind + sunk:
                failed.append((where, complaint))
            note = "+".join(tag for tag, found in (("SIDES", crossed), ("THIGHS", behind),
                                                   ("FLOOR", sunk)) if found)
            rows.append((where, len(phase["angles"]), counts["applied"], counts["superseded"],
                         counts["DROPPED"], note or "ok"))
    return rows, dropped, failed


def main():
    data = pose_model.load()
    rows, dropped, failed = audit(data)
    for exercise in data["exercises"]:
        for complaint in horizontal_drift(exercise) + vertical_drift(exercise):
            failed.append(((exercise["name"], "start+finish"), complaint))
    inert = [((exercise["name"], phase["phase"]), angle)
             for exercise in data["exercises"] for phase in exercise["phases"]
             for angle in inert_entries(phase)]

    print(f"{'exercise':46s} {'phase':7s} {'angles':>6s} {'applied':>8s} {'superseded':>11s} "
          f"{'dropped':>8s}  checks")
    print("-" * 100)
    totals = [0, 0, 0, 0]
    for (name, phase), angles, applied, superseded, drops, note in rows:
        if applied is None:
            print(f"{name[:46]:46s} {phase:7s} {angles:6d} {'—':>8s} {'—':>11s} {'—':>8s}  {note}")
            totals[0] += angles
            continue
        print(f"{name[:46]:46s} {phase:7s} {angles:6d} {applied:8d} {superseded:11d} "
              f"{drops:8d}  {note}")
        for index, value in enumerate((angles, applied, superseded, drops)):
            totals[index] += value
    print("-" * 100)
    print(f"{'TOTAL':46s} {'':7s} {totals[0]:6d} {totals[1]:8d} {totals[2]:11d} {totals[3]:8d}")

    # What the file asks for against what the solver declares it can apply. Not the assertion —
    # that is the table above — but the one thing a reviewer wants next: which measures the data
    # uses, and whether any of them is unknown to the solver.
    pairs = {}
    for exercise in data["exercises"]:
        for phase in exercise["phases"]:
            for angle in phase["angles"]:
                pairs[(angle["joint"], angle["measure"])] = \
                    pairs.get((angle["joint"], angle["measure"]), 0) + 1
    # `getattr` because this file is also pointed at older copies of the solver to measure them,
    # and the whole point of the before-and-after is that those copies declare nothing.
    declared = getattr(pose_model, "APPLIES", None)
    print(f"\n{len(pairs)} (joint, measure) pairs used by the data, "
          f"{len(declared) if declared is not None else 'no'} declared in pose_model.APPLIES:")
    for (joint, measure), count in sorted(pairs.items()):
        known = "" if declared is None else (
            "declared" if (joint, measure) in declared else "NOT DECLARED")
        print(f"  {joint:9s} {measure:16s} {count:3d} entr{'y' if count == 1 else 'ies'}  {known}")
    unused = sorted(pair for pair in (declared or ()) if pair not in pairs)
    if unused:
        print(f"  declared but unused by the data: "
              f"{', '.join(f'{j}/{m}' for j, m in unused)}")

    print(f"\n{len(inert)} angle(s) are INERT \u2014 removing them changes no joint at all. Most of "
          f"these state the rest pose, which is correct and expected; read the list for one that "
          f"does not:")
    for (name, phase), angle in inert:
        print(f"  {name[:40]:42s} {phase:7s} {angle['joint']} ({angle['side']}) "
              f"{angle['measure']} {angle['degrees']}")

    print("\nhow far the body itself moves between the two phases, measured at the pelvis:")
    print(f"  {'exercise':46s} {'along the floor':>16s} {'vertically':>12s}")
    for exercise in data["exercises"]:
        phases = {phase["phase"]: phase for phase in exercise["phases"]}
        if "start" not in phases or "finish" not in phases:
            continue
        a = pose_model.solve(phases["start"]).at("pelvis")
        b = pose_model.solve(phases["finish"]).at("pelvis")
        flat = math.dist(a[:2], b[:2]) * 1000
        up = (b[2] - a[2]) * 1000
        tags = []
        if flat > DRIFT * 1000:
            tags.append("walks" if not exercise.get("travels_horizontally") else "travels")
        if abs(up) > DRIFT * 1000:
            undeclared = not exercise.get("travels_vertically")
            tags.append(("rises" if up > 0 else "sinks") if undeclared else "rises/drops")
        bad = [tag for tag in tags if tag in ("walks", "rises", "sinks")]
        mark = ("  <-- " + ", ".join(tags)) if bad else (("  " + ", ".join(tags)) if tags else "")
        print(f"  {exercise['name'][:44]:46s} {flat:13.0f} mm {up:+9.0f} mm{mark}")

    if dropped:
        print(f"\n{len(dropped)} angle(s) DROPPED — read from the file and never applied:")
        for (name, phase), angle in dropped:
            print(f"  {name} [{phase}]: {angle['joint']} ({angle['side']}) "
                  f"{angle['measure']} {angle['degrees']}  {angle.get('source', '')[:70]!r}")
    if failed:
        print(f"\n{len(failed)} phase(s) failed to solve or posed a figure inside out:")
        for (name, phase), problem in failed:
            print(f"  {name} [{phase}]: {problem}")

    if dropped or failed:
        raise SystemExit(1)
    print(f"\nclean: {totals[0]} angles, {totals[1]} applied, {totals[2]} superseded by a "
          f"preferred sibling, 0 dropped.")


if __name__ == "__main__":
    main()
