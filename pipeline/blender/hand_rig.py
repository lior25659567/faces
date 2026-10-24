"""The cartoon's own hand: its fingers measured, a bone chain fitted inside each, weights designed per
finger, and a closed grip on a bar solved against the skin itself.

Why (task 13). Task 12 treated the cartoon's hand as a fused mitten: one finger chain (MPFB's
middle finger, rebuilt on the wrist-to-tip line) carried all four fingers and the thumb was rigid on
the hand. Measured on the welded mesh, the boy's hand is not a mitten. Past about 70% of its length
it is five separate pieces: four fingers 26-39 mm long and 12-14 mm thick, and a thumb. Each
fingertip has its own nail shell. MPFB's finger bones sit 6-44 mm off those fingers and point 9-63
degrees away from them. So each finger gets its own three joints, placed on its own centreline,
and its own weights. The grip is then solved the way a hand closes on a bar: each joint folds until
that segment meets the bar, judged on the deformed skin (linear blend skinning, exactly as Blender
and UsdSkel play it).

The core is plain numpy (no bpy) so it can be tested on exported geometry; the Blender side (bones,
pose) is in the functions at the end, which import bpy when called.

Frames and names
  The hand frame at rest: h = wrist -> middle knuckle (MCP), k = the knuckle line (index MCP ->
  little-finger MCP) made perpendicular to h, n = the palm's normal (toward the palm side, which in
  the bind's T-pose faces the floor). A finger folds about f = h x n, toward n.
  Finger ids: 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky; Mixamo names in FINGER_NAMES.
"""
import heapq
import math

import numpy as np

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Where the joints go, along each finger's centreline. The mesh shows where a finger separates from
# its neighbours (the web); the knuckle (MCP) lies inside the palm before it. Human anatomy puts
# the MCP about a fifth of the finger's length before the web (MCP_BACK x the free length), and
# divides MCP-to-tip as 0.45 / 0.27 / 0.28 between the three phalanges (Buryanov & Kotiuk 2010:
# index P1 39.8, P2 22.4, P3 15.8 mm plus the pad).
MCP_BACK = 0.25
PIP_AT, DIP_AT = 0.45, 0.72
THUMB_IP_AT = 0.5             # thumb: IP halfway from its MCP to its tip
THUMB_CMC_AT = 0.35           # thumb: CMC this far from the wrist to the thumb's MCP
# Weight blends at each joint, as half-widths in units of that finger's thickness
MCP_BLEND = 0.35
JOINT_BLEND = 0.30
# The palm's share near the knuckle row is split between fingers by distance to each finger's
# line, with this softness (x the finger's thickness)
LATERAL_SOFT = 0.25
# The thenar: palm points this close to the thumb's metacarpal (x thumb thickness) follow it
THENAR_IN, THENAR_OUT = 0.55, 1.1
SLAB_BLEND = 0.45             # x thumb width: the thumb's root fades into the palm over this


# --- topology ----------------------------------------------------------------------------------

def components(n, edges, mask):
    """Connected components of the vertices in `mask` (edges among them only). Returns a label per
    vertex (-1 outside the mask), labels are the smallest vertex index of each component."""
    lab = np.where(mask, np.arange(n), -1)
    e = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    if len(e) == 0:
        return lab
    while True:
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        before = lab.copy()
        np.minimum.at(lab, e[:, 0], m)
        np.minimum.at(lab, e[:, 1], m)
        lab[mask] = lab[lab[mask]]                     # pointer jumping
        if np.array_equal(lab, before):
            return lab


def welded_edges(edges, group):
    """Edges plus the coincident-vertex links (`group[v]` = its coincident group's representative),
    so a mesh split at UV seams reads as the surface it is."""
    v = np.arange(len(group))
    links = np.stack([v, group], axis=1)[group != v]
    return np.concatenate([edges, links]) if len(links) else edges


def pieces(n, edges, mask, min_size):
    lab = components(n, edges, mask)
    u, c = np.unique(lab[mask], return_counts=True)
    keep = [x for x, cc in sorted(zip(u, c), key=lambda t: -t[1]) if cc >= min_size]
    return [np.flatnonzero(lab == x) for x in keep]


# --- measuring the hand ------------------------------------------------------------------------

def _centreline(points, toward, base, samples=12):
    """A finger's centreline as a smooth curve: along its main direction d, the two offsets across
    it fitted as quadratics in the distance along d (least squares over the surface points, whose
    centre at each distance is the centreline; the cut end and the tip cap are left out of the fit).
    Runs from `base` (the centre of the ring where the finger leaves the palm) to the tip apex.
    Returns (polyline, tip, d)."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    d = vt[0] if vt[0] @ (toward - c) > 0 else -vt[0]
    e1 = vt[1] - d * (vt[1] @ d)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    t = (points - c) @ d
    p1, p2 = (points - c) @ e1, (points - c) @ e2
    t0 = float((base - c) @ d)
    hi = float(t.max())
    span = hi - t0
    # each cross-section's centre is the middle of its extent, not the mean of its vertices: a nail
    # welded into the finger (the boy's ring finger) packs vertices on its back and would drag a
    # mean upward
    step = span / 14.0
    ss, c1, c2 = [], [], []
    for s in np.arange(t0 + 0.1 * span, hi - 0.15 * span, step):
        sl = np.abs(t - s) < 0.6 * step
        if sl.sum() >= 6:
            ss.append(s)
            c1.append(0.5 * (p1[sl].min() + p1[sl].max()))
            c2.append(0.5 * (p2[sl].min() + p2[sl].max()))
    if len(ss) < 4:
        raise ValueError("a finger too short to measure")
    # A straight line from the web's centre to the tip cap's centre. A curve fitted through the
    # slice centres bent the wrong way on the ring finger (its welded nail), and the knuckle,
    # extrapolated back along that curve, landed on the back of the hand; the straight chord
    # misses the finger's slight rest curl by ~2 mm, which the joints' blends absorb.
    cap = t > hi - 0.12 * span
    tip = c + d * (hi - 0.04 * span) + e1 * 0.5 * (p1[cap].min() + p1[cap].max()) \
        + e2 * 0.5 * (p2[cap].min() + p2[cap].max())
    line = np.array([base + (tip - base) * f for f in np.linspace(0.0, 1.0, samples)])
    return line, tip, d


def _arc_point(poly, s):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
    f = (s - cum[i]) / max(seg[i], 1e-12)
    return poly[i] + (poly[i + 1] - poly[i]) * f, cum[-1]


def measure(co, edges, group, wrist, tip, palm_down=(0.0, 0.0, -1.0), forward=(0.0, 1.0, 0.0)):
    """One hand of a T-posed skin (palms down, facing +y): which vertices are which finger, and
    where each finger's joints are. `co` world positions at the bind, `edges` the mesh's edges,
    `group` each vertex's coincident group representative, `wrist`/`tip` the measured wrist joint
    and fingertip. Returns (model, labels): labels per vertex, -1 = not a finger piece (palm, wrist,
    or anything else), else the finger id (thumb 0 .. pinky 4), nails included.
    Raises ValueError if the hand does not read as four fingers and a thumb."""
    co = np.asarray(co, float)
    n = len(co)
    wrist, tip = np.asarray(wrist, float), np.asarray(tip, float)
    L = float(np.linalg.norm(tip - wrist))
    ax = (tip - wrist) / L
    rel = co - wrist
    along = rel @ ax
    across = np.linalg.norm(rel - np.outer(along, ax), axis=1)
    E = welded_edges(edges, group)
    region = (along > -0.1 * L) & (along < 1.2 * L) & (across < 0.6 * L)
    parts = pieces(n, E, region, 1)
    body = parts[0]
    bodymask = np.zeros(n, bool)
    bodymask[body] = True
    shells = [p for p in parts[1:] if len(p) >= 10 and along[p].mean() > 0.5 * L]

    # the cut where the hand first reads as five pieces (scanning from the palm out)
    cut = None
    for f in np.arange(0.40, 0.95, 0.01):
        ps = pieces(n, E, bodymask & (along > f * L), 15)
        if len(ps) == 5:
            cut, seeds = f, ps
            break
    if cut is None:
        raise ValueError("the hand does not separate into four fingers and a thumb")
    fwd = np.asarray(forward, float)
    ys = [float((co[p] @ fwd).mean()) for p in seeds]
    order = np.argsort(ys)[::-1]
    if ys[order[0]] - ys[order[1]] < 0.05 * L:
        raise ValueError("no thumb stands apart at the front of the hand")
    seeds = [seeds[i] for i in order]                  # thumb, index, middle, ring, pinky

    labels = -np.ones(n, dtype=np.int64)
    model = {"wrist": wrist.tolist(), "length": L, "cut": round(float(cut), 3), "fingers": {}}
    free = {}
    for fid, seed in enumerate(seeds):
        others = [s for j, s in enumerate(seeds) if j != fid]
        web, piece = cut, seed
        for f in np.arange(cut - 0.01, 0.2, -0.01):
            ps = pieces(n, E, bodymask & (along > f * L), 1)
            own = [p for p in ps if np.isin(seed[:1], p).any()]
            if not own or any(np.isin(o, own[0]).any() for o in others):
                break
            # ...and stop where the piece reaches the palm proper: a growth step that adds more
            # than the finger itself is (vertices) is the palm joining, not more finger
            if len(own[0]) > 2.0 * len(piece) and f < cut - 0.02:
                break
            web, piece = f, own[0]
        free[fid] = piece
    for fid, piece in free.items():
        labels[piece] = fid
    for sh in shells:                                  # nails: to the finger they sit on
        c = co[sh].mean(0)
        best = min(free, key=lambda fid: np.linalg.norm(co[free[fid]] - c, axis=1).min())
        if np.linalg.norm(co[free[best]] - c, axis=1).min() < 0.06 * L:
            labels[sh] = best

    inpiece = np.zeros(n, bool)
    for fid, piece in free.items():
        p = co[piece]
        inpiece[:] = False
        inpiece[piece] = True
        # the ring where the finger leaves the palm: its vertices with an edge into the palm body
        cross = E[inpiece[E[:, 0]] != inpiece[E[:, 1]]]
        ring = np.unique(np.where(inpiece[cross[:, 0]], cross[:, 0], cross[:, 1]))
        base = co[ring].mean(0) if len(ring) >= 3 else p[np.argsort(along[piece])[:10]].mean(0)
        far = co[piece[np.argmax(np.linalg.norm(p - wrist, axis=1))]]
        poly, ftip, d = _centreline(p, far, base)
        # thickness (the smaller) and width across the finger, from the surface points around the
        # middle of its free length (the spread of a ring is its half-axis / sqrt 2)
        mid_pt = poly[len(poly) // 2]
        dm = poly[len(poly) // 2 + 1] - poly[len(poly) // 2 - 1]
        dm /= np.linalg.norm(dm)
        sl = np.abs((p - mid_pt) @ dm) < 0.02 * L
        q = p[sl] - p[sl].mean(0)
        q = q - np.outer(q @ dm, dm)
        _, s, _ = np.linalg.svd(q, full_matrices=False)
        thick = float(2.0 * np.sqrt(2.0) * s[1] / np.sqrt(max(len(q), 1)))
        wide = float(2.0 * np.sqrt(2.0) * s[0] / np.sqrt(max(len(q), 1)))
        line = poly
        free_len = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
        d0 = line[min(2, len(line) - 1)] - line[0]
        d0 = d0 / np.linalg.norm(d0)
        name = FINGERS[fid]
        if fid == 0:
            mcp = line[0] - d0 * MCP_BACK * free_len
            chain = np.vstack([mcp, poly])
            ip, total = _arc_point(chain, THUMB_IP_AT * (free_len * (1 + MCP_BACK)))
            cmc = wrist + (mcp - wrist) * THUMB_CMC_AT
            joints = [cmc, mcp, ip, ftip]
        else:
            mcp = line[0] - d0 * MCP_BACK * free_len
            chain = np.vstack([mcp, poly])
            total = free_len * (1 + MCP_BACK)
            pip, _ = _arc_point(chain, PIP_AT * total)
            dip, _ = _arc_point(chain, DIP_AT * total)
            joints = [mcp, pip, dip, ftip]
        model["fingers"][name] = {
            "joints": [list(map(float, j)) for j in joints],
            "centreline": [list(map(float, x)) for x in chain],
            "thickness": thick, "width": wide, "free_length": free_len,
            "web_along": float(along[piece].min()), "vertices": int(len(piece)),
            "nail_vertices": int(sum(len(sh) for sh in shells if labels[sh[0]] == fid)),
        }
    model["frame"] = hand_frame(model, palm_down)
    return model, labels


def hand_frame(model, palm_down=(0.0, 0.0, -1.0)):
    f = model["fingers"]
    w = np.array(model["wrist"])
    mcp = {k: np.array(f[k]["joints"][0]) for k in ("index", "middle", "pinky")}
    h = mcp["middle"] - w
    h /= np.linalg.norm(h)
    k = mcp["index"] - mcp["pinky"]
    k = k - h * (k @ h)
    k /= np.linalg.norm(k)
    nn = np.cross(h, k)
    if nn @ np.asarray(palm_down) < 0:
        nn = -nn
    return {"h": h.tolist(), "k": k.tolist(), "n": nn.tolist()}


# --- weights -----------------------------------------------------------------------------------

def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _chain_param(p, joints):
    """Arc-length parameter of each point's nearest point on the joint polyline, and its distance."""
    best_d = np.full(len(p), np.inf)
    best_s = np.zeros(len(p))
    start = 0.0
    for a, b in zip(joints[:-1], joints[1:]):
        ab = b - a
        ln = float(np.linalg.norm(ab))
        t = np.clip(((p - a) @ ab) / (ln * ln), 0.0, 1.0)
        q = a + np.outer(t, ab)
        d = np.linalg.norm(p - q, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = start + t[better] * ln
        start += ln
    return best_s, best_d


def finger_split(p, joints, thick, proximal=True):
    """Shares of (proximal parent, bone 1, bone 2, bone 3) for points on a finger, blended at
    each joint. With proximal=False the points are wholly on the chain (no share to the parent)."""
    joints = [np.asarray(j) for j in joints]
    s, _ = _chain_param(p, joints)
    lens = [float(np.linalg.norm(b - a)) for a, b in zip(joints[:-1], joints[1:])]
    j1, j2 = lens[0], lens[0] + lens[1]
    b0, bj = MCP_BLEND * thick, JOINT_BLEND * thick
    # proximal side of the first joint: project along the first bone, not clamped
    d1 = (joints[1] - joints[0]) / lens[0]
    s0 = (p - joints[0]) @ d1
    on = _smooth((s0 + b0) / (2 * b0)) if proximal else np.ones(len(p))
    m2 = _smooth((s - (j1 - bj)) / (2 * bj))
    m3 = _smooth((s - (j2 - bj)) / (2 * bj))
    return np.stack([1.0 - on, on * (1.0 - m2), on * (m2 - m3), on * m3], axis=1)


def design_weights(points, labels, model):
    """Designed weights for one hand's points (already known to be past the wrist): columns
    [Hand, Thumb1-3, Index1-3, Middle1-3, Ring1-3, Pinky1-3] (16). Finger pieces and nails
    (labels >= 0) are wholly their finger's; the palm body is the hand's, except near each knuckle
    (shared with the nearest finger's first bone over the MCP blend) and over the thenar (the thumb's
    metacarpal)."""
    P = np.asarray(points, float)
    labels = np.asarray(labels)
    W = np.zeros((len(P), 16))
    F = model["fingers"]
    # palm body: memberships
    body = labels < 0
    member = np.zeros((len(P), 5))
    if body.any():
        q = P[body]
        g = np.zeros((len(q), 5))
        lat = np.zeros((len(q), 5))
        reach = np.zeros((len(q), 5))
        for fid in range(1, 5):
            fd = F[FINGERS[fid]]
            j = np.array(fd["joints"])
            d1 = (j[1] - j[0]) / np.linalg.norm(j[1] - j[0])
            s0 = (q - j[0]) @ d1
            b0 = MCP_BLEND * fd["thickness"]
            g[:, fid] = _smooth((s0 + b0) / (2 * b0))
            # distance to the finger itself (its chain, and its knuckle for points before it), so
            # nothing far from a finger — the thumb's root under the palm — is taken for it
            _, dist = _chain_param(q, j)
            lat[:, fid] = dist
            reach[:, fid] = _smooth((fd["width"] - dist) / (0.5 * fd["width"]))
        soft = LATERAL_SOFT * np.mean([F[FINGERS[i]]["thickness"] for i in range(1, 5)])
        e = np.exp(-(lat[:, 1:] - lat[:, 1:].min(axis=1, keepdims=True)) / soft)
        e /= e.sum(axis=1, keepdims=True)
        member[body, 1:] = g[:, 1:] * e * reach[:, 1:]
        # the thenar and the thumb's root: near the thumb's chain (CMC -> MCP -> IP), by its width
        # (the cartoon's thumb root is a mass hanging below the palm, as wide as the thumb)
        th = F["thumb"]
        j = np.array(th["joints"])
        s, r = _chain_param(q, j[:3])
        ln = float(np.linalg.norm(j[1] - j[0]))
        t = th["width"]
        near = _smooth((THENAR_OUT * t - r) / ((THENAR_OUT - THENAR_IN) * t))
        along = _smooth(s / (0.5 * ln))
        # ...and the slab that joins it to the palm: everything palmward of the palm's own palmar
        # level, on the thumb's side of the middle finger's line (measured in the hand frame)
        fr = model["frame"]
        w0, h, k, nn = (np.array(model["wrist"]), np.array(fr["h"]), np.array(fr["k"]), np.array(fr["n"]))
        qh, qk, qn = (q - w0) @ h, (q - w0) @ k, (q - w0) @ nn
        k_mid = float((np.array(F["middle"]["joints"][0]) - w0) @ k)
        mcp_h = min(float((np.array(F[f]["joints"][0]) - w0) @ h) for f in FINGERS[1:])
        core = (qk < k_mid) & (qh > 0.25 * mcp_h) & (qh < mcp_h)
        n_palm = float(np.percentile(qn[core], 90)) if core.sum() > 10 else 0.0
        band = SLAB_BLEND * t
        slab = _smooth((qn - n_palm) / band) * _smooth((qk - k_mid) / band) * _smooth(qh / (0.5 * ln))
        member[body, 0] = np.maximum(near * along, slab)
        model.setdefault("measured", {})["palm_palmar_level"] = n_palm
        tot = member[body].sum(axis=1)
        over = tot > 1.0
        if over.any():
            member[np.flatnonzero(body)[over]] /= tot[over, None]
    for fid in range(5):
        fd = F[FINGERS[fid]]
        joints = fd["joints"]
        cols = [0] + [1 + 3 * fid + i for i in range(3)]
        sel = labels == fid
        if sel.any():
            W[np.ix_(np.flatnonzero(sel), cols)] += finger_split(P[sel], joints, fd["thickness"], proximal=False)
        selb = body & (member[:, fid] > 0)
        if selb.any():
            sp = finger_split(P[selb], joints, fd["thickness"], proximal=False)
            W[np.ix_(np.flatnonzero(selb), cols)] += sp * member[selb, fid][:, None]
    W[:, 0] += 1.0 - W.sum(axis=1)
    return np.clip(W, 0.0, 1.0)


# --- posing: the hand's bones as rigid transforms in the hand's rest frame -----------------------

def rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def about(Rm, point):
    """4x4 rigid transform: rotate by Rm about `point`."""
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = point - Rm @ point
    return T


def chain_transforms(joints, rotations):
    """Armature-space deltas (4x4, applied to rest positions) of a three-bone chain whose bone b
    turns by rotations[b] (3x3, in the frame of the rest pose) about its own head."""
    out = []
    T = np.eye(4)
    for b in range(3):
        head = np.asarray(joints[b], float)
        head_now = (T @ np.append(head, 1.0))[:3]
        Rb = T[:3, :3] @ rotations[b] @ T[:3, :3].T          # the turn, carried by the parents
        T = about(Rb, head_now) @ T
        out.append(T.copy())
    return out


def skin_points(rest, W, deltas):
    """Linear blend skinning of rest points by 16 bone deltas (list of 4x4)."""
    hom = np.c_[rest, np.ones(len(rest))]
    out = np.zeros((len(rest), 3))
    for b, D in enumerate(deltas):
        wb = W[:, b]
        nz = wb > 0
        if nz.any():
            out[nz] += wb[nz, None] * (hom[nz] @ D.T)[:, :3]
    return out


def hand_deltas(model, angles):
    """All 16 deltas (Hand = identity) for per-finger joint rotations: angles[fid] = list of three
    3x3 rotations (rest frame)."""
    deltas = [np.eye(4)]
    for fid in range(5):
        joints = model["fingers"][FINGERS[fid]]["joints"]
        rs = angles.get(fid, [np.eye(3)] * 3)
        deltas += chain_transforms(joints, rs)
    return deltas


# --- the bar and the grip ----------------------------------------------------------------------

def bar_gap(points, B, k, r):
    rel = points - B
    radial = np.linalg.norm(rel - np.outer(rel @ k, k), axis=1)
    return radial - r


def bar_angle(points, B, k, h, n):
    """Angle of each point around the bar's axis, degrees in [0, 360): 0 toward the palm (-n),
    90 toward the fingers' direction (+h, over the top in the grip), 180 the far side, 270 toward
    the wrist."""
    rel = points - B
    a = np.degrees(np.arctan2(rel @ h, rel @ (-n)))
    return np.mod(a, 360.0)


def coverage(angles, edges=None, near=None):
    """How far round the bar's axis a patch of skin reaches, degrees: the longest contiguous arc
    covered. With `edges` (pairs of vertex indices) and `near` (which vertices count), every mesh
    edge with both ends near covers the arc between its two ends — the skin surface, not its
    vertices, which on these fingers sit ~15 degrees apart round a bar this thin. Without them,
    1-degree bins of the given angles."""
    occ = np.zeros(360, bool)
    angles = np.asarray(angles, float)
    if edges is None:
        if len(angles) == 0:
            return 0.0
        occ[(angles // 1.0).astype(int) % 360] = True
    else:
        e = edges[near[edges[:, 0]] & near[edges[:, 1]]]
        if len(e) == 0:
            return 0.0
        a, b = angles[e[:, 0]], angles[e[:, 1]]
        d = np.mod(b - a + 180.0, 360.0) - 180.0        # the short way round
        for a0, dd in zip(a, d):
            lo, hi = (a0, a0 + dd) if dd >= 0 else (a0 + dd, a0)
            occ[np.arange(int(math.floor(lo)), int(math.floor(hi)) + 1) % 360] = True
    if occ.all():
        return 360.0
    start = int(np.argmin(occ))
    best = run = 0
    for i in range(1, 361):
        if occ[(start + i) % 360]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return float(best)


def _rest_flex(joints, fold):
    """How much each joint is already folded at rest (radians, about `fold`): the angle from the
    previous bone's direction to the next, signed about the fold axis."""
    j = [np.asarray(x) for x in joints]
    out = [0.0]
    for b in (1, 2):
        u, v = j[b] - j[b - 1], j[b + 1] - j[b]
        u = u - fold * (u @ fold)
        v = v - fold * (v @ fold)
        out.append(math.atan2(np.cross(u, v) @ fold, u @ v))
    return out


def finger_rotations(model, fid, kb, theta, nn=None):
    """The three rotations (rest frame) of a finger closing on a bar along `kb`: at the knuckle,
    first a sideways turn (abduction, about the palm normal) that sets the finger square to the bar
    (the cartoon's relaxed fingers drift ~25 degrees toward the little finger), then the fold by
    theta[0]; at PIP and DIP, folds by theta[1], theta[2] about the finger's own hinge. +theta
    folds toward the palm side. Returns (rotations, the rest flex at PIP and DIP, abduction)."""
    fr = model["frame"]
    n = np.array(fr["n"]) if nn is None else nn
    j = [np.asarray(x) for x in model["fingers"][FINGERS[fid]]["joints"]]
    d = j[3] - j[0]
    d = d - n * (d @ n)
    d /= np.linalg.norm(d)
    hb = np.cross(n, kb)
    hb -= n * (hb @ n)
    hb /= np.linalg.norm(hb)
    if hb @ d < 0:
        hb = -hb
    psi = ABDUCT * math.atan2(np.cross(d, hb) @ n, d @ hb)
    d_sq = rot(n, psi) @ d                              # the finger's direction once turned
    fold_b = np.cross(d_sq, n)                          # +angle turns it toward n
    fold_b /= np.linalg.norm(fold_b)
    own = np.cross(d, n)                               # the finger's own hinge at rest
    base = _rest_flex(j, own)
    rs = [rot(fold_b, theta[0]) @ rot(n, psi), rot(own, theta[1]), rot(own, theta[2])]
    return rs, base, psi


WRAP_NEAR = 0.004             # m: a finger "touches" where its skin is within this of the bar
PALM_CLEAR = 0.0              # m: a finger's own skin keeps this far from the palm's (off: it cost wrap and the
                              # finger-palm crossings it was for are in the knuckle's crease, which it cannot see)
NEIGHBOUR_GIVE = 0.001        # m: how much nearer than at rest a finger may come to its neighbour (they press)
TIP_BEHIND = 300.0            # degrees round the bar: a fingertip past this has curled back, not gone round
TIP_FAR = 200.0               # degrees: no credit for a fingertip further round than this
TIP_TOUCH = 0.001             # m: a fingertip this near the bar is on it
OWN_SINK = 0.0008             # m: the most a finger's own skin may go into the bar
BLEND_SINK = 0.0022           # m: ... and a joint's blend zone, where skin creases
LIMITS = ((0.0, 100.0), (0.0, 110.0), (0.0, 90.0))    # degrees: knuckle from rest, PIP / DIP from straight


def tip_vertex(rest, W, model, fid):
    """The fingertip: of the vertices its last bone carries (weight > 0.5), the one furthest along
    that bone. The same vertex the acceptance measure reads."""
    j = [np.asarray(x) for x in model["fingers"][FINGERS[fid]]["joints"]]
    d = (j[3] - j[2]) / np.linalg.norm(j[3] - j[2])
    cand = np.flatnonzero(W[:, 1 + 3 * fid + 2] > 0.5)
    return int(cand[np.argmax((rest[cand] - j[2]) @ d)])


def _finger_score(posed, allow, own, B, kb, r, h, n, tip, edges):
    g = bar_gap(posed, B, kb, r)
    if (g < allow).any():
        return None
    near = own & (g < WRAP_NEAR)
    cov = coverage(bar_angle(posed, B, kb, h, n), edges, near) if near.any() else 0.0
    ta = float(bar_angle(posed[tip][None], B, kb, h, n)[0])
    if ta > TIP_BEHIND:
        ta -= 360.0                     # curled back toward the palm: behind where it started, not round
    snug = float(np.mean(np.clip(g[near], 0, None))) if near.any() else 1.0
    return cov, ta, snug, float(g[tip])


def wrap_finger(model, fid, rest, W, B, kb, r, angles, edges, coarse=10.0, fine=2.0):
    """The three fold angles that wrap one finger furthest round the bar: the finger is squared to
    the bar (`finger_rotations`), then (knuckle, PIP, DIP) are searched within LIMITS — a coarse
    grid, then a fine one round the best — for the largest angle round the bar's axis covered by
    the finger's own skin within WRAP_NEAR of the bar's surface, with no skin inside the bar beyond
    OWN_SINK (BLEND_SINK in the joints' blend zones). Ties go to the fingertip further round, then to
    the snugger wrap. Mutates angles[fid]; returns a dict of the result."""
    fr = model["frame"]
    h, n = np.array(fr["h"]), np.array(fr["n"])
    cols = [1 + 3 * fid + i for i in range(3)]
    sel = W[:, cols].sum(axis=1) > 1e-4
    Rs, Ws = rest[sel], W[sel]
    own = Ws[:, cols].sum(axis=1) >= 0.5
    allow = np.where(Ws.max(axis=1) < 0.8, -BLEND_SINK, -OWN_SINK)
    el = local_edges(edges, np.flatnonzero(sel), len(rest))
    tip = int(np.searchsorted(np.flatnonzero(sel), tip_vertex(rest, W, model, fid)))
    _, base, psi = finger_rotations(model, fid, kb, [0.0, 0.0, 0.0])
    off = [0.0, -base[1], -base[2]]                    # angles are measured from straight
    pivot = [tuple(model["fingers"][FINGERS[f]]["joints"][0]) for f in range(5)]
    others = {f: v for f, v in angles.items() if f != fid}

    # What this finger must not run into: the palm near its knuckle, and the neighbour already
    # closed (the index first, then each finger beside the last). Measured between the vertices each
    # part carries almost alone (weight >= 0.9), as posed; neighbours touch at rest, so theirs is held
    # to its rest distance less NEIGHBOUR_GIVE, the palm to PALM_CLEAR.
    core = Ws[:, cols].sum(axis=1) >= 0.9
    knuckle = np.asarray(model["fingers"][FINGERS[fid]]["joints"][0])
    palm_core = (W[:, 0] >= 0.98) & (np.linalg.norm(rest - knuckle, axis=1) < 0.02)
    blockers = [(skin_points(rest[palm_core], W[palm_core], hand_deltas(model, others)), PALM_CLEAR)]
    for nb in (fid - 1, fid + 1):
        if nb in others and nb >= 1:
            ncols = [1 + 3 * nb + i for i in range(3)]
            nc = W[:, ncols].sum(axis=1) >= 0.9
            pts = skin_points(rest[nc], W[nc], hand_deltas(model, others))
            d_rest = math.sqrt(((Rs[core][:, None, :] - rest[nc][None, :, :]) ** 2).sum(axis=2).min())
            blockers.append((pts, max(d_rest - NEIGHBOUR_GIVE, 0.0)))

    def evaluate(th):
        a = dict(others)
        a[fid] = finger_rotations(model, fid, kb, [math.radians(t) + o for t, o in zip(th, off)])[0]
        posed = skin_points(Rs, Ws, hand_deltas(model, a))
        mine = posed[core]
        for pts, least in blockers:
            if least > 0 and len(pts) and ((mine[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2).min() < least * least:
                return None
        return _finger_score(posed, allow, own, B, kb, r, h, n, tip, el)

    def rank(s):
        # one number: how far round the fingertip gets, plus half the skin wrapped, less 10 for each
        # mm the fingertip hovers off the bar beyond TIP_TOUCH. (Coverage first let a finger curled
        # into a fist on top of the bar win; the fingertip's angle first let one float over the far
        # side.) Ties: the snugger wrap.
        return (min(s[1], TIP_FAR) + 0.5 * s[0] - 10.0 * max(s[3] - TIP_TOUCH, 0.0) * 1000.0, -s[2])

    def better(s, best):
        if s is None:
            return False
        if best is None:
            return True
        return rank(s) > rank(best)

    best, best_th = None, None
    grids = [np.arange(lo, hi + 1e-9, coarse) for lo, hi in LIMITS]
    for t1 in grids[0]:
        for t2 in grids[1]:
            for t3 in grids[2]:
                s = evaluate((t1, t2, t3))
                if better(s, best):
                    best, best_th = s, (t1, t2, t3)
    if best_th is None:
        raise ValueError(f"no fold of the {FINGERS[fid]} keeps it out of the bar")
    centre = best_th
    span = coarse
    while span > fine:
        span /= 2.0
        for t1 in (centre[0] - span, centre[0], centre[0] + span):
            for t2 in (centre[1] - span, centre[1], centre[1] + span):
                for t3 in (centre[2] - span, centre[2], centre[2] + span):
                    th = tuple(float(np.clip(t, lo, hi)) for t, (lo, hi) in zip((t1, t2, t3), LIMITS))
                    s = evaluate(th)
                    if better(s, best):
                        best, best_th = s, th
        centre = best_th
    angles[fid] = finger_rotations(model, fid, kb, [math.radians(t) + o for t, o in zip(best_th, off)])[0]
    return {"fold_from_straight_deg": [round(t, 1) for t in best_th], "abduction_deg": round(math.degrees(psi), 1),
            "rest_flex_deg": [round(math.degrees(b), 1) for b in base[1:]],
            "coverage_deg": best[0], "tip_angle_deg": round(best[1] % 360.0, 1), "mean_gap_mm": round(best[2] * 1000, 2),
            "tip_gap_mm": round(best[3] * 1000, 2)}


def local_edges(edges, idx, n_total=None):
    """The mesh edges among the vertices `idx`, re-indexed to positions in `idx`."""
    idx = np.asarray(idx)
    n_total = int(max(edges.max(), idx.max())) + 1 if n_total is None else n_total
    pos = -np.ones(n_total, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    e = pos[edges]
    return e[(e >= 0).all(axis=1)]


def close_finger(model, fid, rest, W, B, kb, r, angles, edges, gap=0.0006, step=1.0):
    """Fold one finger onto the bar the way a hand closes, joint by joint from the knuckle out.
    It is squared to the bar and straightened, then each joint folds until skin it carries comes
    within `gap` of the bar's surface, or it reaches its limit (LIMITS, degrees from straight).
    "Comes within" ignores skin already resting on the bar when the joint starts: the finger's root
    lies on the bar from the start and barely moves.
    A fold also stops before any skin sinks into the bar further than allowed:
      - OWN_SINK for a bone's own skin;
      - BLEND_SINK in a crease, where no bone has 80% of the weight.
    `edges` are the mesh's (welded) edges indexed like `rest`, for measuring the wrap.
    Mutates angles[fid]; returns a dict of the result."""
    fr = model["frame"]
    h, n = np.array(fr["h"]), np.array(fr["n"])
    cols = [1 + 3 * fid + i for i in range(3)]
    sel = W[:, cols].sum(axis=1) > 1e-4
    Rs, Ws = rest[sel], W[sel]
    crease = Ws.max(axis=1) < 0.8
    allow = np.where(crease, -BLEND_SINK, -OWN_SINK)
    carried = [Ws[:, cols[b:]].sum(axis=1) >= 0.5 for b in range(3)]
    own = Ws[:, cols].sum(axis=1) >= 0.5
    tipj = np.asarray(model["fingers"][FINGERS[fid]]["joints"][3])
    tip = int(np.argmin(np.linalg.norm(Rs - tipj, axis=1)))
    _, base, psi = finger_rotations(model, fid, kb, [0.0, 0.0, 0.0])
    off = [0.0, -base[1], -base[2]]
    others = {f: v for f, v in angles.items() if f != fid}

    def gaps(th):
        a = dict(others)
        a[fid] = finger_rotations(model, fid, kb, [math.radians(t) + o for t, o in zip(th, off)])[0]
        return bar_gap(skin_points(Rs, Ws, hand_deltas(model, a)), B, kb, r)

    theta = [LIMITS[0][0], 0.0, 0.0]
    g0 = gaps(theta)
    if (g0 < allow).any():
        raise ValueError(f"the {FINGERS[fid]} is inside the bar before it folds")
    for b in range(3):
        lo, hi = LIMITS[b]
        theta[b] = lo
        resting = gaps(theta) <= gap + 0.0005
        watch = carried[b] & ~resting
        prev = lo
        for a in np.arange(lo + step, hi + 1e-9, step):
            theta[b] = a
            g = gaps(theta)
            if (g < allow).any() or (g[watch] <= gap).any():
                theta[b] = prev
                break
            prev = a
    g = gaps(theta)
    a = dict(others)
    a[fid] = finger_rotations(model, fid, kb, [math.radians(t) + o for t, o in zip(theta, off)])[0]
    posed = skin_points(Rs, Ws, hand_deltas(model, a))
    near = own & (g < WRAP_NEAR)
    cov = coverage(bar_angle(posed, B, kb, h, n), local_edges(edges, np.flatnonzero(sel), len(rest)), near)
    angles[fid] = a[fid]
    return {"fold_from_straight_deg": [round(t, 1) for t in theta], "abduction_deg": round(math.degrees(psi), 1),
            "rest_flex_deg": [round(math.degrees(x), 1) for x in base[1:]], "coverage_deg": cov,
            "tip_angle_deg": round(float(bar_angle(posed[tip][None], B, kb, h, n)[0]), 1),
            "deepest_mm": round(float(-g.min()) * 1000, 2)}


ABDUCT = 0.5                  # how much of each finger's sideways drift the knuckle takes out (all of it crossed neighbours: task 13)
THUMB_TIP_NEAR = 0.004        # m: the thumb's tip must end this close to the bar
THUMB_SEPARATION = 120.0      # degrees round the bar from the fingertips: "from the other side"


def thumb_rotations(model, kb, alpha, beta, th2, th3, nn=None):
    """Thumb: Thumb1 swings at the CMC by alpha about the bar's direction (+ carries the thumb
    toward the palm side, n) after turning by beta about the hand's long axis; Thumb2 and Thumb3
    fold by th2, th3 about the thumb's own hinge (the axis of its rest bend)."""
    fr = model["frame"]
    h = np.array(fr["h"])
    n = np.array(fr["n"]) if nn is None else nn
    fold = np.cross(np.cross(n, kb), n)
    fold /= np.linalg.norm(fold)
    hb = np.cross(n, kb)
    hb /= np.linalg.norm(hb)
    if hb @ h < 0:
        hb = -hb
    fold = np.cross(hb, n)                             # +angle turns hb toward n
    j = [np.asarray(x) for x in model["fingers"]["thumb"]["joints"]]
    u, v = j[1] - j[0], j[3] - j[1]
    hinge = np.cross(u, v)
    if np.linalg.norm(hinge) < 1e-9:
        hinge = fold
    hinge /= np.linalg.norm(hinge)
    return [rot(fold, alpha) @ rot(hb, beta), rot(hinge, th2), rot(hinge, th3)]


def solve_thumb(model, rest, W, B, kb, r, angles, edges, finger_tips_deg, keep=40):
    """The thumb onto the bar from the other side: a search over Thumb1's swing (alpha, beta) and
    the folds of Thumb2/Thumb3, keeping all its skin (its root under the palm included) out of the
    bar within OWN_SINK/BLEND_SINK and its tip within THUMB_TIP_NEAR of the bar. Preferred, in order:
    the tip at least THUMB_SEPARATION round the bar from the mean fingertip, the most skin wrapped,
    the smallest swing. The best `keep` are then checked against the posed fingers and palm, and the
    first whose thumb comes no nearer than THUMB_CLEAR to them is taken. Mutates angles[0]."""
    fr = model["frame"]
    h, n = np.array(fr["h"]), np.array(fr["n"])
    cols = [1, 2, 3]
    sel = W[:, cols].sum(axis=1) > 1e-4
    Rs, Ws = rest[sel], W[sel]
    own = Ws[:, cols].sum(axis=1) >= 0.5
    core = Ws[:, cols].sum(axis=1) >= 0.9
    allow = np.where(Ws.max(axis=1) < 0.8, -BLEND_SINK, -OWN_SINK)
    el = local_edges(edges, np.flatnonzero(sel), len(rest))
    tip = int(np.searchsorted(np.flatnonzero(sel), tip_vertex(rest, W, model, 0)))
    ft = np.radians(np.asarray(finger_tips_deg, float))
    mean_tip = math.degrees(math.atan2(np.sin(ft).mean(), np.cos(ft).mean())) % 360.0
    others = {f: v for f, v in angles.items() if f != 0}
    # the rest of the hand as posed: fingers and palm, the parts the thumb must not run into
    rest_part = (W[:, 4:].sum(axis=1) >= 0.9) | (W[:, 0] >= 0.9)
    rest_part &= ~sel
    other_pts = skin_points(rest[rest_part], W[rest_part], hand_deltas(model, others))
    found = []
    for alpha in np.arange(-60.0, 80.1, 10.0):
        for beta in np.arange(-90.0, 90.1, 15.0):
            for th2 in np.arange(-20.0, 80.1, 10.0):
                for th3 in np.arange(-20.0, 80.1, 10.0):
                    a = dict(others)
                    a[0] = thumb_rotations(model, kb, *(math.radians(x) for x in (alpha, beta, th2, th3)))
                    posed = skin_points(Rs, Ws, hand_deltas(model, a))
                    g = bar_gap(posed, B, kb, r)
                    if (g < allow).any() or g[tip] > THUMB_TIP_NEAR:
                        continue
                    ang = bar_angle(posed, B, kb, h, n)
                    sep = abs((ang[tip] - mean_tip + 180.0) % 360.0 - 180.0)
                    near = own & (g < WRAP_NEAR)
                    cov = coverage(ang, el, near)
                    # far enough round first; then the smaller swing (a big one folds the
                    # thumb's root into the palm); then the most wrapped
                    found.append(((min(sep, THUMB_SEPARATION), -round((abs(alpha) + abs(beta)) / 30.0), cov),
                                  (alpha, beta, th2, th3), sep, cov, float(ang[tip]), float(g[tip]), float(-g.min())))
    if not found:
        raise ValueError("no swing of the thumb keeps it out of the bar with its tip on it")
    found.sort(key=lambda t: t[0], reverse=True)
    for key, (alpha, beta, th2, th3), sep, cov, tang, tgap, deep in found[:keep]:
        a = dict(others)
        a[0] = thumb_rotations(model, kb, *(math.radians(x) for x in (alpha, beta, th2, th3)))
        posed = skin_points(Rs[core], Ws[core], hand_deltas(model, a))
        d2 = ((posed[:, None, :] - other_pts[None, :, :]) ** 2).sum(axis=2)
        clear = float(np.sqrt(d2.min()))
        if clear >= THUMB_CLEAR:
            angles[0] = a[0]
            return {"alpha_deg": alpha, "beta_deg": beta, "fold_deg": [th2, th3], "tip_angle_deg": round(tang, 1),
                    "separation_deg": round(float(sep), 1), "coverage_deg": cov, "tip_gap_mm": round(tgap * 1000, 2),
                    "deepest_mm": round(deep * 1000, 2), "clear_of_hand_mm": round(clear * 1000, 2),
                    "mean_fingertip_deg": round(mean_tip, 1), "candidates": len(found)}
    raise ValueError("every thumb that reaches the bar runs into the fingers or the palm")


THUMB_CLEAR = 0.0012          # m: the thumb's own skin keeps this far from the fingers' and palm's


# --- the grip, solved once per hand ------------------------------------------------------------

PLACE_A = (-12.0, -8.0, -4.0, 0.0, 4.0)   # mm from the middle knuckle along the hand: where the bar may cross
PLACE_BETA = (-10.0, -5.0, 0.0, 5.0, 10.0)  # degrees: the bar's tilt about the palm normal, from square to the hand
KNUCKLE_LIMIT = 12.0                      # degrees: the most the knuckle line may run off the bar
PALM_GAP = 0.0006                         # m: the bar's clearance from the palm


def bar_in_hand(model, rest, W, a, beta, r, gap=PALM_GAP):
    """The bar's line in the hand's rest frame for a placement: direction kb (the knuckle line's
    perpendicular square to h, tilted beta about n), crossing the hand `a` along hb from the wrist,
    resting on the palm (the nearest it can come to the palm's own skin, 100% hand-weighted,
    without going in). Returns (B, kb, hb, o)."""
    fr = model["frame"]
    w0, h, k, n = (np.array(model["wrist"]), np.array(fr["h"]), np.array(fr["k"]), np.array(fr["n"]))
    kb = rot(n, math.radians(beta)) @ k
    hb = np.cross(n, kb)
    hb /= np.linalg.norm(hb)
    if hb @ h < 0:
        hb = -hb
    palm = W[:, 0] >= 0.98
    ph, pn = (rest[palm] - w0) @ hb, (rest[palm] - w0) @ n
    m = np.abs(ph - a) < r + gap
    if not m.any():
        return None
    o = float((pn[m] + np.sqrt(np.maximum((r + gap) ** 2 - (ph[m] - a) ** 2, 0.0))).max())
    return w0 + a * hb + o * n, kb, hb, o


PALM_CENTRE_NEAR = 0.003      # m: the bar must lie this close to the middle of the palm


def palm_centre(model, rest, W):
    """The middle of the palm: its own skin (100% hand) facing the bar, in the middle half of the
    hand's width and between a quarter of the way to the knuckles and the knuckles."""
    w0 = np.array(model["wrist"])
    fr = model["frame"]
    h, k, n = (np.array(fr[q]) for q in "hkn")
    f = model["fingers"]
    ki, kp = ((np.array(f[x]["joints"][0]) - w0) @ k for x in ("index", "pinky"))
    klo, khi = min(ki, kp), max(ki, kp)
    span = khi - klo
    mcp_h = min((np.array(f[x]["joints"][0]) - w0) @ h for x in FINGERS[1:])
    rr = rest - w0
    return ((W[:, 0] >= 0.98) & (rr @ k > klo + 0.25 * span) & (rr @ k < khi - 0.25 * span)
            & (rr @ h > 0.25 * mcp_h) & (rr @ h < mcp_h) & (rr @ n > 0))


def knuckle_angle(model, kb):
    f = model["fingers"]
    kl = np.array(f["index"]["joints"][0]) - np.array(f["pinky"]["joints"][0])
    kl /= np.linalg.norm(kl)
    return math.degrees(math.acos(min(1.0, abs(float(kl @ kb)))))


def solve_grip(model, rest, W, edges, r, log=print):
    """The grip for one hand, in its rest frame: where the bar crosses the hand and how every finger
    and the thumb close on it. Placements (PLACE_A x PLACE_BETA, the knuckle line within
    KNUCKLE_LIMIT of the bar) are ranked by the least-wrapped finger, then the total; the thumb is
    solved for the best placement that admits it. `rest`/`W`/`edges` are the hand's vertices at the
    bind, their 16 hand weights and the welded mesh edges among them."""
    f = model["fingers"]
    w0, h = np.array(model["wrist"]), np.array(model["frame"]["h"])
    a_mid = float((np.array(f["middle"]["joints"][0]) - w0) @ h)
    tried = []
    for beta in PLACE_BETA:
        for da in PLACE_A:
            got = bar_in_hand(model, rest, W, a_mid + da / 1000.0, beta, r)
            if got is None:
                continue
            B, kb, hb, o = got
            if knuckle_angle(model, kb) > KNUCKLE_LIMIT:
                continue
            centre = palm_centre(model, rest, W)
            if centre.any() and bar_gap(rest[centre], B, kb, r).min() > PALM_CENTRE_NEAR:
                log(f"  grip: bar {da:+.0f} mm, tilt {beta:+.0f} deg refused: off the middle of the palm "
                    f"({bar_gap(rest[centre], B, kb, r).min() * 1000:.1f} mm)")
                continue
            angles, res = {}, {}
            try:
                for fid in (1, 2, 3, 4):
                    res[FINGERS[fid]] = wrap_finger(model, fid, rest, W, B, kb, r, angles, edges, coarse=15.0)
            except ValueError as err:
                log(f"  grip: bar {da:+.0f} mm, tilt {beta:+.0f} deg refused: {err}")
                continue
            covs = [res[x]["coverage_deg"] for x in FINGERS[1:]]
            tried.append(((min(covs), sum(covs)), beta, da, B, kb, hb, o, angles, res))
            log(f"  grip: bar {da:+.0f} mm from the middle knuckle, tilt {beta:+.0f} deg: wrap "
                + ", ".join(f"{x} {res[x]['coverage_deg']:.0f}" for x in FINGERS[1:]))
    if not tried:
        raise ValueError("no placement of the bar lets every finger close on it")
    tried.sort(key=lambda t: t[0], reverse=True)
    for score, beta, da, B, kb, hb, o, angles, res in tried:
        tips = [res[x]["tip_angle_deg"] for x in FINGERS[1:]]
        try:
            thumb = solve_thumb(model, rest, W, B, kb, r, angles, edges, tips)
        except ValueError:
            continue
        # the chosen placement, refined: each finger searched again on the finer grid
        angles2, res2 = {0: angles[0]}, {}
        for fid in (1, 2, 3, 4):
            res2[FINGERS[fid]] = wrap_finger(model, fid, rest, W, B, kb, r, angles2, edges)
        tips = [res2[x]["tip_angle_deg"] for x in FINGERS[1:]]
        thumb = solve_thumb(model, rest, W, B, kb, r, angles2, edges, tips)
        n = np.array(model["frame"]["n"])
        return {"B": B.tolist(), "kb": kb.tolist(), "hb": hb.tolist(), "n": n.tolist(),
                "wrist": w0.tolist(), "middle_mcp": list(f["middle"]["joints"][0]), "a_mm": round((a_mid + da / 1000.0) * 1000, 1), "o_mm": round(o * 1000, 1),
                "tilt_deg": beta, "knuckle_vs_bar_deg": round(knuckle_angle(model, kb), 1),
                "fingers": res2, "thumb": thumb,
                "rotations": {FINGERS[fid]: [np.asarray(m).tolist() for m in angles2[fid]] for fid in range(5)}}
    raise ValueError("no placement of the bar lets the thumb close on it too")


# --- the skin round the closed grip -------------------------------------------------------------
#
# Task 17. Every thumb that reaches this bar swings 77-90 degrees from the hand about its long axis
# (the thumb hangs where the bar goes), from under the palm out beside the index knuckle. The designed
# weights hand the thumb's root from the thumb to the hand within one 6-8 mm step, about 30 degrees
# round the CMC, and the swing turns the skin there over: the thumb's side wall sweeps through the back
# of the hand's skin (1,323 / 1,367 mm² on the boy, every frame: the grip is constant through the rep).
# Linear blending keeps skin the right way out round a turn only where its weights change over more
# angle than the turn. Wider designed blends alone did not get there (311 / 416 mm² at five times the
# width), and changing the design moves the bar's rest on the palm and the fingers' wraps, which the
# grip is solved on. So once the grip is solved, the hand's weights are smoothed where that grip turns
# the thumb's skin over: within RELAX_REACH x the thumb's width of a turned-over face the thumb's bones
# carry, RELAX_PASSES neighbour-average passes over the welded surface, round after round until none is
# turned over or two rounds no longer help (the best round is kept). Only the palm's skin wholly on the
# hand's bones moves: the fingers' and thumb's own pieces keep their weights, and so does the middle of
# the palm the bar rests on (`palm_centre`). The solved angles are kept; `grip_check` re-measures the
# grip on the smoothed skin. On the boy: the thumb root's largest patch 1,323 -> 36 mm² (left), 1,367 ->
# 41 (right); a reach of 1.0 x the width left 129 mm², 2.0 and 2.5 x give the same.

RELAX_REACH = 2.0             # x the thumb's width
RELAX_PASSES = 10
RELAX_ROUNDS = 8
MAX_BONES = 4                 # bones per vertex, as the bake (skin_wrap.MAX_INFLUENCES)
WEIGHT_FLOOR = 1e-3           # as the bake (skin_wrap.WEIGHT_FLOOR)


def weld_labels(rest):
    """One label per vertex; coincident vertices (the mesh is split at UV seams, and the bake gives each
    coincident group one rest position) share it."""
    _, lab = np.unique(np.round(np.asarray(rest) / 1e-7).astype(np.int64), axis=0, return_inverse=True)
    return lab.ravel()


def turned_over(rest, W, faces, deltas):
    """(turned over per face, posed area m²) for the hand posed by `deltas`, judged as the Studio does
    (studio/blender/inspect_run.py): the posed normal against the rest normal carried by the face's
    corners' blended bone rotations."""
    faces = np.asarray(faces)
    posed = skin_points(rest, W, deltas)
    R = np.stack([D[:3, :3] for D in deltas]).reshape(len(deltas), 9)
    rot = (W @ R).reshape(-1, 3, 3)
    r0 = np.cross(rest[faces[:, 1]] - rest[faces[:, 0]], rest[faces[:, 2]] - rest[faces[:, 0]])
    usable = np.linalg.norm(r0, axis=1) > 1e-14
    expect = np.einsum("fij,fj->fi", rot[faces].sum(axis=1), r0)
    cross = np.cross(posed[faces[:, 1]] - posed[faces[:, 0]], posed[faces[:, 2]] - posed[faces[:, 0]])
    area = np.linalg.norm(cross, axis=1) / 2
    return usable & (area > 1e-14) & (np.einsum("fi,fi->f", cross, expect) < 0), area


def patches(faces, flipped, area, weld, mark=None):
    """The turned-over patches, largest first, as (area m², any of its faces in `mark`): turned-over
    faces sharing a welded corner are one patch."""
    flip = np.flatnonzero(flipped)
    if len(flip) == 0:
        return []
    parent = np.arange(len(flip))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, f in enumerate(flip):
        for v in weld[np.asarray(faces)[f]]:
            j = owner.setdefault(int(v), i)
            a, b = root(i), root(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(len(flip))])
    marked = np.zeros(len(flip), bool) if mark is None else np.asarray(mark)[flip]
    out = [(float(area[flip][roots == r].sum()), bool(marked[roots == r].any())) for r in np.unique(roots)]
    return sorted(out, reverse=True)


def _limit(W):
    """At most MAX_BONES bones per vertex, none under WEIGHT_FLOOR, fading the smallest out (the bake's
    `_limit`, so nearly equal weights stay nearly equal)."""
    w = W.copy()
    if w.shape[1] > MAX_BONES:
        w = w - np.sort(w, axis=1)[:, -(MAX_BONES + 1)][:, None]
    w = np.maximum(w - WEIGHT_FLOOR, 0.0)
    return w / w.sum(axis=1, keepdims=True)


def relax_grip_folds(model, rest, W, faces, edges, movable, rotations, log=print):
    """The hand's weights (16 columns, rows summing to one) smoothed where the closed grip
    (`rotations`, as solve_grip returns them) turns the thumb's skin over; see the section's comment.
    `faces` and `edges` index `rest`; only `movable` vertices change, and never the palm's middle.
    Returns (weights, report)."""
    angles = {fid: [np.asarray(m) for m in rotations[FINGERS[fid]]] for fid in range(5)}
    deltas = hand_deltas(model, angles)
    faces = np.asarray(faces)
    movable = np.asarray(movable) & ~palm_centre(model, rest, W)
    weld = weld_labels(rest)
    n = int(weld.max()) + 1
    e = weld[np.asarray(edges)]
    e = np.unique(np.sort(e[e[:, 0] != e[:, 1]], axis=1), axis=0)
    first = np.zeros(n, np.int64)
    first[weld[::-1]] = np.arange(len(weld))[::-1]
    P = rest[first]
    c = 1.0 / np.maximum(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1), 1e-9)
    deg = np.bincount(e[:, 0], c, n) + np.bincount(e[:, 1], c, n)
    free = np.zeros(n, bool)
    np.logical_or.at(free, weld, movable)
    np.logical_and.at(free, weld, movable)        # a group moves only if all its members may
    thumb = (W[:, 1:4].sum(axis=1) > 1e-4)[faces].any(axis=1)     # faces the thumb's bones carry
    Wg = W[first].copy()
    reach = RELAX_REACH * model["fingers"]["thumb"]["width"]

    def measure(Wg):
        flipped, area = turned_over(rest, Wg[weld], faces, deltas)
        found = patches(faces, flipped, area, weld, thumb)
        at_thumb = max([a for a, m in found if m], default=0.0)
        return flipped, (at_thumb, float(area[flipped & thumb].sum())), found

    flipped, now, found = measure(Wg)
    before, largest_before = now, (found[0][0] if found else 0.0)
    best = (now, Wg.copy(), 0, found)
    for rnd in range(1, RELAX_ROUNDS + 1):
        seeds = flipped & thumb
        if not seeds.any():
            break
        seed_pts = P[np.unique(weld[faces[seeds]])]
        near = np.zeros(n, bool)
        for i in range(0, n, 2048):
            d = np.sqrt(((P[i:i + 2048, None, :] - seed_pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            near[i:i + 2048] = d < reach
        zone = near & free & (deg > 0)
        if not zone.any():
            break
        for _ in range(RELAX_PASSES):
            s = np.zeros_like(Wg)
            np.add.at(s, e[:, 0], c[:, None] * Wg[e[:, 1]])
            np.add.at(s, e[:, 1], c[:, None] * Wg[e[:, 0]])
            Wg[zone] = s[zone] / deg[zone, None]
        Wg[zone] = _limit(Wg[zone])
        flipped, now, found = measure(Wg)
        log(f"  grip skin: round {rnd}: {int(zone.sum())} points smoothed; the thumb's largest turned-over patch "
            f"{now[0] * 1e6:.0f} mm², all its turned-over skin {now[1] * 1e6:.0f} mm²")
        if now < best[0]:
            best = (now, Wg.copy(), rnd, found)
        elif rnd - best[2] >= 2:
            break
    (at_thumb, total), Wg, rnd, found = best
    out = Wg[weld]
    out[~movable] = W[~movable]
    changed = np.abs(out - W).sum(axis=1) / 2
    report = {"thumb_largest_before_mm2": round(before[0] * 1e6, 1), "thumb_all_before_mm2": round(before[1] * 1e6, 1),
              "thumb_largest_after_mm2": round(at_thumb * 1e6, 1), "thumb_all_after_mm2": round(total * 1e6, 1),
              "hand_largest_before_mm2": round(largest_before * 1e6, 1),
              "hand_largest_after_mm2": round((found[0][0] if found else 0.0) * 1e6, 1),
              "rounds": rnd, "changed_points": int((changed > 0.01).sum()), "most_changed": round(float(changed.max()), 3)}
    return out, report


def grip_check(model, rest, W, grip, r):
    """The grip re-measured on a skin's weights (16 columns): the deepest any of the hand's skin goes
    into the bar, the thumb's tip gap and how close its own skin (>= 0.9 thumb) comes to the rest of the
    hand, all in mm."""
    angles = {fid: [np.asarray(m) for m in grip["rotations"][FINGERS[fid]]] for fid in range(5)}
    posed = skin_points(rest, W, hand_deltas(model, angles))
    B, kb = np.asarray(grip["B"]), np.asarray(grip["kb"])
    g = bar_gap(posed, B, kb, r)
    tip = tip_vertex(rest, W, model, 0)
    th = W[:, 1:4].sum(axis=1)
    mine = posed[th >= 0.9]
    other = posed[(th < 1e-4) & ((W[:, 4:].sum(axis=1) >= 0.9) | (W[:, 0] >= 0.9))]
    clear = min(float(np.sqrt(((mine[i:i + 512, None, :] - other[None, :, :]) ** 2).sum(axis=2).min()))
                for i in range(0, len(mine), 512)) if len(mine) and len(other) else float("nan")
    return {"deepest_in_bar_mm": round(float(-g.min()) * 1000, 2), "thumb_tip_gap_mm": round(float(g[tip]) * 1000, 2),
            "thumb_clear_of_hand_mm": round(clear * 1000, 2)}


# --- Blender: the bones and their pose ----------------------------------------------------------

MIX = "mixamorig:"


def hand_bones(side):
    """The 16 bone names in this module's column order, for "Left"/"Right"."""
    return [MIX + side + "Hand"] + [MIX + side + "Hand" + f + str(i) for f in FINGER_NAMES for i in (1, 2, 3)]


def fit_finger_bones(rig, models):
    """Put each finger's three bones (and the thumb's) on the joints measured on the cartoon's own
    fingers (`measure`), rolled so their Z points to the back of the hand. The hand bone itself is
    left where the cage fit put it. The bind is the rest pose (identity), so moving these bones moves
    no skin; they only decide where the skin will bend."""
    import bpy
    from mathutils import Vector
    bpy.context.view_layer.objects.active = rig
    for obj in bpy.context.view_layer.objects:
        obj.select_set(obj is rig)
    inv = rig.matrix_world.inverted()
    bpy.ops.object.mode_set(mode="EDIT")
    edit = rig.data.edit_bones
    connected = {e.name: e.use_connect for e in edit}
    for e in edit:
        e.use_connect = False
    placed = {}
    for side, model in models.items():
        S = side.title()
        n = Vector(model["frame"]["n"])
        for fid, name in enumerate(FINGER_NAMES):
            j = [inv @ Vector(x) for x in model["fingers"][FINGERS[fid]]["joints"]]
            for i in range(3):
                e = edit[MIX + S + "Hand" + name + str(i + 1)]
                e.head, e.tail = j[i], j[i + 1]
                e.align_roll(-n)
                placed[e.name] = (list(e.head), list(e.tail))
    for e in edit:
        e.use_connect = connected[e.name] if e.name not in placed else (
            connected[e.name] and e.parent is not None and (e.parent.tail - e.head).length < 1e-6)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return placed


def pose_basis(rig, side, rotations):
    """Pose-bone quaternions (local basis) for the solved finger rotations: a bone b turned by R_b
    (rest frame, about its head, carried by its parents) has basis Rest_b^-1 R_b Rest_b."""
    from mathutils import Matrix
    out = {}
    S = side.title()
    for fid, name in enumerate(FINGER_NAMES):
        for i in range(3):
            bn = MIX + S + "Hand" + name + str(i + 1)
            rest = rig.data.bones[bn].matrix_local.to_3x3()
            Rb = Matrix([list(row) for row in rotations[FINGERS[fid]][i]])
            out[bn] = (rest.inverted() @ Rb @ rest).to_quaternion()
    return out


def hand_turn(grip, sign, bar_axis, forearm):
    """The hand's rotation from rest to a frame (3x3 numpy): the bar's direction in the hand (kb)
    onto the bar's axis (the index toward the middle of the bar: `sign` +1 for the right hand), then
    about that axis so the hand's long axis lies as close to the forearm as it can."""
    kb, h = np.array(grip["kb"]), np.array(grip["hb"])
    x = np.asarray(bar_axis, float)
    x = x / np.linalg.norm(x)
    target_k = -sign * x
    # first: kb onto target_k, by the shortest arc
    c = np.cross(kb, target_k)
    s, d = np.linalg.norm(c), float(kb @ target_k)
    R1 = np.eye(3) if s < 1e-9 else rot(c / s, math.atan2(s, d))
    # then about target_k: R1 h's part across the bar onto the forearm's part across the bar
    hh = R1 @ h
    v = np.asarray(forearm, float)
    u1 = hh - target_k * (hh @ target_k)
    u2 = v - target_k * (v @ target_k)
    ang = math.atan2(np.cross(u1, u2) @ target_k, u1 @ u2)
    return rot(target_k, ang) @ R1
