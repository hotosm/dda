"""Shape-preserving footprint regulariser; a fidelity IoU guard reverts any step that hurts a polygon."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.affinity import rotate, translate
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping
from shapely.ops import snap, unary_union
from shapely.strtree import STRtree

log = logging.getLogger(__name__)

# Metre-to-degree at the equator; sub-percent error at Venezuela latitudes.
DEG_PER_M = 1.0 / 111320.0


@dataclass(frozen=True)
class RegulariseParams:
    """Tunable knobs surfaced for HPO; every field carries a documented default."""

    min_area_m2: float = 5.0
    simplify_m: float = 2.0
    simplify_perimeter_pct: float = 0.03
    reflex_min_notch_m: float = 0.5
    ortho_min_area_ratio: float = 0.70
    ortho_45_tol_deg: float = 30.0
    pool_min_fraction: float = 0.50
    sliver_max_area_m2: float = 8.0
    sliver_max_aspect: float = 4.0
    multiblob_open_m: float = 2.0
    fidelity_min_iou: float = 0.70
    max_iterations: int = 5
    converge_tilted_pct_tol: float = 1.0


DEFAULT_PARAMS = RegulariseParams()

SPLIT_AREA_COVERAGE_MIN = 0.9
POOL_BLUE_OVER_RED = 15
POOL_BLUE_OVER_GREEN = 5
POOL_CYAN_H_LO = 170.0
POOL_CYAN_H_HI = 200.0
POOL_CYAN_S_MIN = 0.3
POOL_CYAN_V_MIN = 0.5
SALVAGE_MIN_AREA_RATIO = 0.7
SNAP_MAX_ITER = 10
EDGE_BUCKET_TOL_DEG = 5.0
SPIKE_DEG = 45.0
SPIKE_MAX_PASSES = 20
VERTEX_ADOPT_M = 2.5
SLIVER_MIN_ANGLE_DEG = 30.0
SLIVER_TRIANGLE_MAX_AREA_M2 = 5.0
SLIVER_ABSORB_PROBE_M = 0.5
OVERLAP_MAX_INNER_PASSES = 8
OVERLAP_MIN_M2 = 0.01


def regularise_footprints(
    gdf: gpd.GeoDataFrame,
    params: RegulariseParams | None = None,
    raster_path: str | None = None,
) -> gpd.GeoDataFrame:
    """Regularise EPSG:4326 footprints; guarantees zero overlaps and IoU >= fidelity_min_iou vs raw."""
    if not len(gdf):
        return gdf.copy()
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError(f"regularise_footprints expects EPSG:4326, got {gdf.crs}")

    p = params if params is not None else DEFAULT_PARAMS

    input_drops, valid_polys = _classify_input(gdf, p.min_area_m2)

    pool_prefilter_count = 0
    if raster_path is not None:
        keep_mask = _pool_keep_mask(valid_polys, raster_path, p.pool_min_fraction)
        pool_prefilter_count = sum(1 for k in keep_mask if not k)
        valid_polys = [poly for poly, k in zip(valid_polys, keep_mask, strict=True) if k]

    canonical = _split_multiblob(valid_polys, p.multiblob_open_m, p.min_area_m2)
    tagged: list[tuple[int, Polygon]] = list(enumerate(canonical))

    canonical_drops: dict[int, str] = {}
    absorbed_ids: set[int] = set()
    tilted_history: list[float] = []

    for pass_index in range(p.max_iterations):
        tagged = _apply_with_guard(
            lambda t: _simplify_all(t, p.simplify_m, p.simplify_perimeter_pct, p.reflex_min_notch_m),
            tagged,
            canonical,
            "simplify",
            p.fidelity_min_iou,
        )
        tagged = _apply_with_guard(
            lambda t: _orthogonalise_all(t, p.ortho_min_area_ratio, p.ortho_45_tol_deg),
            tagged,
            canonical,
            "ortho",
            p.fidelity_min_iou,
        )
        tagged = _apply_with_guard(
            _adopt_vertices,
            tagged,
            canonical,
            "adopt",
            p.fidelity_min_iou,
        )

        tagged, absorbed_now, sliver_dropped = _absorb_slivers(tagged, canonical, p)
        absorbed_ids |= absorbed_now
        for rid in sliver_dropped:
            canonical_drops.setdefault(rid, "sliver_dropped")

        if raster_path is not None:
            tagged, pool_post_dropped = _pool_postfilter(tagged, raster_path, p.pool_min_fraction)
            for rid in pool_post_dropped:
                canonical_drops.setdefault(rid, "pool_postfilter")

        tagged, overlap_dropped = _eliminate_overlaps(tagged, p.min_area_m2)
        for rid in overlap_dropped:
            canonical_drops.setdefault(rid, "overlap_dropped")

        tilted_pct = _tilted_percentage([poly for _, poly in tagged])
        tilted_history.append(tilted_pct)
        log.info("regularise: pass %d tilted=%.2f%% kept=%d", pass_index + 1, tilted_pct, len(tagged))
        if len(tilted_history) >= 2:
            delta = abs(tilted_history[-1] - tilted_history[-2])
            if delta < p.converge_tilted_pct_tol:
                break

    tagged, sub_quad_dropped = _drop_sub_quad(tagged)
    for rid in sub_quad_dropped:
        canonical_drops.setdefault(rid, "sub_quad_dropped")

    present_ids = {rid for rid, _ in tagged}
    missing_bug = [
        rid
        for rid in range(len(canonical))
        if rid not in present_ids and rid not in absorbed_ids and rid not in canonical_drops
    ]

    _log_run_summary(
        raw_count=len(gdf),
        input_drops=input_drops,
        pool_prefilter_count=pool_prefilter_count,
        canonical_count=len(canonical),
        canonical_drops=canonical_drops,
        absorbed_ids=absorbed_ids,
        output_count=len(tagged),
        tilted_history=tilted_history,
        missing_bug=missing_bug,
    )

    if missing_bug:
        preview = missing_bug[:20]
        raise AssertionError(
            f"regularise: {len(missing_bug)} canonical raw polygons missing without cause; "
            f"first ids: {preview}"
        )

    output_polys = [poly for _, poly in tagged]
    _log_quality_metrics(output_polys, canonical, [rid for rid, _ in tagged], p.reflex_min_notch_m)
    return gpd.GeoDataFrame(geometry=output_polys, crs=gdf.crs)


def _classify_input(gdf: gpd.GeoDataFrame, min_area_m2: float) -> tuple[dict[int, str], list[Polygon]]:
    """Split input into per-index drop reasons and a list of valid polygons."""
    input_drops: dict[int, str] = {}
    valid: list[Polygon] = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            input_drops[i] = "invalid_input"
            continue
        if _area_m2(geom) < min_area_m2:
            input_drops[i] = "sub_min_area"
            continue
        valid.append(geom)
    return input_drops, valid


def _split_multiblob(polys: list[Polygon], open_m: float, min_area_m2: float) -> list[Polygon]:
    """Symmetric morphological open; revert when pieces cover < SPLIT_AREA_COVERAGE_MIN of the source."""
    open_deg = open_m * DEG_PER_M
    out: list[Polygon] = []
    for poly in polys:
        try:
            opened = poly.buffer(-open_deg).buffer(open_deg)
        except GEOSException:
            # Numerical failure on skinny polys: keep the original as if no split occurred.
            out.append(poly)
            continue
        if opened.is_empty or opened.geom_type == "Polygon" or not isinstance(opened, MultiPolygon):
            out.append(poly)
            continue
        pieces = [g for g in opened.geoms if g.geom_type == "Polygon" and _area_m2(g) >= min_area_m2]
        if len(pieces) < 2:
            out.append(poly)
            continue
        coverage = sum(pc.area for pc in pieces) / poly.area
        if coverage < SPLIT_AREA_COVERAGE_MIN:
            out.append(poly)
            continue
        out.extend(pieces)
    return out


def _simplify_all(
    tagged: list[tuple[int, Polygon]],
    simplify_m: float,
    simplify_perimeter_pct: float,
    reflex_min_notch_m: float,
) -> list[tuple[int, Polygon]]:
    """Reflex-anchored DP simplify; tol is simplify_m capped by simplify_perimeter_pct*perimeter."""
    base_tol_deg = simplify_m * DEG_PER_M
    out: list[tuple[int, Polygon]] = []
    for rid, poly in tagged:
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            out.append((rid, poly))
            continue
        tol_deg = min(base_tol_deg, simplify_perimeter_pct * poly.length)
        out.append((rid, _simplify_preserving_reflex(poly, tol_deg, reflex_min_notch_m)))
    return out


def _simplify_preserving_reflex(poly: Polygon, tol_deg: float, reflex_min_notch_m: float) -> Polygon:
    """DP-simplify each convex arc between reflex anchors so real inward setbacks survive."""
    ring = list(poly.exterior.coords[:-1])
    if len(ring) < 4:
        return poly
    anchors = _reflex_indices(ring, poly.exterior.is_ccw, reflex_min_notch_m * DEG_PER_M)
    if not anchors:
        return _plain_simplify(poly, tol_deg)
    new_coords = _simplify_between_anchors(ring, anchors, tol_deg)
    if len(new_coords) < 4:
        return poly
    result = _build_polygon(new_coords, list(poly.interiors))
    if result is None:
        return _plain_simplify(poly, tol_deg)
    return result


def _plain_simplify(poly: Polygon, tol_deg: float) -> Polygon:
    try:
        simplified = poly.simplify(tol_deg, preserve_topology=True)
    except GEOSException:
        # GEOS simplify occasionally fails on borderline invalid rings; keep the untouched input.
        return poly
    if simplified is None or simplified.is_empty or simplified.geom_type != "Polygon":
        return poly
    return simplified


def _simplify_between_anchors(ring: list, anchors: list[int], tol_deg: float) -> list[tuple[float, float]]:
    """DP-simplify each arc bounded by two anchor indices, preserving the anchors."""
    new_coords: list[tuple[float, float]] = []
    n_anchors = len(anchors)
    for si in range(n_anchors):
        i0 = anchors[si]
        i1 = anchors[(si + 1) % n_anchors]
        segment = ring[i0 : i1 + 1] if i1 > i0 else ring[i0:] + ring[: i1 + 1]
        if len(segment) <= 2:
            new_coords.append(tuple(segment[0]))
            continue
        simplified_arc = LineString(segment).simplify(tol_deg, preserve_topology=False)
        new_coords.extend(tuple(v) for v in list(simplified_arc.coords)[:-1])
    return new_coords


def _orthogonalise_all(
    tagged: list[tuple[int, Polygon]],
    ortho_min_area_ratio: float,
    ortho_45_tol_deg: float,
) -> list[tuple[int, Polygon]]:
    """Snap every polygon to its dominant-axis 8-way grid; keep the pre-ortho polygon on failure."""
    out: list[tuple[int, Polygon]] = []
    for rid, poly in tagged:
        out.append((rid, _orthogonalise_polygon(poly, ortho_min_area_ratio, ortho_45_tol_deg)))
    return out


def _orthogonalise_polygon(poly: Polygon, ortho_min_area_ratio: float, ortho_45_tol_deg: float) -> Polygon:
    """8-way snap in the dominant-axis frame; salvage via longest-edge rigid rotation on area clip."""
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area <= 0:
        return poly
    snapped = _snap_polygon_to_8way(poly, ortho_min_area_ratio, ortho_45_tol_deg)
    if snapped is not None:
        return _remove_polygon_spikes(snapped)
    salvaged = _salvage_longest_edge(poly)
    if salvaged is not None:
        return _remove_polygon_spikes(salvaged)
    return poly


def _snap_polygon_to_8way(
    poly: Polygon, ortho_min_area_ratio: float, ortho_45_tol_deg: float
) -> Polygon | None:
    """8-way snap in the dominant-axis frame; None on area clip or validity break."""
    dominant = _dominant_angle(poly)
    origin = poly.centroid
    rotated = rotate(poly, -dominant, origin=origin)
    ext = list(rotated.exterior.coords[:-1])
    if len(ext) < 4:
        return None
    snapped_ext = _snap_ring_to_8way(ext, ortho_45_tol_deg)
    snapped_holes = [
        _snap_ring_to_8way(list(ring.coords[:-1]), ortho_45_tol_deg)
        if len(ring.coords) - 1 >= 4
        else list(ring.coords[:-1])
        for ring in rotated.interiors
    ]
    candidate = _build_polygon(snapped_ext, snapped_holes)
    if candidate is None or candidate.area < ortho_min_area_ratio * poly.area:
        return None
    rot_back = rotate(candidate, dominant, origin=origin)
    return _translate_to_centroid(rot_back, poly.centroid)


def _adopt_vertices(tagged: list[tuple[int, Polygon]]) -> list[tuple[int, Polygon]]:
    """Snap the less-rectilinear polygon of each overlapping pair onto the anchor, then difference."""
    n = len(tagged)
    if n < 2:
        return tagged
    ids = [rid for rid, _ in tagged]
    polys = [p for _, p in tagged]
    tree = STRtree(polys)
    tol_deg = VERTEX_ADOPT_M * DEG_PER_M
    seen: set[tuple[int, int]] = set()

    for i in range(n):
        p_i = polys[i]
        if p_i is None or p_i.is_empty or p_i.geom_type != "Polygon":
            continue
        probe = p_i.buffer(tol_deg)
        for jj in tree.query(probe):
            j = int(jj)
            if j == i:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in seen:
                continue
            seen.add(pair)
            a, b = polys[pair[0]], polys[pair[1]]
            if a is None or b is None or a.is_empty or b.is_empty:
                continue
            if a.geom_type != "Polygon" or b.geom_type != "Polygon":
                continue
            if not a.intersects(b) or a.touches(b):
                continue
            new_a, new_b = _resolve_pair_by_snap(a, b, tol_deg)
            polys[pair[0]] = _remove_polygon_spikes(new_a)
            polys[pair[1]] = _remove_polygon_spikes(new_b)

    return list(zip(ids, polys, strict=True))


def _resolve_pair_by_snap(a: Polygon, b: Polygon, tol_deg: float) -> tuple[Polygon, Polygon]:
    """Snap the less rectilinear follower onto the anchor and difference off residual overlap."""
    if _rectilinearity(a) >= _rectilinearity(b):
        anchor, follower, follower_is_b = a, b, True
    else:
        anchor, follower, follower_is_b = b, a, False
    new_follower = _snap_and_difference(anchor, follower, tol_deg)
    if new_follower is None:
        return a, b
    return (a, new_follower) if follower_is_b else (new_follower, b)


def _snap_and_difference(anchor: Polygon, follower: Polygon, tol_deg: float) -> Polygon | None:
    """Snap follower onto anchor's edges, then difference to remove overlap."""
    try:
        snapped = snap(follower, anchor, tol_deg)
        if snapped is None or snapped.is_empty or snapped.geom_type != "Polygon":
            return None
        final = snapped.difference(anchor)
        if final.is_empty:
            final = snapped
        if final.geom_type == "MultiPolygon":
            final = max(final.geoms, key=lambda g: g.area)
        if not final.is_valid:
            final = final.buffer(0)
    except GEOSException:
        # snap/difference/buffer0 all propagate GEOS numerical failure; skip the pair.
        return None
    if final.is_empty or final.geom_type != "Polygon":
        return None
    return final


def _absorb_slivers(
    tagged: list[tuple[int, Polygon]],
    canonical: list[Polygon],
    params: RegulariseParams,
) -> tuple[list[tuple[int, Polygon]], set[int], list[int]]:
    """Absorb thin residues into the neighbour sharing their longest edge; unabsorbed slivers are dropped."""
    ids = [rid for rid, _ in tagged]
    polys = [p for _, p in tagged]
    sliver_flags = [_is_sliver(p, params.sliver_max_area_m2, params.sliver_max_aspect) for p in polys]
    if not any(sliver_flags):
        return tagged, set(), []

    non_sliver_positions = [k for k, s in enumerate(sliver_flags) if not s]
    non_sliver_polys = [polys[k] for k in non_sliver_positions]
    tree = STRtree(non_sliver_polys) if non_sliver_polys else None
    probe_deg = SLIVER_ABSORB_PROBE_M * DEG_PER_M

    absorbed: set[int] = set()
    dropped: list[int] = []
    keep = [True] * len(polys)

    for k, is_sliv in enumerate(sliver_flags):
        if not is_sliv:
            continue
        sliver_poly = polys[k]
        if tree is None:
            keep[k] = False
            dropped.append(ids[k])
            continue
        neighbour_pos = _best_shared_neighbour(sliver_poly, tree, non_sliver_polys, probe_deg)
        if neighbour_pos < 0:
            keep[k] = False
            dropped.append(ids[k])
            continue
        neighbour_k = non_sliver_positions[neighbour_pos]
        merged = _merge_into(polys[neighbour_k], sliver_poly)
        if merged is None or _iou(merged, canonical[ids[neighbour_k]]) < params.fidelity_min_iou:
            keep[k] = False
            dropped.append(ids[k])
            continue
        polys[neighbour_k] = merged
        keep[k] = False
        absorbed.add(ids[k])

    kept = [(ids[k], polys[k]) for k in range(len(ids)) if keep[k]]
    return kept, absorbed, dropped


def _best_shared_neighbour(
    sliver: Polygon, tree: STRtree, candidates: list[Polygon], probe_deg: float
) -> int:
    """Index in candidates sharing the longest common boundary with the sliver, or -1."""
    probe = sliver.buffer(probe_deg)
    best_pos = -1
    best_shared = 0.0
    for idx in tree.query(probe):
        pos = int(idx)
        neighbour = candidates[pos]
        if neighbour is sliver or neighbour is None or neighbour.is_empty:
            continue
        shared = neighbour.boundary.intersection(probe).length
        if shared > best_shared:
            best_shared = shared
            best_pos = pos
    return best_pos


def _merge_into(neighbour: Polygon, sliver: Polygon) -> Polygon | None:
    try:
        merged = unary_union([neighbour, sliver])
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        if not merged.is_valid:
            merged = merged.buffer(0)
    except GEOSException:
        # unary_union / buffer0 GEOS failure: skip absorption for this sliver.
        return None
    if merged.is_empty or merged.geom_type != "Polygon":
        return None
    return merged


def _pool_postfilter(
    tagged: list[tuple[int, Polygon]], raster_path: str, pool_min_fraction: float
) -> tuple[list[tuple[int, Polygon]], list[int]]:
    """Reapply the pool signature to catch polys that grew onto pool pixels."""
    keep_mask = _pool_keep_mask([p for _, p in tagged], raster_path, pool_min_fraction)
    kept = [(rid, p) for (rid, p), k in zip(tagged, keep_mask, strict=True) if k]
    dropped = [rid for (rid, _), k in zip(tagged, keep_mask, strict=True) if not k]
    return kept, dropped


def _drop_sub_quad(
    tagged: list[tuple[int, Polygon]],
) -> tuple[list[tuple[int, Polygon]], list[int]]:
    """Drop polygons with fewer than four exterior vertices."""
    kept: list[tuple[int, Polygon]] = []
    dropped: list[int] = []
    for rid, poly in tagged:
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            dropped.append(rid)
            continue
        if len(poly.exterior.coords) - 1 < 4:
            dropped.append(rid)
            continue
        kept.append((rid, poly))
    return kept, dropped


def _eliminate_overlaps(
    tagged: list[tuple[int, Polygon]], min_area_m2: float
) -> tuple[list[tuple[int, Polygon]], list[int]]:
    """Subtract the smaller polygon of each overlapping pair from the larger; iterates to zero overlap."""
    ids = [rid for rid, _ in tagged]
    polys: list[Polygon | None] = [p for _, p in tagged]
    dropped_ids: list[int] = []

    for _ in range(OVERLAP_MAX_INNER_PASSES):
        pairs = _find_overlap_pairs(polys)
        if not pairs:
            break
        for i, j in pairs:
            pi, pj = polys[i], polys[j]
            if pi is None or pj is None:
                continue
            if not pi.intersects(pj) or pi.touches(pj):
                continue
            small_k, big_k = (i, j) if pi.area <= pj.area else (j, i)
            big, small = polys[big_k], polys[small_k]
            if big is None or small is None:
                continue
            new_big = _subtract_and_validate(big, small, min_area_m2)
            if new_big is None:
                dropped_ids.append(ids[big_k])
                polys[big_k] = None
                continue
            polys[big_k] = new_big

    remaining = _find_overlap_pairs(polys)
    if remaining:
        raise RuntimeError(
            f"regularise: overlap elimination failed to reach zero; remaining={len(remaining)}"
        )

    kept: list[tuple[int, Polygon]] = [
        (ids[k], p) for k, p in enumerate(polys) if p is not None and _is_active_polygon(p)
    ]
    return kept, dropped_ids


def _subtract_and_validate(big: Polygon, small: Polygon, min_area_m2: float) -> Polygon | None:
    """big.difference(small); None when the result would drop below min_area_m2."""
    try:
        result = big.difference(small)
    except GEOSException:
        # GEOS difference failure on overlapping pair: drop the larger polygon.
        return None
    result = _largest_polygon_or_none(result)
    if result is None or _area_m2(result) < min_area_m2:
        return None
    return result


def _largest_polygon_or_none(geom) -> Polygon | None:
    """Largest Polygon component of geom (buffer(0)-repaired if invalid), or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return None
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except GEOSException:
            # buffer(0) failed to repair a self-intersection: caller treats as drop.
            return None
    if geom.is_empty or geom.geom_type != "Polygon":
        return None
    return geom


def _find_overlap_pairs(polys: list[Polygon | None]) -> list[tuple[int, int]]:
    """Unordered index pairs whose polygons overlap by more than OVERLAP_MIN_M2."""
    active: list[tuple[int, Polygon]] = [
        (k, p) for k, p in enumerate(polys) if p is not None and _is_active_polygon(p)
    ]
    if len(active) < 2:
        return []
    active_polys = [p for _, p in active]
    active_idx = [k for k, _ in active]
    tree = STRtree(active_polys)
    area_thresh_deg2 = OVERLAP_MIN_M2 * (DEG_PER_M**2)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for m, p in enumerate(active_polys):
        for jj in tree.query(p):
            m2 = int(jj)
            if m2 == m:
                continue
            i, j = active_idx[m], active_idx[m2]
            pair = (i, j) if i < j else (j, i)
            if pair in seen:
                continue
            seen.add(pair)
            pi, pj = polys[pair[0]], polys[pair[1]]
            if pi is None or pj is None:
                continue
            if not pi.intersects(pj) or pi.touches(pj):
                continue
            inter = pi.intersection(pj)
            if inter.is_empty or inter.area <= area_thresh_deg2:
                continue
            pairs.append(pair)
    return pairs


def _apply_with_guard(
    step_fn: Callable[[list[tuple[int, Polygon]]], list[tuple[int, Polygon]]],
    tagged: list[tuple[int, Polygon]],
    canonical: list[Polygon],
    step_name: str,
    fidelity_min_iou: float,
) -> list[tuple[int, Polygon]]:
    """Revert any polygon whose IoU drops below fidelity_min_iou when the pre-step scored strictly better."""
    before_by_id: dict[int, Polygon] = {rid: p for rid, p in tagged if _is_active_polygon(p)}
    after = step_fn(tagged)
    out: list[tuple[int, Polygon]] = []
    reverted = 0
    for rid, poly in after:
        raw = canonical[rid]
        if not _is_active_polygon(poly):
            fallback = before_by_id.get(rid)
            if fallback is not None:
                out.append((rid, fallback))
                reverted += 1
            continue
        iou_after = _iou(poly, raw)
        if iou_after < fidelity_min_iou:
            fallback = before_by_id.get(rid)
            if fallback is not None and _iou(fallback, raw) > iou_after:
                out.append((rid, fallback))
                reverted += 1
                continue
        out.append((rid, poly))
    log.info("regularise: guard %s reverted=%d kept=%d", step_name, reverted, len(out))
    return out


def _log_run_summary(
    *,
    raw_count: int,
    input_drops: dict[int, str],
    pool_prefilter_count: int,
    canonical_count: int,
    canonical_drops: dict[int, str],
    absorbed_ids: set[int],
    output_count: int,
    tilted_history: list[float],
    missing_bug: list[int],
) -> None:
    """Log drop attribution, convergence history, and iteration count."""
    invalid_input = sum(1 for r in input_drops.values() if r == "invalid_input")
    sub_min = sum(1 for r in input_drops.values() if r == "sub_min_area")
    sliver_dropped = sum(1 for r in canonical_drops.values() if r == "sliver_dropped")
    pool_post = sum(1 for r in canonical_drops.values() if r == "pool_postfilter")
    overlap_dropped = sum(1 for r in canonical_drops.values() if r == "overlap_dropped")
    sub_quad = sum(1 for r in canonical_drops.values() if r == "sub_quad_dropped")
    log.info(
        "regularise: drop attribution raw=%d invalid=%d sub_min=%d pool_pre=%d "
        "canonical=%d sliver_absorbed=%d sliver_dropped=%d pool_post=%d overlap_dropped=%d "
        "sub_quad=%d missing_bug=%d output=%d",
        raw_count,
        invalid_input,
        sub_min,
        pool_prefilter_count,
        canonical_count,
        len(absorbed_ids),
        sliver_dropped,
        pool_post,
        overlap_dropped,
        sub_quad,
        len(missing_bug),
        output_count,
    )
    tilted_str = ", ".join(f"{v:.2f}" for v in tilted_history)
    log.info("regularise: tilted%% per pass = [%s] passes=%d", tilted_str, len(tilted_history))


def _log_quality_metrics(
    polys: list[Polygon],
    canonical: list[Polygon],
    ids: list[int],
    reflex_min_notch_m: float,
) -> None:
    """Log IoU distribution vs canonical raws, edge-angle buckets, and reflex counts."""
    if not polys:
        return
    ious = np.array([_iou(p, canonical[rid]) for rid, p in zip(ids, polys, strict=True)])
    buckets, non_aligned = _edge_angle_summary(polys, EDGE_BUCKET_TOL_DEG)
    log.info(
        "regularise: fidelity IoU vs raw n=%d mean=%.3f median=%.3f p5=%.3f p1=%.3f "
        "count<0.7=%d count<0.5=%d",
        len(ious),
        float(ious.mean()),
        float(np.median(ious)),
        float(np.percentile(ious, 5)),
        float(np.percentile(ious, 1)),
        int((ious < 0.7).sum()),
        int((ious < 0.5).sum()),
    )
    log.info(
        "regularise: edge-angle buckets %s; polys with any tilted edge=%d (%.1f%%)",
        {k: v for k, v in buckets.items()},
        non_aligned,
        100.0 * non_aligned / max(len(polys), 1),
    )
    reflex_counts = np.array([_count_reflex_vertices(p, reflex_min_notch_m) for p in polys])
    log.info(
        "regularise: reflex-vertex distribution (notch>=%.1fm) n=%d mean=%.2f median=%d p95=%d "
        "max=%d count>=1=%d (%.1f%%)",
        reflex_min_notch_m,
        len(reflex_counts),
        float(reflex_counts.mean()),
        int(np.median(reflex_counts)),
        int(np.percentile(reflex_counts, 95)),
        int(reflex_counts.max()),
        int((reflex_counts >= 1).sum()),
        100.0 * float((reflex_counts >= 1).mean()),
    )


def _area_m2(poly: Polygon) -> float:
    return poly.area / (DEG_PER_M * DEG_PER_M)


def _iou(a: Polygon | None, b: Polygon | None) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        if inter == 0.0:
            return 0.0
        union = a.union(b).area
    except GEOSException:
        # GEOS boolean failure feeds the fidelity guard as no-overlap, so guard falls back safely.
        return 0.0
    if union == 0.0:
        return 0.0
    return inter / union


def _is_active_polygon(poly: Polygon | None) -> bool:
    return poly is not None and not poly.is_empty and poly.geom_type == "Polygon"


def _dominant_angle(poly: Polygon) -> float:
    """Length-weighted mean edge angle folded to [0, 90) via 4x circular mean."""
    coords = list(poly.exterior.coords[:-1])
    sx = sy = 0.0
    n = len(coords)
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        length = math.hypot(x1 - x0, y1 - y0)
        angle_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
        r = math.radians((angle_deg % 90) * 4)
        sx += length * math.cos(r)
        sy += length * math.sin(r)
    return math.degrees(math.atan2(sy, sx)) / 4


def _reflex_indices(ring: list, is_ccw: bool, min_notch_deg: float) -> list[int]:
    """Indices of reflex vertices whose perpendicular notch depth meets min_notch_deg."""
    n = len(ring)
    out: list[int] = []
    for i in range(n):
        prev = ring[(i - 1) % n]
        curr = ring[i]
        nxt = ring[(i + 1) % n]
        e1x, e1y = curr[0] - prev[0], curr[1] - prev[1]
        e2x, e2y = nxt[0] - curr[0], nxt[1] - curr[1]
        cross = e1x * e2y - e1y * e2x
        is_reflex = (is_ccw and cross < 0) or ((not is_ccw) and cross > 0)
        if not is_reflex:
            continue
        chord_x, chord_y = nxt[0] - prev[0], nxt[1] - prev[1]
        chord_len = math.hypot(chord_x, chord_y)
        if chord_len == 0.0:
            continue
        perp = abs((curr[0] - prev[0]) * chord_y - (curr[1] - prev[1]) * chord_x) / chord_len
        if perp >= min_notch_deg:
            out.append(i)
    return out


def _count_reflex_vertices(poly: Polygon, min_notch_m: float) -> int:
    if poly is None or poly.is_empty or poly.geom_type != "Polygon":
        return 0
    ring = list(poly.exterior.coords[:-1])
    if len(ring) < 3:
        return 0
    return len(_reflex_indices(ring, poly.exterior.is_ccw, min_notch_m * DEG_PER_M))


def _snap_ring_to_8way(ring: list, ortho_45_tol_deg: float, max_iter: int = SNAP_MAX_ITER) -> list:
    """Iteratively snap every edge to 0/45/90/135 deg; terminates on convergence or max_iter."""
    n = len(ring)
    if n < 3:
        return ring
    diag_half = ortho_45_tol_deg / 2.0
    diag_lo = 45.0 - diag_half
    diag_hi = 45.0 + diag_half
    diag_lo_alt = 180.0 - diag_hi
    diag_hi_alt = 180.0 - diag_lo
    new_ring = [tuple(v) for v in ring]
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            j = (i + 1) % n
            lx, ly = new_ring[i]
            cx, cy = new_ring[j]
            dx, dy = cx - lx, cy - ly
            if dx == 0.0 and dy == 0.0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            angle_mod = angle % 180
            if diag_lo <= angle_mod <= diag_hi:
                target = 45.0
            elif diag_lo_alt <= angle_mod <= diag_hi_alt:
                target = 135.0
            else:
                horiz = min(angle_mod, 180.0 - angle_mod)
                vert = abs(angle_mod - 90.0)
                target = 0.0 if horiz <= vert else 90.0
            rad = math.radians(target)
            tx, ty = math.cos(rad), math.sin(rad)
            proj = dx * tx + dy * ty
            if proj < 0.0:
                proj = -proj
                tx, ty = -tx, -ty
            new_tail = (lx + proj * tx, ly + proj * ty)
            if abs(new_tail[0] - cx) > 1e-12 or abs(new_tail[1] - cy) > 1e-12:
                new_ring[j] = new_tail
                moved = True
        if not moved:
            break
    return new_ring


def _build_polygon(ext: list, holes: list[list]) -> Polygon | None:
    try:
        candidate = Polygon(ext, holes=holes)
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
    except (ValueError, GEOSException):
        # Construction/repair failed on a degenerate ring; caller treats as no-op.
        return None
    if candidate.is_empty or candidate.geom_type != "Polygon":
        return None
    return candidate


def _translate_to_centroid(poly: Polygon, target_centroid) -> Polygon:
    c = poly.centroid
    dx = target_centroid.x - c.x
    dy = target_centroid.y - c.y
    if dx == 0.0 and dy == 0.0:
        return poly
    return translate(poly, xoff=dx, yoff=dy)


def _salvage_longest_edge(poly: Polygon) -> Polygon | None:
    """Rigid rotation onto nearest axis via longest edge; None when area < SALVAGE_MIN_AREA_RATIO * input."""
    ring = list(poly.exterior.coords[:-1])
    longest_len = 0.0
    longest_angle = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > longest_len:
            longest_len = length
            longest_angle = math.degrees(math.atan2(dy, dx))
    if longest_len == 0.0:
        return None
    delta = _axis_delta(longest_angle)
    if abs(delta) < 1e-9:
        return poly
    rotated = rotate(poly, delta, origin=poly.centroid)
    if not rotated.is_valid:
        rotated = rotated.buffer(0)
    if rotated.is_empty or rotated.geom_type != "Polygon":
        return None
    if rotated.area < SALVAGE_MIN_AREA_RATIO * poly.area:
        return None
    return rotated


def _axis_delta(angle_deg: float) -> float:
    """Signed rotation in [-90, 90] deg to bring angle_deg to the nearest axis."""
    a = angle_deg % 180.0
    target = 0.0 if a <= 45.0 else (180.0 if a >= 135.0 else 90.0)
    delta = target - angle_deg
    while delta > 90.0:
        delta -= 180.0
    while delta < -90.0:
        delta += 180.0
    return delta


def _remove_polygon_spikes(poly: Polygon) -> Polygon:
    """Drop vertices whose incident edges diverge by less than SPIKE_DEG."""
    if not _is_active_polygon(poly):
        return poly
    cos_thr = math.cos(math.radians(SPIKE_DEG))
    ext_raw = list(poly.exterior.coords[:-1])
    ext, dropped = _drop_spikes_from_ring(ext_raw, cos_thr)
    if len(ext) < 3:
        return poly
    new_interiors = []
    for ring in poly.interiors:
        raw = list(ring.coords[:-1])
        cleaned, ring_dropped = _drop_spikes_from_ring(raw, cos_thr)
        dropped += ring_dropped
        if len(cleaned) >= 3:
            new_interiors.append(cleaned)
    if dropped == 0:
        return poly
    candidate = _build_polygon(ext, new_interiors)
    return candidate if candidate is not None else poly


def _drop_spikes_from_ring(ring: list, cos_thr: float) -> tuple[list, int]:
    """Iteratively drop vertices whose cos(neighbour_angle) > cos_thr."""
    dropped_total = 0
    for _ in range(SPIKE_MAX_PASSES):
        n = len(ring)
        if n < 3:
            return ring, dropped_total
        new_ring: list = []
        pass_dropped = 0
        for i in range(n):
            prev = ring[(i - 1) % n]
            curr = ring[i]
            nxt = ring[(i + 1) % n]
            ax, ay = prev[0] - curr[0], prev[1] - curr[1]
            bx, by = nxt[0] - curr[0], nxt[1] - curr[1]
            la = math.hypot(ax, ay)
            lb = math.hypot(bx, by)
            if la == 0.0 or lb == 0.0:
                pass_dropped += 1
                continue
            cos_a = (ax * bx + ay * by) / (la * lb)
            if cos_a > cos_thr:
                pass_dropped += 1
                continue
            new_ring.append(curr)
        if pass_dropped == 0:
            return ring, dropped_total
        ring = new_ring
        dropped_total += pass_dropped
    return ring, dropped_total


def _rectilinearity(poly: Polygon) -> float:
    """Fraction of exterior edge length aligned to the polygon's dominant axis."""
    if not _is_active_polygon(poly):
        return 0.0
    dominant = _dominant_angle(poly)
    ring = list(poly.exterior.coords[:-1])
    total = 0.0
    aligned = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        total += length
        local = (math.degrees(math.atan2(dy, dx)) - dominant) % 180.0
        horiz = min(local, 180.0 - local)
        vert = abs(local - 90.0)
        if min(horiz, vert) <= EDGE_BUCKET_TOL_DEG:
            aligned += length
    if total == 0.0:
        return 0.0
    return aligned / total


def _edge_angle_summary(polys: list, tol_deg: float) -> tuple[dict, int]:
    buckets: dict = {0: 0, 45: 0, 90: 0, 135: 0, 180: 0, 225: 0, 270: 0, 315: 0, "tilted": 0}
    non_aligned = 0
    for poly in polys:
        if not _is_active_polygon(poly):
            continue
        dominant = _dominant_angle(poly)
        ring = list(poly.exterior.coords[:-1])
        has_tilted = False
        for i in range(len(ring)):
            x0, y0 = ring[i]
            x1, y1 = ring[(i + 1) % len(ring)]
            dx, dy = x1 - x0, y1 - y0
            if math.hypot(dx, dy) == 0.0:
                continue
            local = (math.degrees(math.atan2(dy, dx)) - dominant) % 360.0
            nearest = int(round(local / 45.0) * 45) % 360
            diff = abs(local - nearest)
            diff = min(diff, 360.0 - diff)
            if diff <= tol_deg:
                buckets[nearest] += 1
            else:
                buckets["tilted"] += 1
                has_tilted = True
        if has_tilted:
            non_aligned += 1
    return buckets, non_aligned


def _tilted_percentage(polys: list[Polygon]) -> float:
    if not polys:
        return 0.0
    _, non_aligned = _edge_angle_summary(polys, EDGE_BUCKET_TOL_DEG)
    return 100.0 * non_aligned / len(polys)


def _is_sliver(poly: Polygon, sliver_max_area_m2: float, sliver_max_aspect: float) -> bool:
    if not _is_active_polygon(poly):
        return False
    area_m2 = _area_m2(poly)
    vertex_count = len(poly.exterior.coords) - 1
    if area_m2 < SLIVER_TRIANGLE_MAX_AREA_M2 and vertex_count <= 3:
        return True
    if _min_convex_interior_angle_deg(poly) < SLIVER_MIN_ANGLE_DEG:
        return True
    return area_m2 < sliver_max_area_m2 and _mrr_aspect(poly) > sliver_max_aspect


def _min_convex_interior_angle_deg(poly: Polygon) -> float:
    """Smallest convex interior angle across exterior vertices; reflex ignored."""
    ring = list(poly.exterior.coords[:-1])
    n = len(ring)
    if n < 3:
        return 180.0
    is_ccw = poly.exterior.is_ccw
    min_angle = 180.0
    for i in range(n):
        prev = ring[(i - 1) % n]
        curr = ring[i]
        nxt = ring[(i + 1) % n]
        e1x, e1y = curr[0] - prev[0], curr[1] - prev[1]
        e2x, e2y = nxt[0] - curr[0], nxt[1] - curr[1]
        l1 = math.hypot(e1x, e1y)
        l2 = math.hypot(e2x, e2y)
        if l1 == 0.0 or l2 == 0.0:
            continue
        cross = (e1x * e2y - e1y * e2x) / (l1 * l2)
        dot = (e1x * e2x + e1y * e2y) / (l1 * l2)
        turn = math.degrees(math.atan2(cross, dot))
        if not is_ccw:
            turn = -turn
        interior = 180.0 - turn
        if 0.0 < interior < 180.0 and interior < min_angle:
            min_angle = interior
    return min_angle


def _mrr_aspect(poly: Polygon) -> float:
    try:
        mrr = poly.minimum_rotated_rectangle
    except GEOSException:
        # MRR failure on a degenerate polygon: treat as square (non-sliver default).
        return 1.0
    if mrr is None or mrr.is_empty or mrr.geom_type != "Polygon":
        return 1.0
    coords = list(mrr.exterior.coords[:-1])
    if len(coords) < 4:
        return 1.0
    sides = [
        math.hypot(
            coords[(i + 1) % len(coords)][0] - coords[i][0],
            coords[(i + 1) % len(coords)][1] - coords[i][1],
        )
        for i in range(len(coords))
    ]
    lo = min(sides)
    if lo <= 0.0:
        return float("inf")
    return max(sides) / lo


def _pool_keep_mask(polys: list[Polygon], raster_path: str, pool_min_fraction: float) -> list[bool]:
    """True per polygon whose interior is not pool-coloured (blue-dominant RGB or cyan HSV)."""
    keep: list[bool] = []
    with rasterio.open(raster_path) as src:
        for poly in polys:
            try:
                arr, _ = rasterio_mask(src, [mapping(poly)], crop=True, filled=False, all_touched=False)
            except ValueError:
                # Polygon lies outside the raster; keep (no pool evidence available).
                keep.append(True)
                continue
            if arr.shape[0] < 3 or arr[0].compressed().size < 5:
                keep.append(True)
                continue
            r = arr[0].astype(np.int32).compressed()
            g = arr[1].astype(np.int32).compressed()
            b = arr[2].astype(np.int32).compressed()
            blue_dom = (b > r + POOL_BLUE_OVER_RED) & (b > g + POOL_BLUE_OVER_GREEN)
            h, s, v = _rgb_to_hsv_arrays(r, g, b)
            cyan_dom = (
                (h >= POOL_CYAN_H_LO) & (h <= POOL_CYAN_H_HI) & (s > POOL_CYAN_S_MIN) & (v > POOL_CYAN_V_MIN)
            )
            pool_dom = blue_dom | cyan_dom
            keep.append(bool(pool_dom.mean() < pool_min_fraction))
    return keep


def _rgb_to_hsv_arrays(
    r: np.ndarray, g: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB->HSV; inputs int 0-255, outputs H (0-360) and S/V (0-1)."""
    rf = r.astype(np.float32) / 255.0
    gf = g.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(rf, gf), bf)
    mn = np.minimum(np.minimum(rf, gf), bf)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 0
    rmask = mask & (mx == rf)
    gmask = mask & (mx == gf) & ~rmask
    bmask = mask & (mx == bf) & ~rmask & ~gmask
    h[rmask] = ((gf[rmask] - bf[rmask]) / diff[rmask]) % 6.0
    h[gmask] = (bf[gmask] - rf[gmask]) / diff[gmask] + 2.0
    h[bmask] = (rf[bmask] - gf[bmask]) / diff[bmask] + 4.0
    h = h * 60.0
    s = np.where(mx > 0, diff / np.where(mx > 0, mx, 1.0), 0.0)
    v = mx
    return h, s, v
