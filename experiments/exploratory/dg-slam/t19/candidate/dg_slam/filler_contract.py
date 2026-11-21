"""CPU-safe index and anchor-source contract. No GT, Torch or CUDA dependency."""
import math

def frame_index(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a frame index')
    x = float(value)
    if not math.isfinite(x) or x < 0 or not x.is_integer() or x > 2**24:
        raise ValueError('Expected a nonnegative integer frame index exactly representable in float32')
    return int(x)

def ordered_indices(values):
    ids = [frame_index(x) for x in values]
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError('Frame indices must be unique and strictly increasing')
    return ids

def batch_plan(timestamps, keyframe_timestamps):
    keys = ordered_indices(keyframe_timestamps)
    frames = ordered_indices(timestamps)
    if not keys or (frames and frames[0] < keys[0]):
        raise ValueError('At least one left anchor must cover every requested frame')
    lookup = {frame: ordinal for ordinal, frame in enumerate(keys)}
    slots, nonkey_positions = [], []
    for position, frame in enumerate(frames):
        if frame in lookup:
            slots.append(('anchor', lookup[frame]))
        else:
            slots.append(('temporary', len(nonkey_positions)))
            nonkey_positions.append(position)
    return slots, nonkey_positions

def anchor_policy(arm, policy, mixed_geometry_diagnostic=False):
    if arm not in ('NO_GS_REFINE', 'LOCAL_ID'):
        raise ValueError('Unsupported tracking arm')
    if policy == 'native_final_DROID':
        if arm != 'NO_GS_REFINE':
            raise ValueError('Native DROID anchors would discard ID refinement; cannot label this LOCAL_ID')
        return {'policy': policy, 'mixed_geometry': False, 'diagnostic_only': True,
                'ID_refinement_retained': False}
    if policy == 'mapping_snapshot' and mixed_geometry_diagnostic:
        return {'policy': policy, 'mixed_geometry': True, 'diagnostic_only': True,
                'ID_refinement_retained': arm == 'LOCAL_ID'}
    raise ValueError('Mapping-pose injection needs explicit mixed_geometry_diagnostic=True; coherent ID integration is unresolved')
