"""Lossless class-support counts on the frozen independent sparse evaluation domain."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def count_view(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    assert prediction.shape == truth.shape
    assert np.isin(truth, list(range(19))+[255]).all(), 'Invalid GT class'
    assert np.isin(prediction, list(range(19))+[-1, -2]).all(), 'Invalid predicted class'
    known = truth != 255
    columns = prediction[known].astype(np.int64)
    columns[columns == -1] = 19  # Supported raster, no accumulated class evidence.
    columns[columns == -2] = 20  # No rendered support.
    matrix = np.bincount(truth[known].astype(np.int64)*21+columns, minlength=19*21).reshape(19, 21)
    assert int(matrix.sum()) == int(known.sum())
    return matrix


def class_statistics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    assert matrix.shape == (19, 21) and (matrix >= 0).all()
    support = matrix.sum(1)
    predicted = matrix[:, :19].sum(0)
    intersection = matrix[:, :19].diagonal()
    union = support+predicted-intersection
    rows = [dict(class_id=c, GT_support=int(support[c]), predicted_on_GT_support=int(predicted[c]),
        intersection=int(intersection[c]), union=int(union[c]),
        IoU=float(intersection[c]/union[c]) if union[c] else None,
        unknown_evidence_predictions=int(matrix[c, 19]), uncovered_predictions=int(matrix[c, 20])) for c in range(19)]
    ious = [r['IoU'] for r in rows if r['IoU'] is not None]
    return dict(classes=rows, sparse_mIoU=float(np.mean(ious)) if ious else None,
        semantic_GT_pixels=int(support.sum()), confusion=matrix.tolist())


def write_counts(path, views):
    """Pooled counts are diagnostic; preregistered metrics stay the mean of view IoUs."""
    assert len(views) == 60
    rows = []
    for mode in ['COMMON_GT', 'COMMON_ORB', 'OWN_SELECTED']:
        for cohort in ['EARLY', 'RECENT', 'ALL']:
            selected = [r for r in views if r['mode']==mode and (cohort=='ALL' or r['cohort']==cohort)]
            assert len(selected) == (20 if cohort=='ALL' else 10)
            pooled = np.sum([r['confusion'] for r in selected], axis=0, dtype=np.int64)
            rows.append(dict(mode=mode, cohort=cohort, views=len(selected),
                mean_per_view_sparse_mIoU=float(np.mean([r['sparse_mIoU'] for r in selected])),
                pooled_diagnostic=class_statistics(pooled)))
    result = dict(status='COMPLETE60_CLASS_SUPPORT_AND_CONFUSION_COUNTS',
        label_space='Project Cityscapes19; not SemanticKITTI leaderboard classes',
        row_classes=list(range(19)), column_classes=list(range(19))+['unknown_evidence','uncovered'],
        GT_255_excluded=True, missing_predictions_count_as_false_negatives=True,
        primary_metric='Mean per-view sparse_mIoU, unchanged; pooled class statistics are supplemental',
        rows=views, summary=rows, online_changes=0)
    Path(path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def collect_registered(remote, out, cells, root_index):
    remote, out = Path(remote), Path(out)
    groups, flat = [], []
    for cell in cells:
        entry = root_index[f"{cell['stage']}:K{cell['sequence']}:S{cell['seed']}"]
        root = remote/entry['active_root']
        evaluation = root/'runs'/cell['group']/'evaluation/results.json'
        original = json.loads(evaluation.read_text())
        source = evaluation.with_name('semantic_confusion.json')
        if not source.exists():
            source = remote/'report_assets'/root.name/cell['group']/'semantic_confusion.json'
        evidence = json.loads(source.read_text())
        assert evidence['status'] == 'COMPLETE60_CLASS_SUPPORT_AND_CONFUSION_COUNTS'
        if 'semantic_confusion' in original:
            assert hashlib.sha256(source.read_bytes()).hexdigest() == original['semantic_confusion']['sha256']
        else:
            assert evidence['backfill_original_evaluation_sha256'] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
        indexed = {(r['mode'],r['frame']):r for r in original['rows']}
        assert len(evidence['rows']) == len(indexed) == 60
        assert {(r['mode'],r['frame']) for r in evidence['rows']} == set(indexed)
        for row in evidence['rows']:
            reference = indexed[(row['mode'],row['frame'])]
            assert row['source_frame'] == reference['source_frame']
            assert row['semantic_GT_pixels'] == reference['semantic_GT_pixels']
            assert abs(row['sparse_mIoU']-reference['sparse_mIoU']) < 1e-12
        key = {name:cell[name] for name in ['stage','sequence','seed','group']}
        groups.append(dict(**key, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), evidence=evidence))
        for summary in evidence['summary']:
            for item in summary['pooled_diagnostic']['classes']:
                flat.append(dict(**key, mode=summary['mode'], cohort=summary['cohort'], views=summary['views'],
                    **{k:v for k,v in item.items() if k!='IoU'}, pooled_class_IoU=item['IoU']))
    (out/'各类语义支持与混淆.json').write_text(json.dumps(dict(status='COMPLETE_REGISTERED_CLASS_SUPPORT_COUNTS',
        groups=groups, primary_mIoU_unchanged=True, pooled_class_IoU_is_supplemental=True), ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'各类语义支持.tsv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(flat[0]), delimiter='\t')
        writer.writeheader();writer.writerows(flat)
    return dict(groups=len(groups),class_rows=len(flat),primary_mIoU_unchanged=True)
