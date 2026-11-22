import unittest

import numpy as np

from semantic_confusion import count_view, class_statistics


class ClassSupportControls(unittest.TestCase):
    def test_missing_support_keeps_GT_denominator_and_ignores_255(self):
        truth=np.array([[0,0,1,1,255]])
        prediction=np.array([[0,-2,-1,0,1]])
        result=class_statistics(count_view(prediction,truth))
        self.assertEqual(result['semantic_GT_pixels'],4)
        self.assertEqual(result['classes'][0]['GT_support'],2)
        self.assertEqual(result['classes'][0]['uncovered_predictions'],1)
        self.assertEqual(result['classes'][1]['unknown_evidence_predictions'],1)
        self.assertAlmostEqual(result['classes'][0]['IoU'],1/3)
        self.assertEqual(result['classes'][1]['IoU'],0)
        self.assertAlmostEqual(result['sparse_mIoU'],1/6)

    def test_agrees_with_original_view_metric_and_label_permutation(self):
        truth=np.array([[0,1,2,3,255],[4,4,18,18,2]])
        prediction=np.array([[1,1,-1,-2,18],[4,18,18,4,2]])
        original=[]
        for label in range(19):
            union=((truth==label)|(prediction==label))&(truth!=255)
            if union.any():original.append(float((((truth==label)&(prediction==label))&(truth!=255)).sum()/union.sum()))
        result=class_statistics(count_view(prediction,truth))
        self.assertEqual(result['sparse_mIoU'],float(np.mean(original)))
        permuted_truth=np.where(truth==255,255,18-truth)
        permuted_prediction=np.where(prediction>=0,18-prediction,prediction)
        self.assertEqual(result['sparse_mIoU'],class_statistics(count_view(permuted_prediction,permuted_truth))['sparse_mIoU'])

    def test_all_unknown_is_zero_IoU_and_invalid_labels_fail(self):
        truth=np.array([[0,1,18]])
        self.assertEqual(class_statistics(count_view(np.full_like(truth,-2),truth))['sparse_mIoU'],0)
        with self.assertRaises(AssertionError):count_view(np.array([[19,0,0]]),truth)
        with self.assertRaises(AssertionError):count_view(np.array([[0,0,0]]),np.array([[0,19,255]]))


if __name__=='__main__':unittest.main()
