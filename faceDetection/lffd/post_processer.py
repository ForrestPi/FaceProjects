'''
@Descripttion: This is Forrest Zhu's demo,which is only for reference
@version: 
@Author: Forrest Zhu
@Date: 2019-09-13 21:06:46
@LastEditors: Forrest Zhu
@LastEditTime: 2019-10-01 09:42:48
'''
import torch

from nms import boxes_nms


class PostProcessor:
    def __init__(self,
                 iou_threshold = 0.05,
                 score_threshold = 0.9,
                 image_size = 512,
                 max_per_class=200,
                 max_per_image=-1):
        self.confidence_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.width = image_size
        self.height = image_size
        self.max_per_class = max_per_class
        self.max_per_image = max_per_image

    def __call__(self, confidences, locations, width=None, height=None, batch_ids=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        batch_size = confidences.size(0)
        if batch_ids is None:
            batch_ids = torch.arange(batch_size, device=confidences.device)
        else:
            batch_ids = torch.tensor(batch_ids, device=confidences.device)

        locations = locations[batch_ids]
        confidences = confidences[batch_ids]

        results = []
        for decoded_boxes, scores in zip(locations, confidences):
            # per batch
            filtered_boxes = []
            filtered_labels = []
            filtered_probs = []
            for class_index in range(1, scores.size(1)):
                probs = scores[:, class_index]
                mask = probs > self.confidence_threshold
                probs = probs[mask]
                if probs.size(0) == 0:
                    continue
                boxes = decoded_boxes[mask, :] # x1,y1,x2,y2
                
                boxes[:, 0] *= width
                boxes[:, 2] *= width
                boxes[:, 1] *= height
                boxes[:, 3] *= height
                
                keep = boxes_nms(boxes, probs, self.iou_threshold, self.max_per_class)

                boxes = boxes[keep, :]
                labels = torch.tensor([class_index] * keep.size(0))
                probs = probs[keep]

                filtered_boxes.append(boxes)
                filtered_labels.append(labels)
                filtered_probs.append(probs)

            # no object detected
            if len(filtered_boxes) == 0:
                filtered_boxes = torch.empty(0, 4)
                filtered_labels = torch.empty(0)
                filtered_probs = torch.empty(0)
            else:  # cat all result
                filtered_boxes = torch.cat(filtered_boxes, 0)
                filtered_labels = torch.cat(filtered_labels, 0)
                filtered_probs = torch.cat(filtered_probs, 0)
            if 0 < self.max_per_image < filtered_probs.size(0):
                keep = torch.argsort(filtered_probs, dim=0, descending=True)[:self.max_per_image]
                filtered_boxes = filtered_boxes[keep, :]
                filtered_labels = filtered_labels[keep]
                filtered_probs = filtered_probs[keep]
            results.append((filtered_boxes, filtered_labels, filtered_probs))
        return results
