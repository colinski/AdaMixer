import numpy as np
import torch
from mmdet.apis import init_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from tqdm import tqdm, trange#progress bar
import pickle
import sys

#init model and dataloader
config = sys.argv[1]
checkpoint = sys.argv[2]
pkl_fname = sys.argv[3]
model = init_detector(config, checkpoint).cuda().eval()
dataset = build_dataset(model.cfg.data.val)
# dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

output = []

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for idx in trange(len(dataset)):
    sample = dataset[idx]
    gt = dataset.get_ann_info(idx)
    img = sample['img'][0].cuda().unsqueeze(0)
    img_metas = [sample['img_metas'][0].data]
    img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
    result = {k: v for k, v in img_metas[0].items()}

    with torch.no_grad():

        start.record()

        x = model.extract_feat(img)

        end.record()
        torch.cuda.synchronize()
        result['backbone_time'] = start.elapsed_time(end) ##########

        start.record()
        
        query_xyzr, query_content, imgs_whwh = \
            model.rpn_head.simple_test_rpn(x, img_metas)


        num_imgs = len(img_metas)
        num_queries = query_xyzr.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_queries, 1)
        all_stage_bbox_results = []
        all_stage_loss = {}
        
        all_cls_score, all_bbox_pred = [], []
        for stage in range(model.roi_head.num_stages):
            bbox_results = model.roi_head._bbox_forward(stage, x, query_xyzr, query_content,
                                              img_metas)
            
            #for future layers
            query_xyzr = bbox_results['query_xyzr']#.detach()
            query_content = bbox_results['query_content']

            # all_stage_bbox_results.append(bbox_results)
            # if gt_bboxes_ignore is None:
                # gt_bboxes_ignore = [None for _ in range(num_imgs)]
            # cls_pred_list = bbox_results['detach_cls_score_list']
            # bboxes_list = bbox_results['detach_bboxes_list']

            # query_xyzr = bbox_results['query_xyzr'].detach()
            # query_content = bbox_results['query_content']

            cls_score, bbox_pred = [], []          
            for i in range(num_imgs):
                cls_score.append(bbox_results['cls_score'][i])
                norm_bbox = bbox_xyxy_to_cxcywh(bbox_results['decode_bbox_pred'][i] / imgs_whwh[i])
                bbox_pred.append(norm_bbox)
            all_cls_score.append(torch.stack(cls_score))
            all_bbox_pred.append(torch.stack(bbox_pred))
        all_cls_score = torch.stack(all_cls_score).squeeze()
        all_cls_score = torch.softmax(all_cls_score, dim=-1)
        all_bbox_pred = torch.stack(all_bbox_pred).squeeze()

    end.record()
    torch.cuda.synchronize()
    result['decoder_time'] = start.elapsed_time(end) ##########

        
    #collect output
    # result['query_embeds'] = query_embeds.cpu().numpy()
    result['bbox_preds'] = all_bbox_pred.cpu().numpy()
    result['cls_probs'] = all_cls_score.cpu().numpy()

    result['gt_bboxes'] = gt['bboxes']
    result['gt_labels'] = gt['labels']
    output.append(result)
    
#save to pickle file
with open(pkl_fname, 'wb') as f:
    pickle.dump(output, f)
