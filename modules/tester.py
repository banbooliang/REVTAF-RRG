import logging
import os
from abc import abstractmethod
import numpy as np
import time

import cv2
import torch

from .metrics_clinical import CheXbertMetrics

class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, logger):
        self.logger = logger
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        # self.logger = logging.getLogger(__name__)

        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, logger, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device, logger)
        self.test_dataloader = test_dataloader

    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        
        with torch.no_grad():
            # best_f1 = 0.0
            test_gts, test_res = [], [] 
            for batch_idx, (images, captions, cls_labels, clip_memory, region_txt, region_image) in enumerate(self.test_dataloader):
                # single_res, single_gts = [], []
                images = images.to(self.device) 
                cls_labels = cls_labels.numpy().tolist()
                clip_memory = clip_memory.to(self.device)
                region_txt = region_txt.to(self.device) 
                region_image = region_image.to(self.device) 
                ground_truths = captions
                reports, _, _ = self.model.generate(images, clip_memory, region_txt, region_image, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                
                # single_res.extend(reports)
                # single_gts.extend(ground_truths)
                if batch_idx % 10 == 0:
                    self.logger.info('{}/{}'.format(batch_idx, len(self.test_dataloader)))
                    print('{}/{}'.format(batch_idx, len(self.test_dataloader)))
                # single_met = self.metric_ftns({i: [gt] for i, gt in enumerate(single_gts)},
                #                         {i: [re] for i, re in enumerate(single_res)})
                # single_ce = self.chexbert_metrics.compute(single_gts, single_res)
                # msg_met = '{}/{} '.format(batch_idx, len(self.test_dataloader)) + ' '.join([f'single_{k}: {v}' for k, v in single_met.items()])
                # self.logger.info(msg_met)
                # msg_ce = '{}/{} '.format(batch_idx, len(self.test_dataloader)) + ' '.join([f'single_{k}: {v}' for k, v in single_ce.items()])
                # self.logger.info(msg_ce)
                
                # if single_ce['ce_f1'] > best_f1:
                #     best_f1 = single_ce['ce_f1']
                    # print('Best F1: {}, batch_idx: {}'.format(best_f1, batch_idx))
                
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            # for i in range(len(test_ce)):
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
            
            # log.update(**{'test_' + k: v for k, v in test_ce.items()})
            # with open(os.path.join(self.save_dir, 'test_res.txt'), 'w') as f:
            #     for i in range(len(test_res)):
            #         f.write(test_res[i] + '\n')
            # with open(os.path.join(self.save_dir, 'test_gts.txt'), 'w') as f:
            #     for i in range(len(test_gts)):
                    # f.write(test_gts[i] + '\n')
        return log

