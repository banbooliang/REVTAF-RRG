import os, json
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.tester import Tester
from models.blip import blip_decoder
from dataset import create_dataset_test 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from transformers import BertTokenizer 
from modules.logger import create_logger
import torch.distributed as dist


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--local_rank', type=int, default=0, help='local rank in DDP.')
    parser.add_argument('--exp_name', type=str, default='mimic_cxr',
                        help='the name of the experiments.')
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings 
    parser.add_argument('--load_pretrained', type=str, default=None, help='pretrained path if any')

    # Sample related
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--gen_max_len', type=int, default=150, help='the maximum token length for text generation.')
    parser.add_argument('--gen_min_len', type=int, default=100, help='the minimum token length for text generation.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--init_lr', type=float, default=5e-5, help='.')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='the weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed     training')
    parser.add_argument('--device', default='cuda')

    # cls head
    parser.add_argument('--cls_weight', type=float, default=4, help='Loss weight of classification branch.')
    parser.add_argument('--clip_k', type=int, default=21, help='Number of retrieved reports from database.')
    #3 ot
    parser.add_argument('--d_model', type=int, default=1024, help='d_model of OT attention.')
    parser.add_argument('--nhead', type=int, default=8, help='n_head of OT attention.')
    parser.add_argument('--two_stage_class_embed_share', default=False, help='num_layers of OT attention.')
    parser.add_argument('--align_weight', type=float, default=1, help='align_weight of global and local alignment.')
    # hyperbolic
    parser.add_argument('--c', type=float, default=0.01, help='hyperblic space')
    parser.add_argument(
        "--manifold", type=str, default="PoincareBall", choices=["Euclidean", "Hyperboloid", "PoincareBall"]
    )
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=int, default=1)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--num-layers_post", type=int, default=3)
    parser.add_argument("--dim", type=int, default=75,help='region channels')
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument('--rank_weight', type=float, default=0.1, help='rank_weight of learning to rank')
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    utils.init_distributed_mode(args) # from blip
    device = torch.device(args.device)

    # fix random seeds
    seed = args.seed + utils.get_rank() # from blip
    torch.manual_seed(args.seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    
    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    #### Dataset #### 
    print("Creating dataset...")
    test_dataset = create_dataset_test('generation_%s'%args.dataset_name, tokenizer, args)
    print('number of testing samples: %d'%len(test_dataset))
    
    samplers = [None]

    test_dataloader = create_loader([test_dataset], samplers, batch_size=[args.batch_size], num_workers=[4], is_trains=[False], collate_fns=[None])[0] 

    # build model architecture
    labels_temp = ['[BLA]'] * 18 # for calculate length only
    prompt_temp = ' '.join(labels_temp)+' '
    model = blip_decoder(args, device, tokenizer, image_size=args.image_size, prompt=prompt_temp)
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        model.load_state_dict(state_dict)
        print("load checkpoint from {}".format(args.load_pretrained))

    # get function handles of loss and metrics
    criterion_cls = nn.CrossEntropyLoss()
    metrics = compute_scores

    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl',rank=0, world_size = 1)

    save_dir = os.path.join(args.save_dir, args.dataset_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir, dist_rank=args.local_rank, name=args.exp_name)

    # build trainer and start to train
    tester = Tester(model, criterion_cls, metrics, args, logger, device, test_dataloader)

    log = tester.test_blip()
    for key, value in log.items():
        print('\t{:15s}: {}'.format(str(key), value))
        logger.info('\t{:15s}: {}'.format(str(key), value))
    
if __name__ == '__main__':
    main()
