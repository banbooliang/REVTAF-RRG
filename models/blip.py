import os
import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from models.resnet import blip_resnet

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer
from einops import rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from hyptorch.pmath import dist_matrix
import numpy as np
import json
import manifolds
import models.encoders as encoders

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 args,
                 device,
                 tokenizer=None,
                 image_size = 224,
                 prompt = '',
                 ):
        super().__init__()
        self.args = args
        self.annotation = json.load(open('./data/mimic_cxr/mimic_annotation_promptmrg.json','r'))['train']
        vision_width = 2048
        self.visual_encoder = blip_resnet(args)
        
        self.cls_head = nn.Linear(vision_width+512, 18*4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        self.vision_proj = nn.Linear(vision_width, 512)
        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('configs/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.memory = Transformer(d_model=512,
                                  num_encoder_layers=2,
                                  num_decoder_layers=2,
                                  num_queries=1)
        ### OT-based attention
        self.multihead_attn = AttentionOT( 
            args.d_model, num_heads=args.nhead, qkv_bias=True, attn_drop=0.1)
        self.mpsa_attention = TPN_DecoderLayer(d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.d_model * 4)
        self.ot_vision_proj = nn.Linear(vision_width, 1024)
        self.ot_txt_proj1 = nn.Linear(768, 1024)
        self.ot_txt_proj2 = nn.Linear(512, 1024)
        self.image_size =  args.image_size
        ## hyperbolic
        self.manifold = getattr(manifolds, "PoincareBall")() 
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))       
        self.gwd = getattr(encoders, "Post_HNN")(self.c, args)
        self.device = device
        
        with open('./data/mimic_cxr/image_region_score_train.json', 'r') as f:
            r_i_score = [torch.tensor(json.loads(i.strip()),dtype=torch.float32) for i in f]
        self.r_i_score = torch.stack(r_i_score, dim=0).to(self.device) # 270790,75
        
        
    def forward(self, image, caption, cls_labels, clip_memory, global_txt, region_txt, local_image, criterion_cls, base_probs):
        image_embeds, avg_embeds = self.visual_encoder(image)   # b,49,2048    b,2048
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        b, hw, c = image_embeds.shape
        ##########################
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None) # [21,16,512]+[1,16,512]->[1,16,1,512]
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1) 
        ##########################

        cls_preds = self.cls_head(avg_embeds) # [16,2560]->[16,72]
        cls_preds = cls_preds.view(-1, 4, 18)
        # logit adjustment
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(image.device)
        
        loss_cls = criterion_cls(cls_preds, cls_labels)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id  # input_ids=[b,127]
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) # 把0填充为-100
        decoder_targets[:,:self.prompt_length] = -100
        
        ### learning to rank
        loss_rank = self.learning_to_rank(local_image, cls_labels)
        ### consistent constraint
        region_txt = self.ot_txt_proj1(region_txt)
        enhance_region_embed, region_map = self.ot_cross_attention(image_embeds,region_txt) # [b,49,768], [b.75,224,224]
        global_txt = self.ot_txt_proj2(global_txt)
        enhance_global_embed, global_map = self.ot_cross_attention(image_embeds, global_txt) # [b,max_seq_num,h,w]
        sim = F.cosine_similarity(region_txt.unsqueeze(2), global_txt.unsqueeze(1), dim=-1) #b,75,max_seq_num 
        iou_loss = self.overlap_loss(region_map, global_map, sim)
            
        image_embeds = torch.cat([enhance_global_embed, enhance_region_embed], dim=-1)  # b,49,768*2                                                
        ### LM loss
        ###
        decoder_output = self.text_decoder(
                                           text['input_ids'],  # [16,121]
                                           attention_mask = text['attention_mask'],  # [16,121]
                                           encoder_hidden_states = image_embeds, # [16,49,1536]
                                           labels = decoder_targets,  # [16,121]
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss                
        return loss_lm, loss_cls, iou_loss, loss_rank
    
    def learning_to_rank(self, cls_preds, cls_labels):
        cls_labels_onehot = F.one_hot(cls_labels, num_classes=4).flatten(1)  # b,72
        xor_result = torch.bitwise_xor(cls_labels_onehot.unsqueeze(1), 
                                       cls_labels_onehot.unsqueeze(0))
        hamming_dist = (xor_result != 0).sum(-1) # b,b
        cls_preds_hyper = self.gwd.encode(cls_preds)  # b,b
  
        pred_dist_hyper = dist_matrix(cls_preds_hyper, cls_preds_hyper, c=self.c)  # b,b
        b = cls_preds_hyper.shape[0]
        mask = ~torch.eye(b, dtype=bool)
        pred_dist_hyper = pred_dist_hyper[mask].reshape(b,b-1) # [b, b-1]
        hamming_dist = hamming_dist[mask].reshape(b,b-1) # [b, b-1]
       
        true_labels = torch.argmin(hamming_dist.detach(), dim=1)  # [b]
        loss = F.cross_entropy(-pred_dist_hyper, true_labels)       
        return loss
    
    def ot_cross_attention(self, img, txt):
        img_re = self.ot_vision_proj(img) # [b,49,768]\
        # txt_re = self.ot_txt_proj(txt)
        q_, attn_ = self.mpsa_attention(img_re.transpose(0,1).unsqueeze(0), txt)   # mbc  bmk  
        attn = attn_.squeeze(-1) 
        attns = self.d3_to_d4(attn)  # B, K, H, W
        
        qs = q_.squeeze(0).permute(1, 0, 2).contiguous()  # B, 49, C,    
        # Upsample
        pred = F.interpolate(attns, size=(self.image_size, self.image_size),mode='bilinear', align_corners=False)
        return qs, pred
    
    def overlap_loss(self, l, g, s, threshold=0.5):
        
        l = (l - l.min()) / (l.max() - l.min() + 1e-6)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)

        l_bin = (l >= threshold).float().unsqueeze(2) # [b, k, 1, h, w]
        g_bin = (g >= threshold).float().unsqueeze(1)   # [b, 1, n, h, w]
    
        intersection = (l_bin * g_bin).sum(dim=(-1, -2))  # [b, k, n]
        union = (l_bin + g_bin - l_bin * g_bin).sum(dim=(-1, -2))  # [b, k, n]

        iou = intersection / (union + 1e-8)  # [b, k, n]
        similarity = torch.sigmoid(s)  # 将 similarity 转换到 [0, 1]
  
        iou_loss = (1 - similarity * iou).mean()  
         
        return iou_loss
    
    
    def generate(self, image, clip_memory, region_txt, region_image, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds, avg_embeds = self.visual_encoder(image) 
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)

        # classification branch
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18)
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds[:, 1, :14]
        cls_preds = torch.argmax(cls_preds, dim=1).cpu().numpy().tolist()

        prompts = []
        for j in range(len(cls_preds)):
            prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prompts.append(prompt)
        
        text = self.tokenizer(prompts, return_tensors="pt")
        input_ids = text.input_ids.to(image.device)
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        attn_masks = attn_masks[:, :-1] 
        # retrive global txt embeddings
        h_embedding_region = self.gwd.encode(region_image)
        h_embedding_ref = self.gwd.encode(self.r_i_score)
        hyper_dist = dist_matrix(h_embedding_region, h_embedding_ref, c=self.c)
        index_select = hyper_dist.argmin(dim=-1) # b
        global_txt_embeddings = []
        for b in range(image.shape[0]):
            t_path = os.path.join('./data/mimic_cxr', 'medclip_txt_embeddings', self.annotation[index_select[b]]['image_path'][0]).replace('.jpg', '.npy')
            global_txt_embedding_tmp = torch.from_numpy(np.load(t_path)).to(dtype=torch.float32)
            global_txt_embeddings.append(global_txt_embedding_tmp)
        global_txt = torch.stack(global_txt_embeddings,dim=0).to(self.device)
        ### ot-based alignment
        region_txt = self.ot_txt_proj1(region_txt)
        enhance_region_embed, region_map = self.ot_cross_attention(image_embeds, region_txt) # [b,49,768], [b.75,224,224]
        global_txt = self.ot_txt_proj2(global_txt)
        enhance_global_embed, global_map = self.ot_cross_attention(image_embeds, global_txt)  # [b,max_seq_num,h,w]
        image_embeds = torch.cat([enhance_global_embed, enhance_region_embed], dim=-1)  

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        #beam search
        outputs = self.text_decoder.generate(
                                            input_ids=input_ids,
                                             min_length=min_length, # 4.25 Transformers
                                             max_new_tokens=max_length,
                                             num_beams=num_beams,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id, 
                                             repetition_penalty=repetition_penalty,
                                             attention_mask = attn_masks,
                                             **model_kwargs)            
        captions = []    
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(prompts[i]):])
        return captions, cls_preds, cls_preds_logits
    
    def d3_to_d4(self, t):
        b, k, hw = t.size()
        # if hw % 2 != 0:
        #     t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.reshape(b, k, h, w)


def Sinkhorn_log_exp_sum(C, mu, nu, epsilon):
    
    def _log_boltzmann_kernel(u, v, epsilon, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= epsilon
        return kernel
  
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-6
    max_iter = 100
            
    for i in range(max_iter):
       
        u0 = u  # useful to check the update
        K = _log_boltzmann_kernel(u, v, epsilon, C)
        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
        u = epsilon * u_ + u
        
        K_t = _log_boltzmann_kernel(u, v, epsilon, C).permute(0, 2, 1).contiguous()
        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
        v = epsilon * v_ + v
        
        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break
    
    K = _log_boltzmann_kernel(u, v, epsilon, C)  # B,K,HW
    T = torch.exp(K)

    return T


class AttentionOT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.eps = 0.05 

    def forward(self, xq, xk, xv): #IMG,txt,txt  #  HW, B, C  B, K, C
        Nq, M, B, C = xq.size() 
        
        xq = self.q(xq)
        xk = self.k(xk)
        v = self.v(xv)
        
        # assign variables
        _, K, _ = xk.shape
        xq = F.normalize(xq, dim=-1, p=2)
        xk = F.normalize(xk, dim=-1, p=2)
   
        # compute score map 
        sim = torch.einsum('bkc,nmbc->bnkm', xk, xq)
        sim = sim.permute(0,2,3,1)
        sim = sim.contiguous().view(B*K, M, Nq) 
        wdist = 1.0 - sim   # cost C

        # optimally transport score map
        xx = torch.zeros(B*K, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(B*K, Nq, dtype=sim.dtype, device=sim.device).fill_(1. / Nq)
        T = Sinkhorn_log_exp_sum(wdist, xx,yy, self.eps) # B*K, M, Nq
        
        # T * score map
        score_map = (M * Nq * sim * T).view(B, K, M, Nq) 
        # attn_save = score_map.clone().contiguous().sum(dim=-1).squeeze(-1)
        attn = rearrange(T.view(B, K, M, Nq), 'b k m n -> n b k m', b = B, k = K, n=Nq) 
        attn = self.attn_drop(attn)

        x = torch.einsum('nbkm,bkc->nmbc', attn, v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, score_map
    

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn

        # MPSA (Multi-prompts Sinkhorn Attention)
        self.multihead_attn = AttentionOT( 
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt, memory):
        tgt2, attn2 = self.multihead_attn(tgt, memory, memory)  # K,B, C    B, K, HW
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2
    

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None: 
            output = self.norm(output)

        return outputs, attns
    
    
def blip_decoder(args, device, tokenizer, **kwargs):
    model = BLIP_Decoder(args, device, tokenizer, **kwargs)
    return model    


    
