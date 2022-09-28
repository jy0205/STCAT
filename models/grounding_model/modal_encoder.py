import copy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Tuple

from utils.misc import NestedTensor
from .position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine


class CrossModalEncoder(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        # attention configuration
        d_model = cfg.MODEL.STCAT.HIDDEN
        nhead = cfg.MODEL.STCAT.HEADS
        dim_feedforward = cfg.MODEL.STCAT.FFN_DIM
        dropout = cfg.MODEL.STCAT.DROPOUT
        activation = "relu"
        num_layers = cfg.MODEL.STCAT.ENC_LAYERS
        self.d_model = d_model
        
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = None
        self.encoder = SpatialTemporalEncoder(cfg, encoder_layer, num_layers, encoder_norm)
        self.fusion = nn.Linear(d_model, d_model)
        
        # The position embedding for feature map
        # self.spatial_embed = PositionEmbeddingLearned(d_model // 2)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, videos : NestedTensor = None, vis_pos = None, texts : Tuple = None):
        pos_embed = vis_pos   # The sin pos embeding from backbone encoder
        vis_features, vis_mask, vis_durations = videos.decompose() 
          
        assert pos_embed.shape[0] == sum(vis_durations)
    
        vis_mask[:, 0, 0] = False  # avoid empty masks
        device = vis_features.device
    
        b = len(vis_durations)
        _, _, H, W = vis_features.shape
        # n_frames x c x h x w => hw x n_frames x c
        vis_features = vis_features.flatten(2).permute(2, 0, 1)   
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        vis_mask = vis_mask.flatten(1)
        num_vis_tokens = vis_features.shape[0]

        # prepare the text encodings
        text_attention_mask, text_memory_resized, _ = texts
        assert b == len(vis_durations)
        
        # expand the attention mask from [b, len] to [n_frames, len]
        text_mask_list = []
        for i_b in range(b):
            frame_length = vis_durations[i_b]
            text_mask_list.append(
                torch.stack([text_attention_mask[i_b] for _ in range(frame_length)])
            )
        text_attention_mask = torch.cat(text_mask_list)
        
        # expand the text token from [len, b, d_model] to [len, n_frames, d_model]
        text_fea_list = []
        for i_b in range(b):
            frame_length = vis_durations[i_b]
            text_fea_list.append(
                torch.stack([text_memory_resized[:, i_b] for _ in range(frame_length)],dim=1)
            )
        text_memory_resized = torch.cat(text_fea_list, dim=1)   # [text_len, n_frames, d_model]
        
        # concat visual and text features and Pad the pos_embed with 0 for the text tokens
        features = torch.cat([vis_features,text_memory_resized],dim=0)
        mask = torch.cat([vis_mask, text_attention_mask], dim=1)
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
        
        # perfrom cross-modality interaction
        img_memory, frames_cls, videos_cls = self.encoder(
            features, 
            src_key_padding_mask=mask,
            pos=pos_embed, 
            durations=vis_durations
        )
        
        memory_cache = {
            "encoded_memory": img_memory,
            "mask": mask,  # batch first
            "frames_cls" : frames_cls,  # n_frame, d_model
            "videos_cls" : videos_cls, # b , d_model
            "durations": vis_durations,
            "fea_map_size" : (H, W)
        }
        
        return memory_cache
        

class SpatialTemporalEncoder(nn.Module):
    def __init__(self, cfg, encoder_layer, num_layers, norm=None, return_weights=False):
        super().__init__()
        self.spatial_layers = _get_clones(encoder_layer, num_layers)
        self.temporal_layers = _get_clones(encoder_layer, num_layers)
        video_max_len = cfg.INPUT.MAX_VIDEO_LEN
        d_model = cfg.MODEL.STCAT.HIDDEN 
        self.d_model = d_model
        
        # The position embedding of global tokens
        if cfg.MODEL.STCAT.USE_LEARN_TIME_EMBED:
            self.time_embed = SeqEmbeddingLearned(video_max_len + 1 , d_model)
        else:
            self.time_embed = SeqEmbeddingSine(video_max_len + 1, d_model) 
    
        # The position embedding of local frame tokens
        self.local_pos_embed = nn.Embedding(1, d_model) # the learned pos embed for frame cls token
        
        # The learnd local and global embedding
        self.frame_cls = nn.Embedding(1, d_model)  # the frame level local cls token
        self.video_cls = nn.Embedding(1, d_model)  # the video level global cls token
        
        self.num_layers = num_layers
        self.norm = norm
        self.return_weights = return_weights

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        durations=None
    ):
        output = src
        b = len(durations)
        t = max(durations)
        n_frames = sum(durations)
        device = output.device
        
        # The position embedding, token mask, src feature for local frame token, in spatial layer
        frame_src = self.frame_cls.weight.unsqueeze(1).repeat(1, n_frames, 1) # 1 x n_frames X d_model
        frame_pos = self.local_pos_embed.weight.unsqueeze(1).repeat(1, n_frames, 1) # 1 x n_frames X d_model
        frame_mask = torch.zeros((n_frames,1)).bool().to(device)
        
        output = torch.cat([frame_src, output], dim=0)
        src_key_padding_mask = torch.cat([frame_mask, src_key_padding_mask],dim=1)
        pos = torch.cat([frame_pos, pos],dim=0)
        
        # The position embedding, token mask, in temporal layer
        video_src = self.video_cls.weight.unsqueeze(0).repeat(b, 1, 1)  # b x 1 x d_model
        temp_pos = self.time_embed(t + 1).repeat(1, b, 1)  # (T + 1) x b x d_model
        temp_mask = torch.ones(b, t + 1).bool().to(device)
        temp_mask[:, 0] = False       # the mask for the video cls token
        for i_dur, dur in enumerate(durations):
            temp_mask[i_dur, 1 : 1 + dur] = False
        
        for i_layer, layer in enumerate(self.spatial_layers):
            # spatial interaction on each single frame
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            
            frames_src = torch.zeros(b, t+1, self.d_model).to(device)    # b x seq_len x C
            frames_src_list = torch.split(output[0,:,:], durations)  # [(n_frames, C)]
            
            for i_dur, dur in enumerate(durations):
                frames_src[i_dur, 0 : 1, :] = video_src[i_dur]  # pad the video cls token
                frames_src[i_dur, 1 : 1 + dur, :] = frames_src_list[i_dur]
                    
            frames_src = frames_src.permute(1, 0, 2)  # permute BxLenxC to LenxBxC
            
            # temporal interaction between all video frames
            frames_src = self.temporal_layers[i_layer](
                frames_src,
                src_mask=None,
                src_key_padding_mask=temp_mask,
                pos=temp_pos
            )
            
            frames_src = frames_src.permute(1, 0, 2) # permute LenxBxC to BxLenxC
            # dispatch the temporal context to each single frame token
            frames_src_list = []
            for i_dur, dur in enumerate(durations):
                video_src[i_dur] = frames_src[i_dur, 0 : 1]
                frames_src_list.append(frames_src[i_dur, 1 : 1 + dur])  # LenxC
            
            frames_src = torch.cat(frames_src_list, dim=0)
            output[0,:,:] = frames_src

        if self.norm is not None:
            output = self.norm(output)

        frame_src = output[0,:,:]
        output = output[1:,:,:]
        video_src = video_src.squeeze(1)  # b x 1 x d_model => b x d_model
        
        return output, frame_src, video_src


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
