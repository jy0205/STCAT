import torch
from torch import nn
import torch.nn.functional as F

from .net_utils import MLP, inverse_sigmoid
from .vision_model import build_vis_encoder
from .language_model import build_text_encoder
from .grounding_model import build_encoder, build_decoder
from utils.misc import NestedTensor


class STCATNet(nn.Module):
    """
    The general pipeline of STCAT for video spatio-temporal grounding, It consists of 
    the following several parts:
    - visual encoder
    - language encoder:
    - spatio-temporal multimodal Interactor
    - Temporal Localizer
    - Spatial Localizer 
    """

    def __init__(self, cfg):
        super(STCATNet, self).__init__()
        self.cfg = cfg.clone()
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN
        self.use_attn = cfg.SOLVER.USE_ATTN
        
        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS  # use the output of each transformer layer
        self.use_actioness = cfg.MODEL.STCAT.USE_ACTION
        self.query_dim = cfg.MODEL.STCAT.QUERY_DIM

        self.vis_encoder = build_vis_encoder(cfg)
        vis_fea_dim = self.vis_encoder.num_channels
        self.text_encoder = build_text_encoder(cfg)
        
        self.ground_encoder = build_encoder(cfg)
        self.ground_decoder = build_decoder(cfg)
        
        hidden_dim = cfg.MODEL.STCAT.HIDDEN  
        self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.action_embed = None
        if self.use_actioness:
            self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)
        
        # add the iteration anchor update
        self.ground_decoder.decoder.bbox_embed = self.bbox_embed

    def forward(self, videos, texts, logger=None):
        """
        Arguments:
            videos  (NestedTensor): N * C * H * W, N = sum(T) 
            durations : batch video length
            texts   (NestedTensor]): 
            targets (list[TargetTensor]): ground-truth
        Returns: 
        """
        # Visual Feature
        vis_outputs, vis_pos_embed = self.vis_encoder(videos)
        vis_features, vis_mask, vis_durations = vis_outputs.decompose()
        vis_features = self.input_proj(vis_features)
        vis_outputs = NestedTensor(vis_features, vis_mask, vis_durations)
        
        # Textual Feature
        device = vis_features.device
        text_outputs, text_cls = self.text_encoder(texts, device)  # text_cls : [b, d_model]
        
        # Multimodal Feature Encoding 
        encoded_memory = self.ground_encoder(
            videos=vis_outputs, vis_pos=vis_pos_embed, texts=text_outputs
        )
        
        # Query-based decoding
        outputs, outputs_temp = self.ground_decoder(
            memory_cache=encoded_memory, vis_pos=vis_pos_embed,
            text_cls=text_cls
        )
        
        out = {}
        if self.use_attn:
            time_hs, weights = outputs_temp
            out["weights"] = weights[-1]
        
        # the final decoder embeddings and the refer anchors
        hs, reference = outputs     # hs : [num_layers, b, T, d_model], reference : [num_layers, b, T, 4]
        ###############  predict bounding box ################        
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.bbox_embed(hs)
        tmp[..., :self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()  # [num_layers, b, T, 4]
        outputs_coord = outputs_coord.flatten(1,2)
        out.update({"pred_boxes": outputs_coord[-1]})
        #######################################################
        
        ###############  predict the start and end probability ################ 
        outputs_temp = self.temp_embed(time_hs)
        out.update({"pred_sted": outputs_temp[-1]})
        #######################################################
        
        if self.use_actioness:
            outputs_actioness = self.action_embed(time_hs)
            out.update({"pred_actioness": outputs_actioness[-1]})
        
        if self.use_aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_sted": a,
                    "pred_boxes": b,
                }
                for a, b in zip(outputs_temp[:-1], outputs_coord[:-1])
            ]
            for i_aux in range(len(out["aux_outputs"])):
                if self.use_attn:
                    out["aux_outputs"][i_aux]["weights"] = weights[i_aux]
                if self.use_actioness:
                    out["aux_outputs"][i_aux]["pred_actioness"] = outputs_actioness[i_aux]
        
        return out