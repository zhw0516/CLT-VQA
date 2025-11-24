from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from modeling_t5_our import VLT5
import math

from dual_balance import BCP, MFA, cross_entropy_loss

class VLT5VQA(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)
        self.config = config
        self.num_answers = num_answers
        self.label2ans = label2ans
        self.mfa = MFA(feat_in=config.d_model, eps=1, max_iter=200, reduction='mean', dis=config.dis).cuda()

        self.bcp = BCP(config.d_model,num_answers)

        self.criterion = cross_entropy_loss()


    def train_step(self, batch):

        if self.config.classifier:
            device = next(self.parameters()).device
            vis_feats = batch['vis_feats'].to(device)
            input_ids = batch['input_ids'].to(device)
            vis_pos = batch['boxes'].to(device)

            target = batch['targets'].to(device)
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1] # (80, 1, 768)
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1] # (80,768)


            logit = self.bcp(last_hidden_state)
            F = self.mfa.feature(last_hidden_state)

            P = self.bcp.prototypes()

            loss_mfa = self.mfa(F, P.T)
            loss = self.criterion(logit,target) + loss_mfa * self.config.l_mfa
            result = {
                'loss': loss,
            }
        return result


    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        # self.margin_model.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            logit = self.bcp(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        return result


