import random
import typing as T

import torch
from torch import nn
from diffusers.models.controlnet import ControlNetConditioningEmbedding


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResMlpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        y = self.block(x)
        if self.in_channels == self.out_channels:
            return x + y
        else:
            return x


class ResMlp(nn.Module):

    def __init__(self, num_layers=3, num_hidden_states=768, in_channel=704):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden_states = num_hidden_states
        self.fc1 = nn.Linear(in_channel, num_hidden_states)
        self.act1 = nn.SiLU()
        self.blocks = nn.ModuleList([
            ResMlpBlock(num_hidden_states, num_hidden_states)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(self.num_hidden_states, self.num_hidden_states)
        # nn.init.zeros_(self.out.weight)
        # nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        for module in self.blocks:
            x = module(x)
        y = self.out(x)
        return y
    

class SDXLMultiTokenConditioner(nn.Module):

    def __init__(self, in_channels={}, num_layers={}, num_tokens={}, token_dropout_rate={}) -> None:
        super().__init__()
        self.models = nn.ModuleDict()
        for k, ic in in_channels.items():
            self.models[k + '_te1'] = nn.ModuleList([
                ResMlp(in_channel=ic, num_layers=num_layers[k], num_hidden_states=768)
                for _ in range(num_tokens.get(k, 1))
            ])
            self.models[k + '_te2'] = nn.ModuleList([
                ResMlp(in_channel=ic, num_layers=num_layers[k], num_hidden_states=1280)
                for _ in range(num_tokens.get(k, 1))
            ])
        self.keys = sorted(in_channels.keys())
        self.token_dropout_rate = token_dropout_rate

    def forward(self, encoder_hidden_states, extra_conditions={}):

        # 1. encode extra conditions into tokens using mlps
        original_extra_features = []
        extra_features = []
        keys = [k for k in self.keys if k in extra_conditions]
        for k in keys:
            extra_feature = extra_conditions[k]
            if len(extra_feature.shape) == 2:
                batch_size = 1
                seq_len, ndim = extra_feature.shape
            elif len(extra_feature.shape) == 3:
                batch_size, seq_len, ndim = extra_feature.shape
            else:
                raise ValueError(f"extra_feature {k} must be 2 or 3 dimension")
            extra_feature = torch.reshape(extra_feature, (batch_size * seq_len, ndim))
            extra_feature1 = [m(extra_feature) for m in self.models[k + '_te1']]
            extra_feature2 = [m(extra_feature) for m in self.models[k + '_te2']]
            extra_feature1 = [torch.reshape(f, (batch_size, seq_len, -1)) for f in extra_feature1]
            extra_feature2 = [torch.reshape(f, (batch_size, seq_len, -1)) for f in extra_feature2]
            extra_features.append((extra_feature1, extra_feature2))
            original_extra_features.append(extra_conditions[k])
        
        assert len(encoder_hidden_states) == 3 and len(extra_features[0]) == 2, \
            "ensure encoder = %d and feature = %d" % (len(encoder_hidden_states), len(extra_features))
        
        # 2. replacing the tokens in encoder_hidden_states with the tokens from extra_features
        new_encoder_hidden_states = []
        for sdxl_te_idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if sdxl_te_idx == 2:
                new_encoder_hidden_states.append(encoder_hidden_state)
                continue
            token_replace_offset_each_sample = [0 for _ in range(batch_size)]
            counter = {k: [0 for _ in range(batch_size)] for k in self.keys}
            for extra_feature_idx, key in enumerate(keys):
                extra_feature = extra_features[extra_feature_idx][sdxl_te_idx] # [(b, seq, dim) * num_tokens]
                for extra_token in extra_feature:
                    original_extra_feature = original_extra_features[extra_feature_idx]
                    inplaced = False
                    for batch_idx in range(batch_size):
                        sample_token = extra_token[batch_idx] # (seq, dim)
                        original_sample_feature = original_extra_feature[batch_idx] # (seq, dim)
                        num_positive = sum((original_sample_feature[:, -256:].abs().sum(-1) > 1e-6).cpu().numpy())
                        if not self.training or random.random() > self.token_dropout_rate.get(key, 0):
                            if num_positive > 0:
                                for token_count in range(0, num_positive):
                                    token_pos = - token_replace_offset_each_sample[batch_idx] - 1
                                    encoder_hidden_state[batch_idx, token_pos] = sample_token[token_count]
                                    token_replace_offset_each_sample[batch_idx] += 1
                                    counter[key][batch_idx] += 1
                                encoder_hidden_state[batch_idx, -num_positive-token_replace_offset_each_sample[batch_idx]:-token_replace_offset_each_sample[batch_idx]] = sample_token[:num_positive]
                                inplaced = True
                    if not inplaced:
                        encoder_hidden_state += sample_token.sum() * 0 # enable (zero) gradient backprop to conditioner weights
            # print(token_replace_offset_each_sample, counter)
            new_encoder_hidden_states.append(encoder_hidden_state)
        return tuple(new_encoder_hidden_states)

class Conditioner(nn.Module):

    def __init__(self, configs={}):
        super().__init__()
        self.models = nn.ModuleDict()
        for k, v in configs.items():
            self.models[k] = ResMlp(**v)
        self.keys = sorted(configs.keys())

    def forward_sdxl(self, encoder_hidden_states, extra_conditions={}, position='inplace'):
        original_extra_features = []
        extra_features = []
        # print(self.keys)
        for k in self.keys:
            extra_feature = extra_conditions[k]
            if len(extra_feature.shape) == 2:
                b = 1
                seq_len, ndim = extra_feature.shape
            elif len(extra_feature.shape) == 3:
                b, seq_len, ndim = extra_feature.shape
            else:
                raise ValueError(f"extra_feature {k} must be 2 or 3 dimension")
            extra_feature = torch.reshape(extra_feature, (b * seq_len, ndim))
            extra_feature = self.models[k](extra_feature)
            extra_feature = torch.reshape(extra_feature, (b, seq_len, -1))
            extra_features.append(extra_feature)
            original_extra_features.append(extra_conditions[k])
        assert len(encoder_hidden_states) == 3 and len(extra_features) == 2, \
            "ensure encoder = %d and feature = %d" % (len(encoder_hidden_states), len(extra_features))
        
        new_encoder_hidden_states = []
        for i, encoder_hidden_state in enumerate(encoder_hidden_states):
            if i != 2:
                b, seq_len, ndim = encoder_hidden_state.shape
                if position == 'front':
                    encoder_hidden_state = torch.cat([extra_features[i], encoder_hidden_state], dim=1)
                elif position == 'back':
                    encoder_hidden_state = torch.cat([encoder_hidden_state, extra_features[i]], dim=1)
                elif position == 'replace':
                    encoder_hidden_state = extra_features[i]
                elif position == 'inplace':
                    f = extra_features[i]
                    of = original_extra_features[i]
                    for _b in range(b):
                        _f = f[_b]
                        _of = of[_b]
                        num_positive = sum((_of[:, -256:].abs().sum(-1) > 1e-6).cpu().numpy())
                        if num_positive > 0:
                            encoder_hidden_state[_b, -num_positive:] = _f[:num_positive]
                        else:
                            encoder_hidden_state += _f.sum() * 0 # enable (zero) gradient backprop to conditioner weights
            new_encoder_hidden_states.append(encoder_hidden_state)
        return tuple(new_encoder_hidden_states)


    def forward_sd(self, encoder_hidden_states, extra_conditions={}, position='front', scale=1.0):
        extra_features = []
        # print(self.keys)
        for k in self.keys:
            extra_feature = extra_conditions[k]
            if len(extra_feature.shape) == 2:
                b = 1
                seq_len, ndim = extra_feature.shape
            elif len(extra_feature.shape) == 3:
                b, seq_len, ndim = extra_feature.shape
            else:
                raise ValueError(f"extra_feature {k} must be 2 or 3 dimension")
            extra_feature = torch.reshape(extra_feature, (b * seq_len, ndim))
            extra_feature = self.models[k](extra_feature)
            extra_feature = torch.reshape(extra_feature, (b, seq_len, -1))
            extra_features.append(extra_feature)
        extra_features = torch.cat(extra_features, dim=1)
        if position == 'front':
            encoder_hidden_states = torch.cat([extra_features * scale, encoder_hidden_states], dim=1)
        elif position == 'back':
            encoder_hidden_states = torch.cat([encoder_hidden_states, extra_features * scale], dim=1)
        elif position == 'replace':
            encoder_hidden_states = extra_features * scale
        else:
            raise ValueError(f"position must be 'front' or 'back'")
        return encoder_hidden_states

    def forward(self, encoder_hidden_states, extra_conditions={}, position='front', is_sdxl=False):
        return self.forward_sd(encoder_hidden_states, extra_conditions, position) if not is_sdxl else \
            self.forward_sdxl(encoder_hidden_states, extra_conditions, position)


class StructuralConditioner(nn.Module):

    def __init__(
        self, 
        conditions: T.List[str] = ['canny'], 
        conditioning_channels: int = 3,
        conditioning_embedding_channels: int = 320, 
        block_out_channels: T.List[int] = [16, 32, 64, 128]
    ):
        super().__init__()
        self.cond_embedders = nn.ModuleDict()
        for c in conditions:
            self.cond_embedders[c] = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=conditioning_embedding_channels,
                block_out_channels=block_out_channels,
                conditioning_channels=conditioning_channels
            )
    
    def forward(self, conditions: T.Dict):
        x = 0
        for k, cond in conditions.items():
            x += self.cond_embedders[k](cond)
        return x


if __name__ == "__main__":
    conditioner = Conditioner({
        'face_encoder_1': dict(num_layers=3, num_hidden_states=768, in_channel=704),
        'face_encoder_2': dict(num_layers=3, num_hidden_states=1024, in_channel=704),
    }).cuda()
    test_input = torch.cat([torch.randn(2, 3, 704), torch.zeros((2, 3, 704)), ], dim=1).cuda()
    test_hidden_states = torch.randn(2, 77, 768).cuda()
    test_hidden_states2 = torch.randn(2, 77, 1024).cuda()
    pool2 = torch.randn(2, 1024).cuda()
    output = conditioner.forward_sdxl(
        (test_hidden_states, test_hidden_states2, pool2), {
            'face_encoder_1': test_input,
            'face_encoder_2': test_input,
        }, position='inplace'
    )
    for o in output:
        print(o.shape)
