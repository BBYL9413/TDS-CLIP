import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import VisualTransformer, ModifiedResNet
import numpy as np
from einops import rearrange

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = self.net(x)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    
class TemporalShift_VIT(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift_VIT, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        x = self.net(x)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        hw, nt, c = x.size()
        cls_ = x[0,:,:].unsqueeze(0)
        x = x[1:,:,:]
#         print(cls_.size())
        x = x.permute(1,2,0)  # nt,c,hw
        n_batch = nt // n_segment
        h = int(np.sqrt(hw-1))
        w = h
        x = x.contiguous().view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        out = out.contiguous().view(nt, c, h*w)
        out = out.permute(2,0,1) #hw, nt, c
        out = torch.cat((cls_,out),dim=0)
#         print(out.size())
        return out



class TokenShift(nn.Module):
    def __init__(self, n_segment=3, n_div=4):
        super(TokenShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        # n bt c
        n, bt, c = x.size()
        b = bt // self.n_segment
        x = x.permute(1, 0, 2).contiguous().view(b, self.n_segment, n, c)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, 0, :fold] = x[:, 1:, 0, :fold] # shift left
        out[:, 1:,  0, fold:2*fold] = x[:,:-1:, 0, fold:2*fold] # shift right

        out[:, :, 1:, :2*fold] = x[:, :, 1:, :2*fold] # not shift
        out[:, :, :, 2*fold:] = x[:, :, :, 2*fold:] # not shift

        out = out.view(bt, n, c).permute(1, 0, 2).contiguous()

        return out

def make_tokenshift(net, n_segment, n_div=4, locations_list=[]):
    for idx, block in enumerate(net.transformer.resblocks):
        if idx in locations_list:
            net.transformer.resblocks[idx].control_point1 = TokenShift(
                n_segment=n_segment,
                n_div=n_div)
            # net.transformer.resblocks[idx].control_point2 = TokenShift(
            #     n_segment=n_segment,
            #     n_div=n_div)


class TokenT1D(nn.Module):
    def __init__(self, in_channels, n_segment=3, n_div=4, mode='shift'):
        super(TokenT1D, self).__init__()
        self.input_channels = in_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # n bt c
        n, bt, c = x.size()
        b = bt // self.n_segment
        x = x.permute(1, 0, 2).contiguous().view(b, self.n_segment, n, c)
        x = x.permute(0, 2, 3, 1).contiguous() # b, n, c, t
        out = torch.zeros_like(x)
        out[:, 0] = self.conv(x[:, 0])
        out[:, 1:] = x[:, 1:]
        out = out.permute(1, 0, 3, 2).contiguous().view(n, bt, c)
        return out

def make_tokenT1D(net, n_segment, n_div=4, locations_list=[]):
    for idx, block in enumerate(net.transformer.resblocks):
        if idx in locations_list:
            block.control_point1 = TokenT1D(
                in_channels=block.control_point1.inplanes,
                n_segment=n_segment,
                n_div=n_div)
            # block.control_point2 = TokenT1D(
            #     in_channels=block.control_point2.inplanes,
            #     n_segment=n_segment,
            #     n_div=n_div)


class TSM4VIT(nn.Module):
    def __init__(self, in_channels, n_segment=8, n_div=4, mode='fixed'):
        super(T1D4VIT, self).__init__()
        self.input_channels = in_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=3, padding=1, groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # n+1 bt c
        xt = x[1:, :, :]
        n, bt, c = xt.size()
        b = bt // self.n_segment
        H = W = int(n ** 0.5)
        xt = xt.view(H, W, b, self.n_segment, c).permute(2, 0, 1, 4, 3).contiguous()  # b h w c t
        xt = xt.view(b * H * W, c, self.n_segment)  # bhw c t
        xt = self.conv(xt)  # bhw c t
        xt = xt.view(b, H, W, c, self.n_segment).permute(1, 2, 0, 4, 3).contiguous().view(n, bt, c)
        x = torch.cat([x[:1, :, :], xt], dim=0)  # n+1 bt c
        return x

def make_TSM4VIT(net, n_segment, n_div=4, locations_list=[]):
    for idx, block in enumerate(net.transformer.resblocks):
        if idx in locations_list:
            block.control_point1 = T1D4VIT(
                in_channels=block.control_point1.inplanes,
                n_segment=n_segment,
                n_div=n_div)
            # block.control_point2 = T1D4VIT(
            #     in_channels=block.control_point2.inplanes,
            #     n_segment=n_segment,
            #     n_div=n_div)



class T1D4VIT(nn.Module):
    def __init__(self, in_channels, n_segment=8):
        super(T1D4VIT, self).__init__()
        self.n_segment = n_segment

        pos_kernel_size = 3
        padding = pos_kernel_size // 2
        h_dim = 512 if in_channels == 768 else 768 # 768 -> 512, 1024 -> 768

        self.tconv = nn.Sequential(
            nn.Conv3d(in_channels, h_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(h_dim),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(
                h_dim, h_dim, kernel_size=(pos_kernel_size, 1, 1), 
                stride=(1, 1, 1), padding=(padding, 0, 0), groups=h_dim),
            nn.BatchNorm3d(h_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(h_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

        # Zero-initialize the last conv in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        nn.init.constant_(self.tconv[-1].weight, 0)

    def forward(self, x):
        # n+1 bt c
        xt = x[1:, :, :]
        n, bt, c = xt.size()
        b = bt // self.n_segment
        H = W = int(n ** 0.5)
        xt = xt.view(H, W, b, self.n_segment, c).permute(2, 4, 3, 0, 1).contiguous()  # b c t h w
        xt = xt + self.tconv(xt)  # b c t h w
        xt = xt.view(b, c, self.n_segment, n).permute(3, 0, 2, 1).contiguous().view(n, bt, c)
        x = torch.cat([x[:1, :, :], xt], dim=0)  # n+1 bt c
        return x

def make_T1D4VIT(net, n_segment, locations_list=[]):
    for idx, block in enumerate(net.transformer.resblocks):
        if idx in locations_list:
            if hasattr(block, 'control_point1'):
                block.control_point1 = T1D4VIT(
                    in_channels=block.control_point1.inplanes,
                    n_segment=n_segment)
            if hasattr(block, 'control_point2'):        
                block.control_point2 = T1D4VIT(
                    in_channels=block.control_point2.inplanes,
                    n_segment=n_segment)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x
        
def make_temporal_shift_vit(net, n_segment, n_div=8, place='block', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    if isinstance(net, VisualTransformer):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift_VIT(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.transformer.resblocks = make_block_temporal(net.transformer.resblocks, n_segment_list[0])
            
#             net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
#             net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
#             net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        
    else:
        raise NotImplementedError(place)




def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    if isinstance(net, ModifiedResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)



def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class Local_MHRA(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__() 

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), 
                stride=(1, 1, 1), padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        print('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)


class LocalUni(nn.Module):
    def __init__(self, d_model, n_segment, n_head, dropout, attn_mask):
        super(LocalUni, self).__init__()
        self.d_model = d_model
        self.T = n_segment
        self.temporal_attn = Local_MHRA(d_model, dw_reduction=1.5)
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.attn_mask = attn_mask

    def forward(self, x):
        # n+1 bt c
        xt = x[1:, :, :]
        n, bt, c = xt.size()
        b = bt // self.T
        H = W = int(n ** 0.5)
        xt = xt.view(H, W, b, self.T, c).permute(2, 4, 3, 0, 1).contiguous()  # b c t h w
        xt = xt + self.drop_path(self.temporal_attn(xt))
        xt = xt.view(b, c, self.T, n).permute(3, 0, 2, 1).contiguous().view(n, bt, c)
        x = torch.cat([x[:1, :, :], xt], dim=0)  # n+1 bt c
        return x     


def make_LocalUni(net, n_segment, n_head, dropout, attn_mask, locations_list=[]):
    for idx, block in enumerate(net.transformer.resblocks):
        if idx in locations_list:
            block.control_point1 = LocalUni(
                d_model=block.control_point1.inplanes,
                n_segment=n_segment,
                n_head=n_head,
                dropout=dropout,
                attn_mask=attn_mask)
            # block.control_point2 = LocalUni(
            #     d_model=block.control_point2.inplanes,
            #     n_segment=n_segment,
            #     n_head=n_head,
            #     dropout=dropout,
            #     attn_mask=attn_mask)



# split CLS
# class DividedST(nn.Module):
#     def __init__(self, d_model, n_segment, n_head, dropout, attn_mask):
#         super(DividedST, self).__init__()
#         self.d_model = d_model
#         self.T = n_segment
#         self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout)
#         # self.temporal_fc = nn.Linear(d_model, d_model)
#         self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
#         self.temporal_ln = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#     def attention(self, x):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.temporal_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x):
#         # n+1 bt c
#         xt = x[1:, :, :]
#         n, bt, c = xt.size()
#         b = bt // self.T
#         xt = xt.view(n, b, self.T, c).permute(2, 1, 0, 3).contiguous()  # t b n c
#         xt = xt.view(self.T, b * n, c) # t bn c
#         xt = xt + self.drop_path(self.attention(self.temporal_ln(xt)))
#         xt = xt.view(self.T, b, n, c).permute(2, 1, 0, 3).contiguous().view(n, bt, c)
#         x = torch.cat([x[:1, :, :], xt], dim=0)  # n+1 bt c
#         return x     



class DividedST(nn.Module):
    def __init__(self, d_model, n_segment, n_head, dropout, attn_mask):
        super(DividedST, self).__init__()
        self.d_model = d_model
        self.T = n_segment
        self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout)
        # self.temporal_fc = nn.Linear(d_model, d_model)
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.temporal_ln = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.temporal_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        # n+1 bt c
        xt = x
        n, bt, c = xt.size()
        b = bt // self.T
        xt = xt.view(n, b, self.T, c).permute(2, 1, 0, 3).contiguous()  # t b n c
        xt = xt.view(self.T, b * n, c) # t bn c
        xt = xt + self.drop_path(self.attention(self.temporal_ln(xt)))
        xt = xt.view(self.T, b, n, c).permute(2, 1, 0, 3).contiguous().view(n, bt, c)
        return xt    


def make_DividedST(net, n_segment, n_head, dropout, attn_mask, locations_list=[]):
    for idx, block in enumerate(net.transformer.resblocks):
        if idx in locations_list:
            block.control_point1 = DividedST(
                d_model=block.control_point1.inplanes,
                n_segment=n_segment,
                n_head=n_head,
                dropout=dropout,
                attn_mask=attn_mask)



if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')