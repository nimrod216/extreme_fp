import torch
import torch.nn as nn
import cu_sparq
import Config as cfg
import matplotlib.pyplot as plt


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UnfoldConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(UnfoldConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                                           bias=bias, padding_mode=padding_mode)

        # Registering buffers to be saved when calling torch.save
        self.register_buffer('tracked_n', torch.zeros(1))
        self.register_buffer('max_mean', torch.zeros(1))
        self.register_buffer('min_mean', torch.zeros(1))

        # Even if in training mode, the user can disable gathering the tensor min-max values
        self._disable_min_max_update = False

        # Quantization variables
        self._quantize = False
        self._x_bits = 8
        self._w_bits = 8

        # Custom kernel variables
        self._is_round = None
        self._shift_opt_x = None
        self._shift_opt_w = None

    def _reset_stats(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self._reset_stats(v)
            else:
                d[k] = 0

    def reset_stats(self):
        self._reset_stats(self.stats)

    def forward(self, x):
        # Prepare activations, weights, and bias
        if self._quantize:

            # Gather statistics during training
            if self.training and not self._disable_min_max_update:
                tracked_n_old = self.tracked_n.clone()
                self.tracked_n += x.size(0)

                max_sum = x.detach().max(dim=3).values.max(dim=2).values.max(dim=1).values.sum()
                min_sum = x.detach().min(dim=3).values.min(dim=2).values.min(dim=1).values.sum()

                self.max_mean = ((self.max_mean * tracked_n_old) + max_sum) / self.tracked_n
                self.min_mean = ((self.min_mean * tracked_n_old) + min_sum) / self.tracked_n

            # These statistics are mandatory for quantization
            assert (self.max_mean != 0 or self.min_mean != 0)

            # Activations quantization
            # Only supports unsigned uniform quantization
            if torch.min(x) == 0:
                x_q, x_q_delta = self._uniform_quantization(x, self.max_mean, self._x_bits)
                x_q = x_q.int().float()         # Just in case
                assert (x_q.max() <= ((2 ** self._x_bits) - 1) and x_q.min() >= 0)
            else:
                cfg.LOG.write('Error: not supporting signed activation quantization')
                raise NotImplementedError

            # Weights quantization
            w_q, w_q_delta = \
                self._uniform_symmetric_quantization_per_filter(self.weight,
                                                                self.weight.data.min(dim=3)[0].min(dim=2)[0].min(dim=1)[0],
                                                                self.weight.data.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0],
                                                                self._w_bits)

            w_q = w_q.int().float()   # Just in case
            assert (w_q.max() <= ((2 ** self._w_bits) / 2 - 1) and w_q.min() >= (-2 ** self._w_bits) / 2)

            # Bias quantization
            if self.bias is None:
                bias_fp = None
            else:
                bias_q, bias_q_delta = self._uniform_symmetric_quantization(self.bias,
                                                                            torch.min(self.bias.data),
                                                                            torch.max(self.bias.data), self._w_bits)

                assert (bias_q.max() <= ((2 ** self._w_bits) / 2 - 1) and bias_q.min() >= (-2 ** self._w_bits) / 2)

                bias_fp = bias_q * bias_q_delta

        else:
            # The single scalar movement to CUDA may be bad for performance
            x_q, x_q_delta = x, torch.Tensor([1]).cuda()
            w_q, w_q_delta = self.weight, torch.Tensor([1]).cuda()
            bias_fp = self.bias

        out = nn.functional.conv2d(x_q * x_q_delta,
                                   w_q * w_q_delta[:, None, None, None].expand_as(w_q),
                                   bias=bias_fp,
                                   stride=(self.stride[0], self.stride[1]),
                                   padding=(self.padding[0], self.padding[1]), groups=self.groups)
        return out

    def get_status_arr(self):
        key, val = [], []

        key.extend(['quant', 'x_b', 'w_b'])
        val.extend([self._quantize, self._x_bits, self._w_bits])

        key.extend(['sparq_x', 'sparq_w'])
        val.extend([self._sparq_x, self._sparq_w])

        key.extend(['is_round'])
        val.extend([self._is_round]) if self._sparq_x or self._sparq_w else val.extend(['-', '-'])

        key.extend(['shift_opt_x'])
        val.extend([self._shift_opt_x]) if self._sparq_x else val.extend(['-'])

        key.extend(['shift_opt_w'])
        val.extend([self._shift_opt_w]) if self._sparq_w else val.extend(['-'])

        key.extend(['grp_sz_x'])
        val.extend([self._group_sz_x]) if self._sparq_x else val.extend(['-', '-'])

        key.extend(['grp_sz_w'])
        val.extend([self._group_sz_w]) if self._sparq_w else val.extend(['-', '-'])

        return key, val

    @staticmethod
    def _uniform_quantization(x, x_max, bits):
        N = 2 ** bits
        delta = x_max / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, 0, N - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization_per_filter(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = torch.where(x_min.abs() > x_max.abs(), x_min.abs(), x_max.abs()) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta[:, None, None, None].expand_as(x))
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = max(abs(x_min), abs(x_max)) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta
