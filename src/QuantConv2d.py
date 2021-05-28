import torch
import torch.nn as nn
import cu_gemm_quant
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
        self._shift_opt = None
        self._bit_group_x = None
        self._bit_group_w = None
        self._group_sz_x = None
        self._group_sz_w = None
        self._sparq_x = None
        self._sparq_w = None

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

        if not self._sparq_x and not self._sparq_w:
            out = nn.functional.conv2d(x_q * x_q_delta,
                                       w_q * w_q_delta[:, None, None, None].expand_as(w_q),
                                       bias=bias_fp,
                                       stride=(self.stride[0], self.stride[1]),
                                       padding=(self.padding[0], self.padding[1]), groups=self.groups)
        else:
            # At the moment, unfold and quantization must go together
            assert (self._quantize is True)
            assert (self._is_round is not None)
            assert (self._shift_opt == 5)

            if self._sparq_x:
                # C-W-H ordering
                x_q = x_q.permute(0, 2, 3, 1)

                x_sl = torch.where(x_q >= 2 ** 7, torch.ones_like(x_q) * 4, torch.zeros_like(x_q))
                x_sl = x_sl + torch.where((x_q < 2 ** 7) & (x_q >= 2 ** 6), torch.ones_like(x_q) * 3, torch.zeros_like(x_q))
                x_sl = x_sl + torch.where((x_q < 2 ** 6) & (x_q >= 2 ** 5), torch.ones_like(x_q) * 2, torch.zeros_like(x_q))
                x_sl = x_sl + torch.where((x_q < 2 ** 5) & (x_q >= 2 ** 4), torch.ones_like(x_q) * 1, torch.zeros_like(x_q))

                x_sl = x_sl.flatten()
                x_sl = x_sl.reshape([int(x_sl.size(0) / self._group_sz_x), self._group_sz_x])
                x_sl_max = x_sl.max(dim=1)[0]
                x_sl_max = x_sl_max[:, None].expand_as(x_sl)
                x_sl_max = (2 ** x_sl_max).reshape_as(x_q)

                if self._is_round:
                    x_q = torch.round(x_q / x_sl_max) * x_sl_max

                    x_q = torch.where((x_sl_max == 16) & (x_q == 256), torch.ones_like(x_q) * 240, x_q)
                    x_q = torch.where((x_sl_max == 8) & (x_q == 128), torch.ones_like(x_q) * 120, x_q)
                    x_q = torch.where((x_sl_max == 4) & (x_q == 64), torch.ones_like(x_q) * 60, x_q)
                    x_q = torch.where((x_sl_max == 2) & (x_q == 32), torch.ones_like(x_q) * 30, x_q)
                    x_q = torch.where((x_sl_max == 1) & (x_q == 16), torch.ones_like(x_q) * 15, x_q)
                else:
                    x_q = torch.floor(x_q / x_sl_max) * x_sl_max

                x_q = x_q.permute(0, 3, 1, 2)

                x_sl = x_sl_max = None

            if self._sparq_w:
                # N-W-H order
                w_q = w_q.permute(0, 2, 3, 1)

                w_sl = torch.where((w_q.abs() >= 2 ** 6), torch.ones_like(w_q) * 4, torch.zeros_like(w_q))
                w_sl = w_sl + torch.where((w_q.abs() < 2 ** 6) & (w_q.abs() >= 2 ** 5), torch.ones_like(w_q) * 3, torch.zeros_like(w_q))
                w_sl = w_sl + torch.where((w_q.abs() < 2 ** 5) & (w_q.abs() >= 2 ** 4), torch.ones_like(w_q) * 2, torch.zeros_like(w_q))
                w_sl = w_sl + torch.where((w_q.abs() < 2 ** 4) & (w_q.abs() >= 2 ** 3), torch.ones_like(w_q) * 1, torch.zeros_like(w_q))
                # TODO: is it precise with the signed numbers?

                # self._group_sz_w = self.weight.size(2) * self.weight.size(3)

                w_sl = w_sl.flatten()
                w_sl = w_sl.reshape([int(w_sl.size(0) / self._group_sz_w), self._group_sz_w])
                w_sl_max = w_sl.max(dim=1)[0]
                w_sl_max = w_sl_max[:, None].expand_as(w_sl)
                w_sl_max = (2 ** w_sl_max).reshape_as(w_q)

                if self._is_round:
                    w_q = torch.round(w_q / w_sl_max) * w_sl_max

                    w_q = torch.where((w_sl_max == 16) & (w_q == 128), torch.ones_like(w_q) * 112, w_q)
                    w_q = torch.where((w_sl_max == 8) & (w_q == 64), torch.ones_like(w_q) * 56, w_q)
                    w_q = torch.where((w_sl_max == 4) & (w_q == 32), torch.ones_like(w_q) * 28, w_q)
                    w_q = torch.where((w_sl_max == 2) & (w_q == 16), torch.ones_like(w_q) * 14, w_q)
                    w_q = torch.where((w_sl_max == 1) & (w_q == 8), torch.ones_like(w_q) * 7, w_q)
                    # TODO: handle overflows
                else:
                    w_q = torch.floor(w_q / w_sl_max) * w_sl_max

                # Back to W-H-N order
                w_q = w_q.permute(0, 3, 1, 2)

                w_sl = w_sl_max = None

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

        key.extend(['is_round', 'shift_opt'])
        val.extend([self._is_round, self._shift_opt]) if self._sparq_x or self._sparq_w else val.extend(['-', '-'])

        key.extend(['bit_grp_x', 'grp_sz_x'])
        val.extend([self._bit_group_x, self._group_sz_x]) if self._sparq_x else val.extend(['-', '-'])

        key.extend(['bit_grp_w', 'grp_sz_w'])
        val.extend([self._bit_group_w, self._group_sz_w]) if self._sparq_w else val.extend(['-', '-'])

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
