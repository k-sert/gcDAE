"""
Network definitions from https://github.com/ferrine/hyrnn
"""

from audioop import bias
import geoopt
import geoopt.manifolds.stereographic.math as gmath
import math

import torch
import torch.nn.init as init
import numpy as np
import torch.nn
import torch.nn.functional
from torch.cuda.amp import autocast

def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def hyperbolic_softmax(X, A, P, c):
    a_norm = torch.norm(A, dim=1).clamp_min(1e-8)
    lambda_pkc = 2 / (1 - c * P.pow(2).sum(dim=1)).clamp_min(1e-5)
    k = lambda_pkc * a_norm / torch.sqrt(c)
    mob_add = _mobius_addition_batch(-P, X, c)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
    denom = a_norm.view(-1, 1) * (
        1 - c * mob_add.pow(2).sum(dim=2)).clamp_min(1e-5)
    logit = k.unsqueeze(1) * gmath.arsinh(num / denom)
    return logit.permute(1, 0)


def clip_poincare_ball(x, c, max_norm_ratio=None):
    x = gmath.project(x, k=-c)
    if max_norm_ratio is None or max_norm_ratio <= 0:
        return x
    max_norm_ratio = torch.as_tensor(
        max_norm_ratio, dtype=x.dtype, device=x.device)
    max_norm = max_norm_ratio / torch.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = (max_norm / norm).clamp(max=1.0)
    return gmath.project(x * scale, k=-c)


def radius_ratio_to_tangent_norm(max_radius_ratio, c):
    max_radius_ratio = torch.as_tensor(
        max_radius_ratio, dtype=c.dtype, device=c.device)
    max_radius_ratio = max_radius_ratio.clamp(min=1e-6, max=1.0 - 1e-6)
    artanh_ratio = 0.5 * (
        torch.log1p(max_radius_ratio) - torch.log1p(-max_radius_ratio))
    return artanh_ratio / torch.sqrt(c)


def smooth_bound_tangent_norm(x, c, max_radius_ratio=None):
    if max_radius_ratio is None or max_radius_ratio <= 0:
        return x, torch.ones_like(x[..., :1])
    max_tangent_norm = radius_ratio_to_tangent_norm(max_radius_ratio, c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = max_tangent_norm * torch.tanh(norm / max_tangent_norm) / norm
    return x * scale, scale


def record_stat(stats, name, value):
    if stats is not None:
        stats[name] = value.detach()


def mobius_linear(
        input,
        weight,
        bias=None,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        k=-1.0,
        max_norm_ratio=None,
        max_tangent_radius_ratio=None,
        normalize_input=False,
        stats=None,
):
    k = torch.as_tensor(k, dtype=input.dtype, device=input.device)
    c = k.abs()
    if hyperbolic_input:
        output = mobius_matvec(weight, input, k=k)
    else:
        input_norm = input.norm(dim=-1)
        record_stat(stats, 'input_norm_mean', input_norm.mean())
        record_stat(stats, 'input_norm_max', input_norm.max())
        if normalize_input:
            input = torch.nn.functional.layer_norm(input, input.shape[-1:])
            normalized_input_norm = input.norm(dim=-1)
            record_stat(stats, 'normalized_input_norm_mean', normalized_input_norm.mean())
            record_stat(stats, 'normalized_input_norm_max', normalized_input_norm.max())
        output = torch.nn.functional.linear(input, weight)
        tangent_norm = output.norm(dim=-1)
        tangent_radius_frac = torch.tanh(torch.sqrt(c) * tangent_norm)
        record_stat(stats, 'tangent_norm_mean', tangent_norm.mean())
        record_stat(stats, 'tangent_norm_max', tangent_norm.max())
        record_stat(stats, 'unbounded_radius_frac_mean', tangent_radius_frac.mean())
        record_stat(stats, 'unbounded_radius_frac_max', tangent_radius_frac.max())
        output, tangent_scale = smooth_bound_tangent_norm(
            output,
            c,
            max_radius_ratio=max_tangent_radius_ratio)
        bounded_tangent_norm = output.norm(dim=-1)
        bounded_radius_frac = torch.tanh(torch.sqrt(c) * bounded_tangent_norm)
        record_stat(stats, 'bounded_tangent_norm_mean', bounded_tangent_norm.mean())
        record_stat(stats, 'bounded_tangent_norm_max', bounded_tangent_norm.max())
        record_stat(stats, 'bounded_radius_frac_mean', bounded_radius_frac.mean())
        record_stat(stats, 'bounded_radius_frac_max', bounded_radius_frac.max())
        record_stat(stats, 'tangent_scale_mean', tangent_scale.mean())
        record_stat(stats, 'tangent_scale_min', tangent_scale.min())
        #print(weight)
        #print(output)
        output = gmath.expmap0(output, k=k)
        #output = gmath.expmap0(input, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = gmath.expmap0(bias, k=k)
        output = gmath.mobius_add(output, bias.unsqueeze(0).expand_as(output), k=k)
    if nonlin is not None:
        output = gmath.mobius_fn_apply(nonlin, output, k=k)
    output = clip_poincare_ball(output, c=c, max_norm_ratio=max_norm_ratio)
    return output


def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_matvec(m, x, k, dim=dim)


def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        # mx = torch.tensordot(x, m, [dim], [1])
        mx = torch.matmul(m, x.transpose(1, 0)).transpose(1, 0)
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = gmath.tan_k(mx_norm / x_norm * gmath.artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res

class HyperbolicMLR(torch.nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(
            self,
            ball_dim,
            n_classes,
            c,
            point_init_std=1e-2,
            normal_init_gain=5.0,
            max_point_norm_ratio=0.9):
        super(HyperbolicMLR, self).__init__()
        c_value = float(torch.as_tensor(c).detach().cpu())
        self.ball = geoopt.PoincareBall(c=c_value)
        self.a_vals = torch.nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = geoopt.ManifoldParameter(
            torch.zeros(n_classes, ball_dim), manifold=self.ball)
        self.c = c_value
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.point_init_std = point_init_std
        self.normal_init_gain = normal_init_gain
        self.max_point_norm_ratio = max_point_norm_ratio
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = clip_poincare_ball(
            self.p_vals.type_as(x),
            c,
            max_norm_ratio=self.max_point_norm_ratio)
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        conformal_factor = conformal_factor.clamp_min(1e-5)
        a_vals_poincare = self.a_vals.type_as(x) * conformal_factor
        logits = hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        with torch.no_grad():
            self.a_vals.mul_(self.normal_init_gain)
            c = torch.as_tensor(self.c, dtype=self.p_vals.dtype, device=self.p_vals.device)
            tangent_points = torch.empty(
                self.n_classes,
                self.ball_dim,
                dtype=self.p_vals.dtype,
                device=self.p_vals.device).normal_(
                    mean=0.0, std=self.point_init_std)
            points = gmath.expmap0(tangent_points, k=-c)
            points = clip_poincare_ball(
                points, c, max_norm_ratio=self.max_point_norm_ratio)
            self.p_vals.copy_(points)


# build custom module for logistic regression
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class MobiusLinear(torch.nn.Linear):
    def __init__(
            self,
            *args,
            hyperbolic_input=True,
            hyperbolic_bias=True,
            nonlin=None,
            k=-1.0,
            fp64_hyper=False,
            init_std=1e-3,
            bias_init_std=1e-3,
            max_norm_ratio=None,
            max_tangent_radius_ratio=None,
            normalize_input=False,
            **kwargs
    ):
        k = torch.as_tensor(k).clone().detach().float()
        super().__init__(*args, **kwargs)
        self.register_buffer("k", k, persistent=False)
        self.ball = geoopt.PoincareBall(c=k.abs())
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
            with torch.no_grad():
                tangent_bias = torch.empty_like(self.bias).normal_(
                    mean=0.0, std=bias_init_std)
                if hyperbolic_bias:
                    self.bias.set_(gmath.expmap0(tangent_bias, k=k))
                else:
                    self.bias.copy_(tangent_bias)
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=init_std)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.max_norm_ratio = max_norm_ratio
        self.max_tangent_radius_ratio = max_tangent_radius_ratio
        self.normalize_input = normalize_input
        self.fp64_hyper = fp64_hyper
        self.last_tangent_stats = {}

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            stats = {}
            output = mobius_linear(
                input,
                weight=self.weight,
                bias=self.bias,
                hyperbolic_input=self.hyperbolic_input,
                nonlin=self.nonlin,
                hyperbolic_bias=self.hyperbolic_bias,
                k=self.k,
                max_norm_ratio=self.max_norm_ratio,
                max_tangent_radius_ratio=self.max_tangent_radius_ratio,
                normalize_input=self.normalize_input,
                stats=stats,
            )
            self.last_tangent_stats = stats
            return output
            
    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        if self.max_tangent_radius_ratio is not None:
            info += ", max_tangent_radius_ratio={}".format(
                self.max_tangent_radius_ratio)
        if self.normalize_input:
            info += ", normalize_input=True"
        return info


class SphericalLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            k=1.0,
            normalize_input=False,
            max_tangent_radius_ratio=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = torch.tensor(k)
        self.normalize_input = normalize_input
        self.max_tangent_radius_ratio = max_tangent_radius_ratio
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        k = self.k.type_as(x)
        if self.normalize_input:
            x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        x_tangent = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.max_tangent_radius_ratio is not None and self.max_tangent_radius_ratio > 0:
            max_norm = (
                torch.as_tensor(self.max_tangent_radius_ratio, dtype=x.dtype, device=x.device)
                * math.pi
                / (2.0 * torch.sqrt(k))
            )
            norm = x_tangent.norm(dim=-1, keepdim=True).clamp_min(1e-15)
            scale = max_norm * torch.tanh(norm / max_norm) / norm
            x_tangent = x_tangent * scale
        return gmath.expmap0(x_tangent, k=k)


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, k=-1.0, fp64_hyper=True):
        k = torch.tensor(k)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=k.abs())
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = gmath.expmap0(point, k=k)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        self.fp64_hyper = fp64_hyper
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            input = input.unsqueeze(-2)
            distance = gmath.dist2plane(
                x=input, p=self.point, a=self.tangent, k=self.ball.c, signed=True
            )
            return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}"
            #             "c={ball.c}".format(
            #                 **self.__dict__
            #             )
        )
