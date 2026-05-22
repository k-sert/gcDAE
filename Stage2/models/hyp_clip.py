"""
This file defines the core research contribution
"""
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from configs.paths_config import model_paths
from models.hyper_nets import MobiusLinear, HyperbolicMLR, LogisticRegression, SphericalLinear
import torch.nn.functional as F
from torch import nn
import torch
import geoopt.manifolds.stereographic.math as gmath
import math
from audioop import bias
import matplotlib
from .xf import LayerNorm, Transformer
matplotlib.use('Agg')


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
              if k[:len(name)] == name}
    return d_filt


class EqualLinear_encoder(nn.Module):
    def __init__(
            self, in_dim, out_dim):
        super(EqualLinear_encoder, self).__init__()
        self.out_dim = out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        return out


class EqualLinear_decoder(nn.Module):
    def __init__(
            self, in_dim, out_dim):
        super(EqualLinear_decoder, self).__init__()
        self.out_dim = out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.fc3 = nn.Linear(in_dim, in_dim)

        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        return out


class transformer_encoder(nn.Module):
    def __init__(self, dim):
        super(transformer_encoder, self).__init__()
        self.encoder = EqualLinear_encoder(1024, dim)

    def forward(self, dw):
        x = self.encoder(dw)
        return x


class transformer_decoder(nn.Module):
    def __init__(self, dim):
        super(transformer_decoder, self).__init__()
        self.dim = dim
        self.decoder_low = EqualLinear_decoder(dim, 1024)

    def forward(self, dw):
        shape = dw[:, :self.dim].shape
        x0 = self.decoder_low(dw[:, :self.dim])
        x = x0.reshape((shape[0], 1024))
        return x
    

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--openai--clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
            1,
            1024,
            5,
            1,
        )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image, inv=False):
        outputs = self.transformer(pixel_values=image)
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image, inv=False, device=None):
        if device is not None:
            self.device = device
        return self(image, inv)


class EuclideanGeometryHead:
    def __init__(self, linear, classifier):
        self.linear = linear
        self.classifier = classifier

    def __call__(self, feature):
        feature_dist = self.linear(feature)
        return self.from_geometry(feature_dist)

    def from_geometry(self, feature_dist):
        logits = F.log_softmax(self.classifier(feature_dist), dim=-1)
        feature_euc = feature_dist
        return logits, feature_dist, feature_euc


class HyperbolicGeometryHead:
    def __init__(self, hyperbolic_linear, classifier):
        self.hyperbolic_linear = hyperbolic_linear
        self.classifier = classifier

    def __call__(self, feature):
        feature_dist = self.hyperbolic_linear(feature)
        return self.from_geometry(feature_dist)

    def from_geometry(self, feature_dist):
        logits = F.log_softmax(
            self.classifier(feature_dist, self.classifier.c), dim=-1)
        feature_euc = gmath.logmap0(
            feature_dist,
            k=torch.tensor(-float(self.classifier.c)).type_as(feature_dist))
        return logits, feature_dist, feature_euc


def clip_tangent_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_norm / norm).clamp(max=1.0)
    return x * scale


class SphericalPrototypeClassifier(nn.Module):
    def __init__(
            self,
            ball_dim,
            n_classes,
            k,
            prototype_init_radius=1.0,
            logit_scale_init=10.0):
        super(SphericalPrototypeClassifier, self).__init__()
        self.prototype_tangents = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.k = k
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.prototype_init_radius = prototype_init_radius
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            directions = torch.randn(
                self.n_classes,
                self.ball_dim,
                device=self.prototype_tangents.device,
                dtype=self.prototype_tangents.dtype)
            directions = F.normalize(directions, dim=1)
            self.prototype_tangents.copy_(
                directions * self.prototype_init_radius)

    def get_prototypes(self, like):
        k = torch.as_tensor(self.k, dtype=like.dtype, device=like.device)
        max_tangent_norm = 0.9 * math.pi / (2.0 * torch.sqrt(k))
        prototype_tangents = clip_tangent_norm(
            self.prototype_tangents.type_as(like),
            max_tangent_norm)
        prototypes = gmath.expmap0(prototype_tangents, k=k)
        return gmath.project(prototypes, k=k)

    def forward(self, x):
        k = torch.as_tensor(self.k).type_as(x)
        prototypes = self.get_prototypes(x)
        distances = gmath.dist(
            x.unsqueeze(1), prototypes.unsqueeze(0), k=k, dim=-1)
        return -distances * self.logit_scale.exp()


class SphericalGeometryHead:
    def __init__(self, spherical_linear, classifier, k):
        self.spherical_linear = spherical_linear
        self.classifier = classifier
        self.k = k

    def __call__(self, feature):
        # Feature maps from Euclidean to Spherical directly inside the layer
        feature_dist = self.spherical_linear(feature)
        
        # Project to ensure floating point precision keeps it strictly on the sphere
        k = torch.as_tensor(self.k, dtype=feature_dist.dtype, device=feature_dist.device)
        feature_dist = gmath.project(feature_dist, k=k)
        
        return self.from_geometry(feature_dist)

    def from_geometry(self, feature_dist):
        k = torch.as_tensor(self.k).type_as(feature_dist)
        logits = F.log_softmax(self.classifier(feature_dist), dim=-1)
        
        # Project back to Euclidean space using the logarithmic map for the decoder
        feature_euc = gmath.logmap0(feature_dist, k=k)
        return logits, feature_dist, feature_euc

class hae_clip(nn.Module):

    def __init__(self, opts):
        super(hae_clip, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder(self.opts.encoder_version)
        self.feature_shape = self.opts.feature_size
        self.curvature = torch.tensor(-self.opts.c, dtype=torch.float32)
        self.spherical_curvature = torch.tensor(self.opts.c, dtype=torch.float32)
        if self.opts.dataset_type == 'flowers_encode':
            self.num_classes = 102  # animal_faces 151, flowers 102
        elif self.opts.dataset_type == 'flowers_encode_eva':
            self.num_classes = 85
        elif self.opts.dataset_type == 'animalfaces_encode':
            self.num_classes = 151
        elif self.opts.dataset_type == 'animalfaces_encode_eva':
            self.num_classes = 121
        elif self.opts.dataset_type == 'vggfaces_encode':
            self.num_classes = 2374
        elif self.opts.dataset_type == 'vggfaces_encode_eva':
            self.num_classes = 1802
        elif self.opts.dataset_type == 'ffhq_encode':
            self.num_classes = 2374
        elif self.opts.dataset_type == 'nabirds_encode':
            self.num_classes = 555
        elif self.opts.dataset_type == 'nabirds_encode_eva':
            self.num_classes = 444
        elif self.opts.dataset_type == 'cub200_encode':
            self.num_classes = 200
        elif self.opts.dataset_type == 'caltech256_encode':
            self.num_classes = 257
        else:
            raise Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        self.transformer_encoder = Transformer(
            1,
            1024,
            5,
            1,
        )
        
        self.linear_encoder = LogisticRegression(1024, self.feature_shape)
        self.linear_decoder = LogisticRegression(self.feature_shape, 1024)
        if self.opts.dataset_type == 'flowers_encode':
            self.transformer_decoder = Transformer(
                1,
                1024,
                5,
                1,
            )
        else:
            self.transformer_decoder = Transformer(
                1,
                1024,
                self.opts.transformer_layers,
                1,
            )
        
        self.geometry = getattr(self.opts, 'geometry', None)
        if self.geometry is None:
            self.geometry = 'hyperbolic' if self.opts.hyperbolic else 'euclidean'
        self.opts.hyperbolic = self.geometry == 'hyperbolic'
        print('Stage II geometry:', self.geometry)
        if self.geometry == 'hyperbolic':
            hyperbolic_clip_radius_ratio = getattr(
                self.opts, 'hyperbolic_clip_radius_ratio', 0.9)
            if hyperbolic_clip_radius_ratio <= 0:
                hyperbolic_clip_radius_ratio = None
            hyperbolic_tangent_radius_ratio = getattr(
                self.opts, 'hyperbolic_tangent_radius_ratio', 0.7)
            if hyperbolic_tangent_radius_ratio <= 0:
                hyperbolic_tangent_radius_ratio = None
            hyperbolic_normalize_input = not getattr(
                self.opts, 'disable_hyperbolic_input_norm', False)
            self.hyperbolic_linear = MobiusLinear(self.feature_shape,
                                                  self.feature_shape,
                                                  hyperbolic_input=False,
                                                  hyperbolic_bias=False,
                                                  nonlin=None,
                                                  k=self.curvature,
                                                  max_norm_ratio=hyperbolic_clip_radius_ratio,
                                                  max_tangent_radius_ratio=hyperbolic_tangent_radius_ratio,
                                                  normalize_input=hyperbolic_normalize_input,
                                                  )
            self.hyp_mlr = HyperbolicMLR(
                ball_dim=self.feature_shape,
                n_classes=self.num_classes,
                c=self.curvature.abs(),
                point_init_std=getattr(
                    self.opts, 'hyperbolic_anchor_init_std', 1e-2),
                normal_init_gain=getattr(
                    self.opts, 'hyperbolic_normal_init_gain', 5.0),
                max_point_norm_ratio=hyperbolic_clip_radius_ratio)
            self.geometry_head = HyperbolicGeometryHead(
                self.hyperbolic_linear, self.hyp_mlr)
        elif self.geometry == 'euclidean':
            self.linear = LogisticRegression(
                self.feature_shape, self.feature_shape)
            self.euc_mlr = LogisticRegression(
                n_inputs=self.feature_shape, n_outputs=self.num_classes)
            self.geometry_head = EuclideanGeometryHead(self.linear, self.euc_mlr)
        elif self.geometry == 'spherical':
            # Use the new layer instead of LogisticRegression
            self.spherical_linear = SphericalLinear(
                self.feature_shape, self.feature_shape, k=self.spherical_curvature)
            self.spherical_classifier = SphericalPrototypeClassifier(
                ball_dim=self.feature_shape,
                n_classes=self.num_classes,
                k=self.spherical_curvature)
            self.geometry_head = SphericalGeometryHead(
                self.spherical_linear,
                self.spherical_classifier,
                self.spherical_curvature)
        else:
            raise ValueError(f'{self.geometry} is not a valid Stage II geometry')
        # Load weights if needed
        self.load_weights()

    def set_encoder(self, version):
        if self.opts.encoder_type == 'CLIPImageEmbedder':
            encoder = FrozenCLIPImageEmbedder(version)        
        else:
            raise Exception('{} is not a valid encoders'.format(
                self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading HAE from checkpoint: {}'.format(
                self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path,
                              map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            if self.geometry == 'hyperbolic':
                self.hyperbolic_linear.load_state_dict(
                    get_keys(ckpt, 'hyperbolic_linear'), strict=True)
                self.hyp_mlr.load_state_dict(
                    get_keys(ckpt, 'hyp_mlr'), strict=True)
            elif self.geometry == 'euclidean':
                self.linear.load_state_dict(
                    get_keys(ckpt, 'linear'), strict=True)
                self.euc_mlr.load_state_dict(
                    get_keys(ckpt, 'euc_mlr'), strict=True)
            elif self.geometry == 'spherical':
                self.spherical_linear.load_state_dict(
                    get_keys(ckpt, 'spherical_linear'), strict=True)
                self.spherical_classifier.load_state_dict(
                    get_keys(ckpt, 'spherical_classifier'), strict=True)
            else:
                raise NotImplementedError(
                    f'Checkpoint loading for {self.geometry} geometry is not implemented yet.')
            self.transformer_encoder.load_state_dict(
                get_keys(ckpt, 'transformer_encoder'), strict=True)
            self.transformer_decoder.load_state_dict(
                get_keys(ckpt, 'transformer_decoder'), strict=True)
            self.linear_encoder.load_state_dict(
                get_keys(ckpt, 'linear_encoder'), strict=True)
            self.linear_decoder.load_state_dict(
                get_keys(ckpt, 'linear_decoder'), strict=True)
            self.encoder.load_state_dict(
                get_keys(ckpt, 'encoder'), strict=True)
        else:
            print('Initializing hyperediting from scratch')
            ckpt = torch.load(self.opts.psp_checkpoint_path,
                              map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k.replace('.module', '')
                new_state_dict[name] = v
            ckpt['state_dict'] = new_state_dict
            encoder_state = get_keys(ckpt, 'cond_stage_model')
            encoder_state.pop('transformer.vision_model.embeddings.position_ids', None)
            self.encoder.load_state_dict(encoder_state, strict=True)

    def forward(self, x, y=None, batch_size=4, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, input_feature=False):
        if not input_feature:
            if input_code:
                codes = x
            else:
                codes = self.encoder(x)

            ocodes = codes
            feature = self.transformer_encoder(ocodes)
            feature = self.linear_encoder(feature)
            feature_reshape = torch.flatten(feature, start_dim=1)
            
            # The geometry head cleanly handles all manifold mapping internally
            logits, feature_dist, feature_euc = self.geometry_head(feature_reshape)
        else:
            feature_dist = x
            ocodes = None
            logits, feature_dist, feature_euc = self.geometry_head.from_geometry(feature_dist)

        # feature_euc is already flattened back to Euclidean space by geometry_head
        decoder_input = feature_euc.unsqueeze(1)
        feature_euc = self.linear_decoder(decoder_input)
        feature_euc = self.transformer_decoder(feature_euc)
        codes = feature_euc

        return logits, feature_dist, ocodes, feature_euc
    def set_opts(self, opts):
        self.opts = opts
