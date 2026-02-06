# Standard library imports
import os
from math import floor
from typing import Dict, Callable, Any, Optional, List, Tuple, Literal, Union
from dataclasses import dataclass

# Third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.nn import Sequential
import pytorch_lightning as tl
from torch_scatter import scatter
from einops import pack, rearrange, unpack
from skimage.measure import marching_cubes

# Local imports
from hyve.model.layers import EdgeConvCustom
from hyve.extensions import PositionalEncoding
from hyve.preprocess import generate_grid
from hyve.postprocess import normal_consistency, save_to_rectilinear_grid, chamfer_distance
from hyve.extensions import uniform_grid_to_points_linear, points_to_uniform_grid_linear

LossFunctionTypes = Literal['l2', 'scaled_l2', 'cross_entropy', 'deepsdf', 'linear', 'linear_unsigned', 'volume', 'relative', 'relative_unsigned']
ActivationFunctionTypes = Literal['leakyrelu', 'relu', 'tanh', 'siren', 'elu']
OptimizerTypes = Literal['adam', 'adamw', 'sgd']

@dataclass
class EikonalParams:
    use_loss: bool = False
    normalized_normal: float = 5e1
    surface_point: float = 3e3
    surface_normal: float = 1e2
    non_surface: float = 1e2
    non_surface_exp: float = 1e1
    non_surface_unsigned: bool = False
    allow_flipped_surface_normal: bool = False

class Sine(nn.Module):
    def __init__(self, omega_0=30.) -> None:
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class HYVE(tl.LightningModule):
    """
    Hybrid Vertex Encoder (HYVE) model.
    Encodes 3D coordinates into a latent grid representation and decodes them back to Signed Distance Functions (SDF).
    """
    def __init__(self,
                 initial_lr=1e-3, lr_step_size=500, lr_step_gamma=0.5, lr_min: float = 1e-4, lr_scheduler: Literal['plateau','cosine', 'cosine_restarts'] = 'plateau', lr_decay=0,
                 dataset_type="volume", samples_ratio: float=0.2, test_custom_grid_res: Optional[int] = None,
                 test_reconstruction_iso: Optional[float] = 0.0, test_custom_grid_chunk: int = 1000000,
                 test_output_grid=False, loss_function: Union[Tuple[LossFunctionTypes], List[Any]] = ('l2',), deepsdf_clamp_value=0.3,
                 normal_loss=False, normal_loss_factor=1., enc_activation_function: ActivationFunctionTypes='leakyrelu',
                 dec_activation_function: ActivationFunctionTypes='leakyrelu', dec_siren_omega_0: float=30., enc_siren_omega_0: float=0.3, scaled_l2_value=2.,
                 positional_encoding: Optional[Tuple[Optional[float], Optional[float]]]=None, eikonal_loss: Optional[EikonalParams] = None, lr_monitor: str = 'val_loss_relative', optimizer: OptimizerTypes = 'adam',
                 
                 latent_size: int = 64,
                 decoder_type: Literal['modulated', 'fc', 'siren', 'relu'] = 'fc',
                 decoder_batch_norm: bool = False,
                 grid_resolutions: List[int] = [4, 8, 16, 32],
                 grid_conv_skip_connection: bool = True,
                 encoder_interpolation: bool = True,
                 encoder_point_to_grid: Literal['max', 'mean', 'pic', 'pic_masked'] = 'max',
                 encoder_pic_masked_value: Union[float,Literal['scheduled']] = 0.5,
                 encoder_pic_masked_keep_n: Optional[int] = None,
                 encoder_use_gnn: bool = True,
                 encoder_gnn_conv: Literal['edgeconv','pointconv'] = 'edgeconv',
                 encoder_gnn_conv_aggr: Literal['mean', 'max'] = 'mean',
                 encoder_use_cnn: bool = True,
                 encoder_cnn_batch_norm: bool = False,
                 encoder_cnn_depth: int = 1,
                 encoder_cnn_kernel_size: int = 2,
                 encoder_cnn_kernel_stride: int = 1,
                 encoder_cnn_mode: Literal['bottleneck', 'even', 'decreasing'] = 'bottleneck',
                 encoder_use_normals: bool = False,
                 encoder_translation_invariant: bool = False,
                 encoder_normal_handling: Literal['plain','oriented','flip_ok'] = 'plain',
                 encoder_conv_bias: bool = True,
                 use_multiple_lod: bool = False,
                 **kwargs):

        super(HYVE, self).__init__()

        # --- Initialization logic from Baseline ---
        self.enc_activation_function_dict: Dict[ActivationFunctionTypes, Callable[..., nn.Module]] = {"leakyrelu": lambda: nn.LeakyReLU(0.2), "relu": nn.ReLU, "tanh": nn.Tanh, "siren": lambda: Sine(omega_0=enc_siren_omega_0)}
        self.dec_activation_function_dict: Dict[ActivationFunctionTypes, Callable[..., nn.Module]] = {"leakyrelu": lambda: nn.LeakyReLU(0.2), "relu": nn.ReLU, "tanh": nn.Tanh, "siren": lambda: Sine(omega_0=dec_siren_omega_0)}
        self.optimizer_dict: Dict[OptimizerTypes, Callable[..., torch.optim.Optimizer]] = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}

        # Handle potential override of activation functions by inheritance logic if needed, but here simple assignment
        self.enc_activation_function = self.enc_activation_function_dict[enc_activation_function]
        self.dec_activation_function = self.dec_activation_function_dict[dec_activation_function]

        self.test_recon_dir: str = ""

        # --- Initialization logic from BaselineLatentGridInvertedInterpolated ---
        self.latent_size: int = latent_size
        self.decoder_type = decoder_type
        self.decoder_batch_norm = decoder_batch_norm
        self.grid_resolutions = grid_resolutions
        self.use_multiple_lod = use_multiple_lod
        self.grid_conv_skip_connection = grid_conv_skip_connection
        self.encoder_interpolation = encoder_interpolation
        self.encoder_point_to_grid = encoder_point_to_grid
        self.encoder_pic_masked_value = encoder_pic_masked_value
        self.encoder_pic_masked_keep_n = encoder_pic_masked_keep_n
        self.encoder_use_gnn = encoder_use_gnn
        self.encoder_gnn_conv = encoder_gnn_conv
        self.encoder_gnn_conv_aggr = encoder_gnn_conv_aggr
        self.encoder_use_cnn = encoder_use_cnn
        self.encoder_cnn_batch_norm = encoder_cnn_batch_norm
        self.encoder_cnn_depth = encoder_cnn_depth
        self.encoder_cnn_kernel_size = encoder_cnn_kernel_size
        self.encoder_cnn_kernel_stride = encoder_cnn_kernel_stride
        self.encoder_cnn_mode = encoder_cnn_mode
        self.encoder_use_normals = encoder_use_normals
        self.encoder_translation_invariant = encoder_translation_invariant
        self.encoder_normal_handling: Literal['plain','oriented','flip_ok'] = encoder_normal_handling if encoder_use_normals else 'plain'
        self.encoder_conv_bias = encoder_conv_bias

        # Save hyperparameters
        # We need to manually construct the dictionary because inspect.getfullargspec might act weird with combined args
        # But most importantly we want to save everything.
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

        self._setup_architecture()
        self._init_weights()

        self.loss_fcts: Dict[LossFunctionTypes, Callable] = {
            "l2": nn.MSELoss(),
            "scaled_l2": self.scaled_l2_loss,
            "cross_entropy": self.cross_entropy,
            "deepsdf": self.deepsdf_loss,
            "linear": self.linear_loss,
            "linear_unsigned": self.linear_loss_unsigned,
            "volume": self.volume_loss,
            "relative": self.relative_l1_loss,
            "relative_unsigned": self.relative_l1_loss_unsigned
        }

        self.loss = self.construct_loss()

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.test_loss = nn.MSELoss(reduction='mean')
        self.validation_step_output = []
        self.test_step_output = []
        self.test_step_reconstructions = []

    def _setup_architecture(self) -> None:
        input_len = 3 if not self.hparams['encoder_use_normals'] else 6
        ls = self.latent_size
        half_ls = floor(ls/2.)
        quarter_ls = floor(ls/4.)

        # Positional encoding for attn
        # self.attn_pe = BarycentricPositionalEncoding3D(ls) if 'bary3d' in self.encoder_attn_pe else None
        self.attn_pe = None

        # First layer
        first_layer = []
        if self.hparams['positional_encoding'] is not None and self.hparams['positional_encoding'][0] is not None:
            if self.encoder_use_normals:
                first_layer.append((lambda x: unpack(x, [(3,),(3,)], "x *"), 'x -> x1, x2'))
                first_layer.append((PositionalEncoding(3, half_ls, self.hparams['positional_encoding'][0]), 'x1 -> x1',))
                # Dont do positional encoding for normals
                # first_layer.append((PositionalEncoding(3, quarter_ls, self.hparams['positional_encoding'][0]), 'x2 -> x2',))
                first_layer.append((lambda x1, x2: pack([x1, x2], "x *"), 'x1, x2 -> x, shapes'))
                input_len = 2*half_ls + 3
            else:
                first_layer.append((PositionalEncoding(input_len, half_ls, self.hparams['positional_encoding'][0]), 'x -> x',))
                input_len = 2*half_ls
        
        if self.hparams['encoder_use_gnn']:
            first_layer.append((EdgeConvCustom(
                input_len, 
                ls, 
                self.enc_activation_function,
                translation_invariant=self.encoder_translation_invariant,
                normal_handling=self.encoder_normal_handling,
                aggr="mean"), "x, edge_index -> x"))

        self.enc_conv0 = Sequential('x, edge_index', first_layer)

        # Hidden layers
        for i in range(len(self.grid_resolutions)):
            if self.hparams['encoder_use_cnn']:
                # Use batch norm after pooling?
                grid_conv_layers = []
                # grid_conv_layers += [nn.MaxPool3d(2)] if self.encoder_point_to_grid == "relu_pic_max" else []
                grid_conv_layers += [nn.BatchNorm3d(ls)] if self.encoder_cnn_batch_norm else []
                if self.encoder_cnn_mode == 'bottleneck':
                    for j in range(self.encoder_cnn_depth):
                        grid_conv_layers += [
                            nn.Conv3d(2**j * ls, 2**(j+1)*ls, (self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size), bias=self.encoder_conv_bias, stride=self.encoder_cnn_kernel_stride),
                            self.enc_activation_function(), 
                        ]
                    # grid_conv_layers += [nn.BatchNorm3d(2**(j+1)*ls)] if self.encoder_cnn_batch_norm else []
                    for j in reversed(range(self.encoder_cnn_depth)):
                        grid_conv_layers += [
                            nn.ConvTranspose3d(2**(j+1) * ls, 2**j * ls, (self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size), bias=self.encoder_conv_bias, stride=self.encoder_cnn_kernel_stride),
                            self.enc_activation_function(), 
                        ]
                # Even should be used with an odd kernel size
                elif self.encoder_cnn_mode == 'even':
                    for j in range(self.encoder_cnn_depth):
                        grid_conv_layers += [
                            nn.Conv3d(ls, ls, (self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size), bias=self.encoder_conv_bias, padding=self.encoder_cnn_kernel_size//2),
                            self.enc_activation_function(), 
                        ]
                # Only down-conv
                elif self.encoder_cnn_mode == 'decreasing':
                    for j in range(self.encoder_cnn_depth):
                        grid_conv_layers += [
                            nn.Conv3d(ls, ls, (self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size, self.encoder_cnn_kernel_size), bias=self.encoder_conv_bias),
                            self.enc_activation_function(), 
                        ]

                # Build layer
                setattr(self, f"grid_conv{i+1}",nn.Sequential(*grid_conv_layers))

            if self.hparams['encoder_use_gnn']:
                setattr(self, f"pre_enc_conv{i+1}", nn.Sequential(nn.Linear(ls, ls),
                                                                self.enc_activation_function()))
                if self.encoder_gnn_conv == 'edgeconv':
                    setattr(self, f"enc_conv{i+1}", tg.nn.EdgeConv(nn.Sequential( 
                                                                nn.Linear(2*ls, ls), 
                                                                self.enc_activation_function()), aggr=self.encoder_gnn_conv_aggr))
                elif self.encoder_gnn_conv == 'pointconv':
                    setattr(self, f"enc_conv{i+1}", Sequential('x, edge_conv', [(nn.Linear(ls, ls), "x->x"),
                                                                (self.enc_activation_function(), "x->x")]))
                else:
                    print(f"Wrong gnn conv mode selected {self.encoder_gnn_conv}.")


        pos_size = 3
        if self.hparams['positional_encoding'] is not None and self.hparams['positional_encoding'][1] is not None:
            pos_size = ls
            self.pre_dec = PositionalEncoding(3, half_ls, self.hparams['positional_encoding'][1])

        # Decoder layer
        from .decoders import FCDecoder
        n_lod = len(self.grid_resolutions)+1 if self.use_multiple_lod else 1
        for i in range(n_lod):
            if self.decoder_batch_norm:
                setattr(self, f"dec_bn_{i}", nn.BatchNorm1d(ls))
            match self.decoder_type:
                case 'modulated':
                    setattr(self, f"dec_{i}", FCDecoder(ls, ls, 2, modulation_activation=nn.LeakyReLU(0.02)))
                case 'siren':
                    setattr(self, f"dec_{i}", FCDecoder(ls, ls, 2, modulation_activation=None))
                case 'relu':
                    setattr(self, f"dec_{i}", FCDecoder(ls, ls, 2, activation=nn.ReLU(), modulation_activation=None))
                case 'fc':
                    setattr(self, f"dec_{i}", nn.Sequential(
                        nn.Conv1d(ls+pos_size, half_ls, 1), self.dec_activation_function(), 
                        nn.Conv1d(half_ls, quarter_ls, 1), self.dec_activation_function(), 
                        nn.Conv1d(quarter_ls, 1, 1)))
                case _:
                    print(f"Unknown decoder type {self.decoder_type}. Using default 'fc'.")
                    setattr(self, f"dec_{i}", nn.Sequential(
                        nn.Conv1d(ls+pos_size, half_ls, 1), self.dec_activation_function(), 
                        nn.Conv1d(half_ls, quarter_ls, 1), self.dec_activation_function(), 
                        nn.Conv1d(quarter_ls, 1, 1)))

    def _init_weights(self):
        # Set encoder weights
        if self.hparams['enc_activation_function'] == 'siren':
            with torch.no_grad():
                enc_omega_0 = self.hparams['enc_siren_omega_0']
                for mod in self.modules():
                    if isinstance(mod, tg.nn.EdgeConv):
                        mod.nn[0].weight.uniform_(-np.sqrt(6./mod.nn[0].in_features) / enc_omega_0, 
                                                np.sqrt(6./mod.nn[0].in_features) / enc_omega_0)
                    elif isinstance(mod, nn.Sequential):
                        if isinstance(mod[0], nn.Linear):
                            mod[0].weight.uniform_(-np.sqrt(6./mod[0].in_features) / enc_omega_0, 
                                                np.sqrt(6./mod[0].in_features) / enc_omega_0)
                # Special init for first layers omitted as enc_conv structures differ from baseline

        if self.hparams['dec_activation_function'] == 'siren':
            with torch.no_grad():
                dec_omega_0 = self.hparams['dec_siren_omega_0']
                for mod in self.modules():
                    if isinstance(mod, nn.Sequential):
                        if isinstance(mod[0], nn.Conv1d):
                            mod[0].weight.uniform_(-np.sqrt(6./mod[0].in_channels) / dec_omega_0, 
                                                np.sqrt(6./mod[0].in_channels) / dec_omega_0)
                # Special init for first layers omitted


    def on_test_epoch_start(self) -> None:
        # Construct the reconstruction directory before anything gets tested
        if self.logger.log_dir:
            self.test_recon_dir = os.path.join(self.logger.log_dir, "reconstructions")
            if not os.path.isdir(self.test_recon_dir):
                os.makedirs(self.test_recon_dir)
        if self.hparams['test_custom_grid_res'] is not None:
            self.custom_grid = {}
            self.custom_grid['sdf_pos'] = generate_grid(self.hparams['test_custom_grid_res']).to(self.device).float()
            n_batch = self.hparams['test_custom_grid_chunk']
            self.custom_grid['n_chunks'] = max(len(self.custom_grid['sdf_pos'])//n_batch, 1)
            self.custom_grid['samples_batch'] = torch.ones((2*n_batch), dtype=torch.int64, device=self.device)

    def construct_loss(self) -> Dict[str, Callable]:
        """
        Get the combination of admissable loss functions
        :return: loss function
        """
        loss: Dict[str, Callable] = {}
        for case in self.hparams['loss_function']:
            if case in self.loss_fcts:
                loss[case] = self.loss_fcts[case]
            else:  # Output a warning
                # loss['l2'] = nn.MSELoss(reduction='sum')
                raise RuntimeWarning(f"Specified loss function '{case}' not found in list of possible choices."
                                     f"Choices are 'l2', 'deepsdf', 'linear', 'volume', 'relative'.")

        return loss

    def evaluate_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate all loss functions contained in the array of losses by stacking the values and summing
        :param pred: predicted sd values
        :param target: target sd values
        :return: loss value
        """
        # Stack loss values and sum up
        loss = {name: self.loss[name](pred, target) for name in self.loss}
        return loss

    def scaled_l2_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = self.hparams['scaled_l2_value']
        weight = scale*torch.exp(-((target * scale)**2))
        loss = (weight*nn.MSELoss(reduction='none')(pred, target)).mean()
        return loss

    def cross_entropy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Assign hard class labels to inside and outside
        t = torch.sgn(target)

        # Ignore the values which are 0 exactly by setting the weight of all others to 1
        w = torch.abs(t)

        # Target -1 should be 0 and target 1 is already 1
        # Initial zero values are ignored using the weight tensor
        t[t < 0] = 0

        # Include sigmoid of pred in loss function. Might play around with pos weight in the future to increase
        # importance of inside samples
        loss = torch.nn.BCEWithLogitsLoss(w, pos_weight=None)(pred, t)
        return loss

    def relative_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the relative l2 loss (target - pred)/(target + epsilon) where epsilon is 1e-4 which can approximately
        be written as 1 - pred/(target + eps)
        """
        eps = 1e-4
        # This is the squared variant
        # pred_div_target = pred/(target + eps)
        # loss = nn.MSELoss()(torch.ones_like(pred_div_target), pred_div_target)
        # MAPE mean absolute percentage error variant
        loss = torch.abs((target - pred)/(target + eps)).mean()
        return loss

    def relative_l1_loss_unsigned(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the relative l2 loss (target - pred)/(target + epsilon) where epsilon is 1e-4 which can approximately
        be written as 1 - pred/(target + eps)
        """
        eps = 1e-4
        # This is the squared variant
        # pred_div_target = pred/(target + eps)
        # loss = nn.MSELoss()(torch.ones_like(pred_div_target), pred_div_target)
        # MAPE mean absolute percentage error variant
        loss = torch.abs((target.abs() - pred.abs())/(target.abs() + eps)).mean()
        return loss

    def deepsdf_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Loss function implemented in the DeepSDF paper which first clamps both the predicted and target sdf value
        and only computed the L1 loss using the clamped values.
        :param pred: predicted sd values
        :param target: target sd values
        :return: loss value
        """
        clamp_value = self.hparams["deepsdf_clamp_value"]
        pred = torch.clamp(pred, min=-clamp_value, max=clamp_value)
        target = torch.clamp(target, min=-clamp_value, max=clamp_value)
        res = self.l1_loss(pred, target)

        return res

    def linear_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Loss function which more heavily penalizes errors for smaller target sdf values. The idea is, that this error
        will enforce better reconstructive properties
        :param pred: predicted sd values
        :param target: target sd values
        :return: loss value
        """
        loss = nn.L1Loss(reduction='none')
        res = 3000 * loss(pred, target).mean()

        return res
    
    def linear_loss_unsigned(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Linear loss that compares pred to the unsigned distance in target
        :param pred: predicted sd values
        :param target: target sd values
        :return: loss value
        """
        loss = nn.L1Loss(reduction='none')
        res = 3000 * loss(pred, target.abs()).mean()

        return res
    
    def volume_loss(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Loss which enforces that the number of negatively signed distances must be equal. Under the assumption that
        each point can be multiplied by some constant volume, this would result in a volume loss.
        :param pred: predicted sd values
        :param target: target sd values
        :return: loss value
        """
        # Number of points inside the volume for prediction
        pred_volume = (pred < 0).sum(dim=-1)

        # Number of points inside the volume for target
        target_volume = (target < 0).sum(dim=-1)

        # Compute the absolute deviation sum up and return
        loss = (torch.abs(pred_volume - target_volume).type_as(pred) / target_volume).mean()
        return loss

    def compute_normal(self, pred_sd: torch.Tensor, input_pos: torch.Tensor, create_graph: bool=False, retain_graph: bool=False) -> torch.Tensor:
        if create_graph:
            pred_normal = torch.autograd.grad(pred_sd, input_pos,
                                              grad_outputs=torch.ones_like(pred_sd),
                                              create_graph=create_graph)[0]
        else:
            pred_sd.backward(torch.ones_like(pred_sd), retain_graph=retain_graph)
            pred_normal = input_pos.grad

        return pred_normal

    def normal_loss(self, pred_normal: torch.Tensor, target_normal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the deviation from the normal vector in the componentwise l2 norm. The normal vector of the network is
        computed using backpropagation d(pred_sd)/d(input_pos) by computing the derivative of the signed distance w.r.t.
        the input position.
        :param pred_normal: Predicted normal
        :param target_normal: Target normal vector
        """
        # Add small value which will disappear in the error, but makes sure that there are no divisions by zero
        pred_norm = torch.norm(pred_normal, dim=-1, keepdim=True) + 1e-8
        target_norm = torch.norm(target_normal, dim=-1, keepdim=True) + 1e-8

        dot_pred_target = torch.sum((pred_normal / pred_norm.expand_as(pred_normal)) *
                                    (target_normal / target_norm.expand_as(target_normal)), dim=-1)

        return {
            # "normal_abs": self.hparams["normal_loss_factor"]*nn.MSELoss()(pred_norm, target_norm),  # Absolute value of the normal should be the same
            "normal_dir": self.hparams["normal_loss_factor"] * nn.MSELoss()(torch.ones_like(dot_pred_target),
                                                                            dot_pred_target)}  # Direction of the normal should be the same

    def eikonal_loss(self, pred: torch.Tensor, target: torch.Tensor, pred_normal: torch.Tensor, target_normal: torch.Tensor) -> Dict[str, torch.Tensor]:
        # eik_param = cast(EikonalParams, self.hparams['eikonal_loss'])
        eik_param = EikonalParams(**self.hparams['eikonal_loss'])
        surface_points_idx = torch.isclose(target, torch.zeros_like(target))

        # Normalize normals
        normalized_normal_loss = torch.abs(pred_normal.norm(dim=-1) - 1.)

        # Surface points
        surface_point_loss = torch.where(surface_points_idx, torch.abs(pred), torch.zeros_like(pred))
        if eik_param.allow_flipped_surface_normal:
            surface_normal_loss = 1 - F.cosine_similarity(pred_normal[surface_points_idx], target_normal[surface_points_idx]).abs()
        else: 
            surface_normal_loss = 1 - F.cosine_similarity(pred_normal[surface_points_idx], target_normal[surface_points_idx])

        # Penalize zero-values away from the surface
        if not eik_param.non_surface_unsigned:
            non_surface_loss = torch.where(surface_points_idx, torch.zeros_like(pred), torch.exp(-eik_param.non_surface_exp * torch.abs(pred)))
        # If unsigned distance then the abs in pred goes away
        else:
            non_surface_loss = torch.where(surface_points_idx, torch.zeros_like(pred), torch.exp(-eik_param.non_surface_exp * pred))

        # Factors found from "original" implementation of SIREN
        return {'siren_normalized_normal': normalized_normal_loss.mean() * eik_param.normalized_normal,
                'siren_surface_point': surface_point_loss.mean() * eik_param.surface_point,
                'siren_surface_normal': surface_normal_loss.mean() * eik_param.surface_normal,
                'siren_non_surface': non_surface_loss.mean() * eik_param.non_surface}

    def encode(self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> List[ torch.Tensor]:
        """
        Encode point cloud into a list of latent grids at different resolutions.
        """
        x = self.enc_conv0(pos, edge_index)
        cluster = batch
        conv = tg.nn.global_max_pool(x, batch) # type: ignore
        grids = [conv]

        for i in range(len(self.grid_resolutions)):
            if i > 0:
                grid_res = self.grid_resolutions[i-1] if self.encoder_cnn_mode != 'decreasing' else self.grid_resolutions[i-1] - (self.encoder_cnn_kernel_size - 1) * self.encoder_cnn_depth

            interpolated_features = conv[cluster] if i == 0 or not self.encoder_interpolation else uniform_grid_to_points_linear(conv, pos[:,:3], batch, grid_res)

            if self.encoder_use_gnn:
                x = getattr(self, f"pre_enc_conv{i+1}")(x + interpolated_features if self.grid_conv_skip_connection else interpolated_features)
                x = getattr(self, f"enc_conv{i+1}")(x, edge_index)
            cluster, max_pool, grid, conv = self.grid_pool_conv(batch, pos, x, self.grid_resolutions[i], getattr(self, f"grid_conv{i+1}", None))  # Return None as default because the function checks for it
            grids += [conv]

        return grids

    def grid_pool_conv(self, batch, pos, x, ngrid, conv):
        """
        Pool in a larger grid so that the central nodes of the grid define the unit cube. In other words, the
        pooling grid has the unit grid embedded
        """
        # Max pooling
        grid_delta = 2. / (ngrid - 1)
        cluster = (torch.matmul(torch.floor((pos[:,:3] - (-1 - grid_delta/2.)) / grid_delta),
                                torch.tensor([1, ngrid, ngrid * ngrid], dtype=torch.float32,
                                             device=self.device)) + batch * ngrid * ngrid * ngrid).type(torch.int64)

        if not 'pic' in self.encoder_point_to_grid:
            # Pooling max or mean
            max_pool = x.new_zeros((ngrid * ngrid * ngrid * (batch.max().item() + 1), self.latent_size))
            max_pool = scatter(x, cluster, dim=0, out=max_pool, reduce=self.encoder_point_to_grid)
        elif self.encoder_point_to_grid == 'pic':
            # Interpolation
            max_pool, weights = points_to_uniform_grid_linear(x, pos[:,:3], batch, ngrid) #, log_grad_fn=self.log_dict)
        elif self.encoder_point_to_grid == 'pic_masked':
            mask = torch.rand((pos.shape[0], 8), device=self.device) 
            # Make the smallest value in each row equal to 0, so that at least one value will always be used
            if self.encoder_pic_masked_keep_n is not None:
                mask[torch.arange(pos.shape[0])[:, None], mask.topk(dim=1,k=self.encoder_pic_masked_keep_n,largest=False).indices] = 0
            mask = mask <= self.encoder_pic_masked_value if self.training else mask <= 1 # Keep all when not training
            max_pool, weights = points_to_uniform_grid_linear(x, pos[:,:3], batch, ngrid, mask=mask)
        
        grid = max_pool.view(-1, ngrid * ngrid * ngrid, self.latent_size).transpose(1, 2).view(-1, self.latent_size, ngrid, ngrid, ngrid)
        if conv is not None:
            conv_ret = rearrange(conv(grid), "B C X Y Z -> (B X Y Z) C")
        else:
            conv_ret = max_pool
        return cluster, max_pool, grid, conv_ret

    def decode(self, samples: torch.Tensor, samples_batch: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Decode latent grids into SDF values at query sample locations.
        """
        samples_flat = samples.view(-1, 3)
        cluster = [x[0][samples_batch]]
        for i, res in enumerate(self.grid_resolutions): 
            grid_res = res if self.encoder_cnn_mode != 'decreasing' else res - (self.encoder_cnn_kernel_size - 1) * self.encoder_cnn_depth
            cluster += [uniform_grid_to_points_linear(x[i+1], samples_flat, samples_batch, grid_res)]

        lod_sums = [sum(cluster)]

        if self.decoder_batch_norm:
            B, N, _ = samples.shape
            lod_sums = [getattr(self, f"dec_bn_{i}")(item.view(B, N, -1).transpose(1,2)).transpose(1,2).reshape(B*N, -1) for i, item in enumerate(lod_sums)]

        if hasattr(self, 'pre_dec'):
            lod_x = [torch.cat([item, self.pre_dec(samples_flat)], dim=-1).view(samples.shape[0], samples.shape[1], -1) for item in
                    lod_sums]
        else:
            lod_x = [torch.cat([item, samples_flat], dim=1).view(samples.shape[0], samples.shape[1], -1) for item in
                    lod_sums]
        lod_y = []
        for i, item in enumerate(lod_x):
            item = item.transpose(1, 2)
            lod_y.append(getattr(self, "dec_"+str(i))(item))
        return lod_y

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, samples: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass: Encode point cloud and decode at sample locations.
        """
        x = self.encode(pos, edge_index, batch)

        samples_batch = torch.cat([torch.ones(s.shape[0], device=self.device, dtype=torch.int64) * i for i, s in enumerate(samples)])
        lod_y = self.decode(samples, samples_batch, x)
        return lod_y, x, samples_batch, torch.tensor([0])


    def training_step(self, batch: Dict[str, Data], batch_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        data = batch[self.hparams["dataset_type"]]

        # Choose a specific percentage of sdf samples in each training iteration
        ny = data.y.size(1)
        nc = floor(self.hparams["samples_ratio"] * ny)
        sdf_samples = data.y[:, torch.randperm(ny)[:nc], :]
        sdf_pos = sdf_samples[:, :, 0:3]
        sdf_sd = sdf_samples[:, :, 3]
        sdf_normal = sdf_samples[:, :, 4:7]
        if self.hparams["normal_loss"] or self.hparams["eikonal_loss"]:
            sdf_pos.requires_grad_(True)


        # Forward propagation
        pos = data.pos
        if self.encoder_use_normals:
            pos, _ = pack([pos, data.x], "n *")
        out, x, samples_batch, z = self(pos, data.edge_index, data.batch, sdf_pos)
        loss_list_lod = [self.evaluate_loss(item.squeeze(), sdf_samples[:, :, 3].squeeze()) for item in out]
        loss_dict_lod = {key + f"_lod_{i}": item[key] for key in loss_list_lod[0] for i, item in
                         enumerate(loss_list_lod)}
        loss_dict = {key: sum([item[key] for item in loss_list_lod]) / len(out) for key in loss_list_lod[0]}

        if self.hparams["normal_loss"] or self.hparams['eikonal_loss']['use_loss']:
            pred_normal = [self.compute_normal(item.squeeze(), sdf_pos, create_graph=True) for item in out]
            if self.hparams['normal_loss']:
                normal_loss = [self.normal_loss(item, sdf_normal) for item in pred_normal]
                normal_loss_dict_lod = {key + f"_lod_{i}": item[key] for key in normal_loss[0] for i, item in
                                        enumerate(normal_loss)}
                normal_loss = {key: sum([item[key] for item in normal_loss]) / len(normal_loss) for key in normal_loss[0]}
                loss_dict.update(normal_loss)
                loss_dict_lod.update(normal_loss_dict_lod)

            if self.hparams['eikonal_loss']['use_loss']:
                eikonal_loss = [self.eikonal_loss(sd.squeeze(), sdf_sd.squeeze(), n, sdf_normal) for (sd, n) in zip(out, pred_normal)]
                eikonal_loss_dict_lod = {key + f"_lod_{i}": item[key] for key in eikonal_loss[0] for i, item in
                                        enumerate(eikonal_loss)}
                eikonal_loss = {key: sum([item[key] for item in eikonal_loss]) / len(eikonal_loss) for key in eikonal_loss[0]}
                loss_dict.update(eikonal_loss)
                loss_dict_lod.update(eikonal_loss_dict_lod)

        # Compute the total loss and add to logs
        loss = torch.stack([loss_dict[name] for name in loss_dict]).sum()

        for name in loss_dict_lod:
            self.log(f"train_loss_{name}", loss_dict_lod[name], sync_dist=True)

        for name in loss_dict:
            self.log(f"train_loss_{name}", loss_dict[name], sync_dist=True)
            self.log(f"{name}", loss_dict[name], prog_bar=True, logger=False, sync_dist=True)
        self.log("train_loss_total", loss, sync_dist=True)

        # Log increases of 20 in epoch number for checkpointing purposes
        self.log("epoch_floordiv_20", float(0 if self.trainer.current_epoch % 20 else self.trainer.current_epoch // 20))

        return {'loss': loss, 'latent_space': z.detach(), 'id': batch["id"].detach()}

    def validation_step(self, batch, batch_idx):
        data = batch[self.hparams["dataset_type"]]

        sdf_pos = data.y[:, :, 0:3]
        sdf_sd = data.y[:, :, 3]
        sdf_normal = data.y[:, :, 4:7]
        # torch.set_grad_enabled(True)
        # sdf_pos.requires_grad = True

        pos = data.pos
        if self.encoder_use_normals:
            pos, _ = pack([pos, data.x], "n *")
        out, x, samples_batch, z = self(pos, data.edge_index, data.batch, sdf_pos)
        # loss_dict = self.evaluate_loss(out.squeeze(), sdf_sd.squeeze())
        # Always evaluate all losses in validation for better comparison
        loss_list_lod = [{loss: self.loss_fcts[loss](item.squeeze(), sdf_sd.squeeze()) for loss in self.loss_fcts} for
                         item in out]
        loss_dict_lod = {key + f"_lod_{i}": item[key] for key in loss_list_lod[0] for i, item in
                         enumerate(loss_list_lod)}
        loss_dict = {key: sum([item[key] for item in loss_list_lod]) / len(out) for key in loss_list_lod[0]}

        # Rename keys for validation
        loss_dict_lod.update(loss_dict)
        loss_dict_val = {"val_loss_{}".format(name): loss_dict_lod[name] for name in loss_dict_lod}

        # Compute the total loss and add to logs
        loss = torch.stack([loss_dict[name] for name in loss_dict]).sum()
        output = {'val_loss_total': loss, **loss_dict_val, 'latent_space': z.detach(), 'id': batch["id"].detach()}
        self.validation_step_output.append(output)

        return output
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_output

        # Turn list of dicts into dict of lists
        val_losses = {name: torch.stack([out[name] for out in outputs]).mean() for name in outputs[0] if
                      name.startswith('val_loss')}
        for name in val_losses:
            self.log(f"{name}", val_losses[name], sync_dist=True)

        self.validation_step_output.clear()

    def test_step(self, batch, batch_idx):
        import meshio
        data = batch[self.hparams["dataset_type"]]

        chamfer_dist: Optional[float] = None
        normal_cons: Optional[float] = None
        surface_point_sdf: Optional[float] = None
        if self.hparams['test_custom_grid_res'] is not None:
            grid_res = self.hparams['test_custom_grid_res']
            grid_delta = 2./(grid_res-1)
            with torch.inference_mode():
                sdf_pos = self.custom_grid['sdf_pos']
                samples_batch = self.custom_grid['samples_batch']

                start_events = [torch.cuda.Event(enable_timing=True) for _ in sdf_pos.chunk(self.custom_grid['n_chunks'])] # + [torch.cuda.Event(enable_timing=True)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in sdf_pos.chunk(self.custom_grid['n_chunks'])] # + [torch.cuda.Event(enable_timing=True)]
                # start_events[0].record()
                # x = self.encode(data.pos, data.edge_index, data.batch)
                # end_events[0].record()
                # Check what the predicted value is on the surface, where it should be zero
                if data.batch.max() == 0:
                    print(f"Running ID: {batch['id']}")
                    pos = data.pos
                    if self.encoder_use_normals:
                        pos, _ = pack([pos, data.x], "n *")

                    surface_point_sdf = self.forward(pos, data.edge_index, data.batch, data.pos.unsqueeze(0))[0][0].abs().mean().item()

                chamfer_dist = 0
                normal_cons = 0 if hasattr(data, "x") else None
                for i, idx in enumerate(batch["id"]):
                    sdf_vals = []
                    for j, grid_pos in enumerate(sdf_pos.chunk(self.custom_grid['n_chunks'])):
                        start_events[j].record()
                        x = self.encode(pos, data.edge_index, data.batch)
                        lod_y = self.decode(grid_pos[None, ...], samples_batch[:len(grid_pos)]*i, x)
                        end_events[j].record()
                        sdf_vals += [lod_y[-1].detach().squeeze().cpu().numpy()]
                    
                    torch.cuda.synchronize()
                    time = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)]) /1000.
                    print(f"Cuda Time: {time}")
                    # Output rectilinear grid
                    if self.hparams['test_output_grid']:
                        grid_output_file = os.path.join(self.test_recon_dir, "{}_grid_{}".format(grid_res, idx))
                        save_to_rectilinear_grid(sdf_pos.detach().squeeze().cpu().numpy(), 
                                                    grid_res, 
                                                    grid_output_file, 
                                                    attr_dict={'pred': pack(sdf_vals, "*")[0]})

                    # Run marching cubes and compute chamfer distance
                    sdf_vals = rearrange(pack(sdf_vals, "*")[0], "(x y z) -> x y z", x=grid_res, y=grid_res, z=grid_res)

                    if self.hparams['test_reconstruction_iso'] is not None:
                        verts, faces, normals, values = marching_cubes(sdf_vals, self.hparams['test_reconstruction_iso'], spacing=[grid_delta, grid_delta, grid_delta])

                        mesh_output_file = os.path.join(self.test_recon_dir, "{}_sf_{}.vtu".format(grid_res, idx))
                        meshio.write_points_cells(mesh_output_file, verts-1, [('triangle', faces)], point_data={'normals': -normals})  # Have to invert the normals to point outward
                        chamfer_dist += chamfer_distance(verts-1, data.pos[data.batch==i].cpu())
                        if normal_cons is not None:
                            normal_cons += normal_consistency(verts-1, data.pos[data.batch==i].cpu(), normals, data.x[data.batch==i].cpu().numpy())
                chamfer_dist /= len(batch['id'])
                normal_cons /= len(batch['id'])

        sdf_pos = data.y[:, :, 0:3]
        sdf_sd = data.y[:, :, 3]
        sdf_normal = data.y[:, :, 4:7]

        # Enable gradients for normal computation
        # torch.set_grad_enabled(True)
        # sdf_pos.requires_grad = True

        pos = data.pos
        if self.encoder_use_normals:
            pos, _ = pack([pos, data.x], "n *")
        out, x, samples_batch, z  = self(pos, data.edge_index, data.batch, sdf_pos)
        # loss_dict = self.evaluate_loss(out.squeeze(), data.y[:, :, 3].squeeze())
        loss_list_lod = [{loss: self.loss_fcts[loss](item.squeeze(), sdf_sd.squeeze()).item() for loss in self.loss_fcts} for
                         item in out]
        loss_dict_lod = {key + f"_lod_{i}": item[key] for key in loss_list_lod[0] for i, item in
                         enumerate(loss_list_lod)}
        loss_dict = {key: sum([item[key] for item in loss_list_lod]) / len(out) for key in loss_list_lod[0]}
        pred_normal = torch.zeros((len(out), *out[0].shape, 3))

        if chamfer_dist is not None:
            loss_dict['chamfer_loss'] = chamfer_dist
        if normal_cons is not None:
            loss_dict['normal_consistency'] = normal_cons
        if surface_point_sdf is not None:
            loss_dict['surface_point_loss'] = surface_point_sdf

        loss_dict.update(loss_dict_lod)

        for i, idx in enumerate(batch["id"]):
            for j, _ in enumerate(out):
                (pos, pred, target, pred_n, target_n) = (sdf_pos[i].detach().squeeze().cpu().numpy(),
                                                         out[j][i].detach().squeeze().cpu().numpy(),
                                                         # Output only highest LOD
                                                         sdf_sd[i].detach().squeeze().cpu().numpy(),
                                                         pred_normal[j][i].detach().squeeze().cpu().numpy(),
                                                         # Output only highest LOD
                                                         sdf_normal[i].detach().squeeze().cpu().numpy())
                # predicted_mesh_file = os.path.join(self.test_recon_dir, "{}_lod_{}.vtu".format(j, idx))
                grid_size = np.cbrt(len(pos)+1).astype(np.int32)
            self.test_step_reconstructions.append([idx, pos, pred, target, pred_n, target_n, grid_size])

        # Return the loss per index to be printed to a json file
        self.test_step_output.append(loss_dict)
        return loss_dict

    def configure_optimizers(self):
        optimizer = self.optimizer_dict[self.hparams["optimizer"]](self.parameters(), lr=self.hparams["initial_lr"], weight_decay=self.hparams['lr_decay'])

        if self.hparams["lr_scheduler"] == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, min_lr=1e-6, patience=self.hparams["lr_step_size"])
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': self.hparams['lr_monitor'], 'name': "plateau"}}
        elif self.hparams["lr_scheduler"] == "cosine":
            lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["lr_step_size"], eta_min=self.hparams["lr_min"])
            # lambda returns 1 as it is a multiplicative factor
            lr_scheduler_2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.hparams["lr_min"] / self.hparams["initial_lr"]) 
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [lr_scheduler_1, lr_scheduler_2], milestones=[self.hparams["lr_step_size"]])
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'name': "cosine_annealing"}}
        elif self.hparams["lr_scheduler"] == "cosine_restarts":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams["lr_step_size"], T_mult=1, eta_min=self.hparams["lr_min"])
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'name': "cosine_annealing"}}
        else:
            RuntimeWarning(f"Specified learning rate scheduler '{self.hparams['lr_scheduler']}' not found in list of possible choices. Running without LR scheduler."
                           f"Choices are 'plateau', 'cosine'.")
            return {'optimizer': optimizer}
