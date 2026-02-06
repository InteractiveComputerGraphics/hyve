from typing import Callable, Optional, Union, Literal

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.nn import Sequential, Linear, Module

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from einops import pack, unpack

def translation_dependent_normal_plain(x_i: Tensor, x_j: Tensor):
    return torch.cat([x_i, x_j - x_i], dim=-1)

def translation_dependent_normal_oriented(x_i: Tensor, x_j: Tensor):
    x_i_p, x_i_n = unpack(x_i, [(3,), (3,)], "n *")
    x_j_p, x_j_n = unpack(x_j, [(3,), (3,)], "n *")
    return pack([x_i_p, x_j_p - x_i_p, cosine_similarity(x_i_n, x_j_n)[:, None]], "n *")[0]

def translation_dependent_normal_flip_ok(x_i: Tensor, x_j: Tensor):
    x_i_p, x_i_n = unpack(x_i, [(3,), (3,)], "n *")
    x_j_p, x_j_n = unpack(x_j, [(3,), (3,)], "n *")
    return pack([x_i_p, x_j_p - x_i_p, cosine_similarity(x_i_n, x_j_n)[:, None].abs()], "n *")[0]

def translation_invariant_normal_plain(x_i: Tensor, x_j: Tensor):
    return torch.cat([x_j - x_i], dim=-1)

def translation_invariant_normal_oriented(x_i: Tensor, x_j: Tensor):
    x_i_p, x_i_n = unpack(x_i, [(3,), (3,)], "n *")
    x_j_p, x_j_n = unpack(x_j, [(3,), (3,)], "n *")
    return pack([x_j_p - x_i_p, cosine_similarity(x_i_n, x_j_n)[:, None]], "n *")[0]

def translation_invariant_normal_flip_ok(x_i: Tensor, x_j: Tensor):
    x_i_p, x_i_n = unpack(x_i, [(3,), (3,)], "n *")
    x_j_p, x_j_n = unpack(x_j, [(3,), (3,)], "n *")
    return pack([x_j_p - x_i_p, cosine_similarity(x_i_n, x_j_n)[:, None].abs()], "n *")[0]

class EdgeConvCustom(MessagePassing):
    def __init__(self, input_len: int, output_len: int, activation_function: Callable[..., Module], aggr: str = 'max', translation_invariant: bool = False, normal_handling: Literal['plain','oriented','flip_ok'] = 'plain', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.input_len = input_len
        self.output_len = output_len
        self.translation_invariant = translation_invariant
        self.normal_handling = normal_handling
        self.message_fn = translation_dependent_normal_plain
        if self.translation_invariant:
            if self.normal_handling == 'plain':
                self.message_fn =  translation_invariant_normal_plain
                self.nn = Sequential(Linear(input_len, output_len), activation_function())
            elif self.normal_handling == 'oriented':
                self.message_fn = translation_invariant_normal_oriented
                self.nn = Sequential(Linear(input_len//2+1, output_len), activation_function())
            else: # Must be flip_ok
                self.message_fn = translation_invariant_normal_flip_ok
                self.nn = Sequential(Linear(input_len//2+1, output_len), activation_function())
        else:
            if self.normal_handling == 'plain':
                self.message_fn = translation_dependent_normal_plain
                self.nn = Sequential(Linear(2*input_len, output_len), activation_function())
            elif self.normal_handling == 'oriented':
                self.message_fn = translation_dependent_normal_oriented
                self.nn = Sequential(Linear(  input_len+1, output_len), activation_function())
            else: # Must be flip_ok
                self.message_fn = translation_dependent_normal_flip_ok
                self.nn = Sequential(Linear(  input_len+1, output_len), activation_function())

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(self.message_fn(x_i, x_j))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'