#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.


from torch.nn import Module, ModuleList
from torch.nn import Conv2d
from torch.nn import InstanceNorm2d
from torch.nn import Dropout, Dropout2d, Dropout1d
from torch.nn import ReLU, GELU
from torch.nn.functional import pad
import random


def get_norm(norm_type, c_):
    if norm_type == 'in':
        return InstanceNorm2d(c_, eps=0.001, momentum=0.99, track_running_stats=False)
    else:
        return None

def get_activation(activation_type):
    if activation_type.lower() == 'relu':
        return ReLU()
    elif activation_type.lower() == 'gelu':
        return GELU(approximate='tanh')
    else:
        return None


class MixDropout(Module):
    def __init__(self, dropout):
        super(MixDropout, self).__init__()
        if type(dropout) is dict:
            dropout_proba = dropout['dropout_proba']
            dropout2d_proba = dropout['dropout2d_proba']
        else: # float
            dropout_proba = dropout
            dropout2d_proba = dropout / 2

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)

class MixDropout1d(Module):
    def __init__(self, dropout):
        super(MixDropout1d, self).__init__()
        if type(dropout) is dict:
            dropout_proba = dropout['dropout_proba']
            dropout1d_proba = dropout['dropout1d_proba']
        else: # float
            dropout_proba = dropout
            dropout1d_proba = dropout / 2
        self.dropout = Dropout(dropout_proba)
        self.dropout1d = Dropout1d(dropout1d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout1d(x)


class DepthSepConv2D(Module):
    def __init__(self, in_, out_, ks=3, stride=(1, 1)):
        super(DepthSepConv2D, self).__init__()
        self.depth_conv = Conv2d(in_, in_, kernel_size=ks, padding=ks//2, stride=stride, groups=in_)
        self.point_conv = Conv2d(in_, out_, kernel_size=(1, 1))

    def forward(self, x, dropout=None):
        x = self.depth_conv(x)
        if dropout: # and pos == 2:
            x = dropout(x)
        x = self.point_conv(x)
        return x


class FCN_Encoder(Module):
    def __init__(self, params):
        super(FCN_Encoder, self).__init__()
        """
        param - model specific parameters
        params - parameter shared by all models
        """
        dropout = params.get('dropout', 0.)
        dropout_func = params.get('dropout_func', 'MixDropout')
        dropout = globals()[dropout_func](dropout)
        drop_inside = params.get('drop_inside', False)
        kernel_size = params.get('kernel_size', 3)
        activation = params.get('activation', 'relu')
        norm = params.get('norm', 'in')

        self.init_blocks = ModuleList([
            ConvBlock([params["input_channels"], 16, 16, 16], stride=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([16, 32, 32, 32], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([32, 64, 64, 64], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([64, 128, 128, 128], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([128, 128, 128, 128], stride=(2, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([128, 128, 128, 128], stride=(2, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
        ])

        self.blocks = ModuleList([
            # representation learning x 4
            DSCBlock([128, 128, 128, 128], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
            DSCBlock([128, 128, 128, 128], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
            DSCBlock([128, 128, 128, 128], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
            DSCBlock([128, 256, 256, 256], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class FCN_Encoder16x16(Module):
    def __init__(self, params):
        super(FCN_Encoder16x16, self).__init__()
        """
        param - model specific parameters
        params - parameter shared by all models
        """
        dropout = params.get('dropout', 0.)
        dropout_func = params.get('dropout_func', 'MixDropout')
        dropout = globals()[dropout_func](dropout)
        drop_inside = params.get('drop_inside', False)
        kernel_size = params.get('kernel_size', 3)
        activation = params.get('activation', 'relu')
        norm = params.get('norm', 'in')

        self.init_blocks = ModuleList([
            ConvBlock([params["input_channels"], 16, 16, 16], stride=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([16, 32, 32, 32], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([32, 64, 64, 64], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([64, 128, 128, 128], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
            ConvBlock([128, 256, 256, 256], stride=(2, 2), ks=kernel_size, activation=activation, norm=norm, dropout=dropout),
        ])

        self.blocks = ModuleList([
            # representation learning x 4
            DSCBlock([256, 256, 256, 256], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
            DSCBlock([256, 256, 256, 256], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
            DSCBlock([256, 256, 256, 256], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
            DSCBlock([256, 256, 256, 256], pool=(1, 1), ks=kernel_size, activation=activation, norm=norm, dropout=dropout, drop_inside=drop_inside),
        ])

    def forward(self, x, intermediate=False):
        xs = []  # intermediate features
        for b in self.init_blocks:
            x = b(x)
            xs.append(x)

        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
            xs.append(x)
        if intermediate:
            return xs
        return x



class ConvBlock(Module):
    def __init__(self, channels, stride=(1, 1), ks=3, activation='relu', norm='in', dropout=None):
        super(ConvBlock, self).__init__()
        assert len(channels) == 4
        self.conv1 = Conv2d(channels[0], channels[1], kernel_size=ks, padding=ks // 2)
        self.conv2 = Conv2d(channels[1], channels[2], kernel_size=ks, padding=ks // 2)
        self.conv3 = Conv2d(channels[2], channels[3], kernel_size=ks, padding=ks // 2, stride=stride)
        self.activation = get_activation(activation)
        self.norm = get_norm(norm, channels[2])
        self.dropout = dropout

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if self.dropout and pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if self.dropout and pos == 2:
            x = self.dropout(x)

        if self.norm:
            x = self.norm(x)

        x = self.conv3(x)
        x = self.activation(x)

        if self.dropout and pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(Module):
    def __init__(self, channels, pool=(2, 1), ks=3, activation='relu', norm=None, dropout=None, drop_inside=False):
        super(DSCBlock, self).__init__()
        assert len(channels) == 4
        self.conv1 = DepthSepConv2D(channels[0], channels[1], ks)
        self.conv2 = DepthSepConv2D(channels[1], channels[2], ks)
        self.conv3 = DepthSepConv2D(channels[2], channels[3], ks, stride=pool)
        self.activation = get_activation(activation)
        self.norm = get_norm(norm, channels[2])
        self.dropout = dropout
        self.drop_inside = drop_inside

    def forward(self, x):
        pos = random.randint(1, 3)

        if self.dropout and self.drop_inside and pos == 1:
            x = self.conv1(x, self.dropout)
        else:
            x = self.conv1(x, None)

        x = self.activation(x)

        if self.dropout and not self.drop_inside and pos == 1:
            x = self.dropout(x)

        if self.dropout and self.drop_inside and pos == 2:
            x = self.conv2(x, self.dropout)
        else:
            x = self.conv2(x, None)

        x = self.activation(x)

        if self.dropout and not self.drop_inside and pos == 2:
            x = self.dropout(x)

        if self.norm:
            x = self.norm(x)

        if self.dropout and self.drop_inside and pos == 3:
            x = self.conv3(x, self.dropout)
        else:
            x = self.conv3(x, None)

        if self.dropout and not self.drop_inside and pos == 3:
            x = self.dropout(x)

        return x
