import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from PyTorchHelpers import *
import nnBlocks as nnb

from collections import OrderedDict
from math import sqrt
import random

class ResNet(nn.Module):
    """ Almost identical to the pytorch implimentation.  Bias is included since there are no batchnorm layers """
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()

        self.inplanes = 20 

        self.groups = groups
        self.base_width = width_per_group
        self.conv1   = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.relu    = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(block, 20,   layers[0])
        self.layer2  = self._make_layer(block, 40,   layers[1], stride=2)
        self.layer3  = self._make_layer(block, 60,   layers[2], stride=2)
        self.layer4  = self._make_layer(block, 80,   layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(80, num_classes, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MLClassifier (nn.Module):
    """Apply 3 channel-wise linear layers to an input"""
    def __init__(self, features):
        """
        Parameters
        ----------
        features: Input features

        Returns
        -------
        torch.Tensor: Shape [1,3] logits
        """
        super (MLClassifier, self).__init__()
        self.O = features

        self.r0    = nn.Linear(self.O, 1)
        self.r1    = nn.Linear(self.O, 1)
        self.r2    = nn.Linear(self.O, 1)

    def forward(self, x):
        x = torch.stack([self.r0(x[0]), self.r1(x[1]), self.r2(x[2])])
        x = x.view(1, 3)
        return x



class ContextLayer (nn.Module):
    """Apply 3 channel-wise linear layers to an input"""
    def __init__(self, features):
        """
        Performs context scaling with learnable affine transformation

        Parameters
        ----------
        features: Input features

        Returns
        -------
        torch.Tensor: Shape [N,L]
        """
        super (ContextLayer, self).__init__()
        self.L = features
        self.bn = nn.BatchNorm1d(self.L, track_running_stats=False )
        self.relu = nn.LeakyReLU(0.1) 
        self.do = nn.Dropout(0.25)
    def forward(self, x):
        x_z0 = self.bn(x) # Feature singifance
        x_a0 = self.do(self.relu(x))
        return x_a0, x_z0 


class Attention(nn.Module):
    """
    A labour of love. Don't mess with it
    """
    def __init__(self, n_classes, class_weights=None):
        super(Attention, self).__init__()
        self.L = 80    # Input features to attention mechanism
        self.D = 40    # Hidden dimension for attention mechanism
        self.O = 1     # Output nodes
        self.K = 3     # Attention maps
        self.C = n_classes

        if class_weights is not None:
            print ("I'm weighting classes according to class_weights=", class_weights)
            self.loss = nnb.CrossEntropyWithProbs(classes=3,weight=class_weights, smoothing=0.25)
        else:
            self.loss = nnb.CrossEntropyWithProbs(classes=3, smoothing=0.25)

        self.cnn  = nn.DataParallel(
            ResNet(block=nnb.BasicResBlock, layers=[3, 3, 3, 3], num_classes=self.L),
            device_ids=[0, 1, 2, 3]
        ).cuda()

        self.context = ContextLayer(self.L) 

        # Attention mechanism --> a_n
        self.attention = nn.Sequential(OrderedDict([
            ('lin1',      nn.Linear(self.L, self.D)),
            ('tanh',      nn.Tanh()),
            ('lin2',      nn.Linear(self.D, self.K)),
        ]))

        # Generate instance codes --> b_n
        self.buffer = nn.Sequential(OrderedDict([
            ('lin1',       nn.Linear(self.L, self.D)),
            ('relu',       nn.LeakyReLU(0.1)),
            ('classifier', nn.Linear(self.D, self.O)),
        ]))

        self.weight_mask = nn.Parameter(torch.tensor([0.25,0.25,0.25]))
        self.off_diag    = 1 - torch.eye(3).cuda()

        # Generate instance codes --> b_n
#        self.classifier = MLClassifier(self.O)

        self.reset_params()

    @staticmethod
    def weight_init(m, name=''):
        if isinstance(m, nn.Linear):
         # Using tanh with fan in
            if 'attention' in name:
                print (name + " with init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')")
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
            elif 'classifier' in name:
                print (name + " init.xavier_normal_(m.weight)")
                init.xavier_normal_(m.weight)
            else:
                print (name + " with init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')")
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m. bias is not None: init.zeros_(m.bias)
        if isinstance(m, nn.Conv2d):
            # From Resnet code
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
            if m. bias is not None: init.zeros_(m.bias)
    def reset_params(self):
        for name, m in self.named_modules():
            self.weight_init(m, name)
        
    def reset_linear(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m. bias is not None: init.zeros_(m.bias)
        
    def forward(self, full_input: torch.Tensor, Y: torch.Tensor = torch.tensor([1]).cuda()):
        # ==========================================================
        # Feature Extractor
        if self.training:
            indices = torch.randperm(full_input.shape[0])[:int (full_input.shape[0] * 0.2)]
            H = self.cnn(full_input[indices].detach())
        else:
            H = self.cnn(full_input.detach())

        # ==========================================================
        # Measure slide feature statistics and activation penality (prevent harmonic shift)
        mu, var  = H.mean(dim=0), H.var(dim=0)
        KLD = 0.5 * H.pow(2).mean()

        # ==========================================================
        # Do the context layer 
        Hm0, Hz0 = self.context (H)

        # ==========================================================
        # Generate Aterm, lock, measure attention weight statistics per map (dim0) for each (sum)
        Aterm_raw   = self.attention(Hz0)

        Aterm_act   = F.softplus(Aterm_raw)
        Aterm_mask  = torch.sigmoid(-10.0*self.weight_mask) * Aterm_act + torch.sigmoid(10.0*self.weight_mask)
        Aterm_1norm = F.normalize(Aterm_mask, p=1, dim=0)
        Aterm_1T    = torch.transpose(Aterm_1norm, 1, 0)  # KxNb

        Aterm_2norm = F.normalize(Aterm_raw,  p=2, dim=0)
        Aterm_2T    = torch.transpose(Aterm_2norm, 1, 0)  # KxNb
        Aterm_var   = (torch.mm(Aterm_2T, Aterm_2norm) * self.off_diag).mean() 
        Aterm_mu    = 0.5 * Aterm_raw.mean(dim=0).pow(2).sum()

        # ==========================================================
        # Generate the instance codes (buffer)
        Bterm  = self.buffer(Hm0)

        # ==========================================================
        # Dynamic pooling across 3 attention maps, M -> [3, 16]
        Mterm = torch.mm(Aterm_1T, Bterm)
        wROIs = Aterm_1T * Bterm.view(Bterm.shape[0])
        slide_embedding = Mterm.view(1, self.O*self.K)

        # ==========================================================
        # Use the multi-level classifier to indepedently score each class, turn into probabilities
        y_logit      = slide_embedding
#        y_logit      = self.classifier(Mterm)
        y_pred       = F.softmax(y_logit, dim=1)

        # ==========================================================
        # Get the predicted label
        Y_real_hat   = Y.long()
        y_pred_hat   = torch.argmax(y_pred).long()
        ce_loss      = self.loss(y_logit, Y_real_hat)
        error        = 1 - y_pred_hat.eq(Y_real_hat).float()

        # ==========================================================
        # Classifier penality
        l2 = torch.stack([p.norm() for n,p in self.buffer.named_parameters() if 'weight' in n]).mean()

        # return template is: (activations), (loss), (regularizations), (predictions)
        output = {
            'Aterm':Aterm_1T.detach(),
            'wROIs':wROIs.detach(),
            'Bterm':Bterm.detach(),
            'Mterm':Mterm.detach(),
            'Fterm':H.detach(),
            'Aterm_mu':Aterm_mu.detach(),
            'Aterm_var':Aterm_var.detach(),
            'loss':ce_loss,
            'l2':l2,
            'KLD':KLD.detach(),
            'y_pred':y_pred.detach(),
            'y_pred_hat':y_pred_hat.detach(),
            'error':error
        }
        return output
#        return (Aterm, Bterm, M, Aterm, Aterm_var, H), ce_loss, (weight_loss, l2, KLD), (y_pred, y_pred_hat, error)

    def tile_activation(self, partial_input, Y, step_input=1):

        H = self.cnn(partial_input, step_input)
        H = self.fbn(H)
        T_raw  = nn.Softmax(dim=1)(self.classifier(H))
        A_raw  = self.attention(H)


        return A_raw, T_raw
