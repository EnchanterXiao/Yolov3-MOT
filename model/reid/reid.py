import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.googlenet import GoogLeNet


class Model(nn.Module):
    def __init__(self, n_parts=8):
        super(Model, self).__init__()
        self.n_parts = n_parts

        self.feat_conv = GoogLeNet()
        self.conv_input_feat = nn.Conv2d(self.feat_conv.output_channels, 512, 1)

        # part net
        self.conv_att = nn.Conv2d(512, self.n_parts, 1)

        for i in range(self.n_parts):
            setattr(self, 'linear_feature{}'.format(i+1), nn.Linear(512, 64))

    def forward(self, x):
        feature = self.feat_conv(x)
        # print('google', feature.shape)
        feature = self.conv_input_feat(feature)

        att_weights = torch.sigmoid(self.conv_att(feature))
        # print('aat', att_weights.shape)
        linear_feautres = []
        for i in range(self.n_parts):
            masked_feature = feature * torch.unsqueeze(att_weights[:, i], 1)
            pooled_feature = F.avg_pool2d(masked_feature, masked_feature.size()[2:4])
            linear_feautres.append(
                getattr(self, 'linear_feature{}'.format(i+1))(pooled_feature.view(pooled_feature.size(0), -1))
            )

        concat_features = torch.cat(linear_feautres, 1)
        normed_feature = concat_features / torch.clamp(torch.norm(concat_features, 2, 1, keepdim=True), min=1e-6)

        return normed_feature


if __name__ == '__main__':
    '''
    plot reid_net structure
    '''
    from graphviz import Digraph
    from torch.autograd import Variable
    from model import net_utils

    def make_dot(var, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph
        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function
        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            assert isinstance(params.values()[0], Variable)
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)
        add_nodes(var.grad_fn)
        return dot

    x = Variable(torch.randn(1, 3, 80, 160))
    model = Model(n_parts=8)
    model.inp_size = (80, 160)
    ckpt = '../../data/googlenet_part8_all_xavier_ckpt_56.h5'
    net_utils.load_net(ckpt, model)
    model.eval()
    y = model(x)
    g = make_dot(y)
    g.view()


