import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)
        c = self.fc(feats.view(feats.shape[0], -1))
        return feats.view(feats.shape[0], -1), c

class StructuralEquationModel(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim=512, depth=1):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # 修改：调整u_to_z的输入层以适应8维u
        self.structural_equations = nn.ModuleDict({
            'x_to_z': nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            'u_to_z': nn.Sequential(
                nn.Linear(u_dim, hidden_dim),  # 现在u_dim=8而不是131
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            'noise_injection': nn.Linear(hidden_dim, hidden_dim)
        })
        
    def handle_missing_demographics(self, x, u):
        """修复：确保返回的u与x的batch size匹配，适应8维demographic"""
        if u is None:
            return torch.zeros(x.size(0), self.u_dim, device=x.device), 0.0
            
        # 如果u是单个样本但x是多个样本，扩展u
        if u.size(0) == 1 and x.size(0) > 1:
            u = u.expand(x.size(0), -1)
            
        mask = torch.isnan(u)
        if mask.any():
            u = torch.where(mask, torch.zeros_like(u), u)
            uncertainty = 0.5  
        else:
            uncertainty = 1.0
        return u, uncertainty

    def structural_equation_transform(self, x, u=None):
        x_contribution = self.structural_equations['x_to_z'](x)
        identity = x_contribution

        if u is not None:
            u, uncertainty = self.handle_missing_demographics(x, u)
            u_contribution = self.structural_equations['u_to_z'](u) * uncertainty
        else:
            u_contribution = torch.zeros_like(x_contribution)

        noise = torch.randn_like(x_contribution) * 0.1
        noise_contribution = self.structural_equations['noise_injection'](noise)

        z = x_contribution + u_contribution + noise_contribution
        return z + identity * 0.1 

    def do_intervention(self, x, u, interventions=None):
        if interventions is None:
            return self.structural_equation_transform(x, u)
        
        x_intervened = x.clone()
        u_intervened = u.clone() if u is not None else None
        
        for var, value in interventions.items():
            if var == 'U' and u_intervened is not None:
                u_intervened = torch.full_like(u_intervened, value)
            elif var == 'X':
                x_intervened = torch.full_like(x_intervened, value)
        
        return self.structural_equation_transform(x_intervened, u_intervened)

    def forward(self, x, u=None, interventions=None):     
        for _ in range(self.depth):
            if interventions is not None:
                x = self.do_intervention(x, u, interventions)
            else:
                x = self.structural_equation_transform(x, u)
        return x

    def get_causal_attribution(self, x, u=None):            
        """修复：增强causal attribution的健壮性，适应8维demographic"""
        try:
            baseline = self.forward(x, u)
            attributions = {}
            
            if u is not None:
                # 确保u的维度正确
                if u.size(0) != x.size(0):
                    if u.size(0) == 1:
                        u = u.expand(x.size(0), -1)
                    else:
                        # 如果维度不匹配且不能简单扩展，使用第一个样本
                        u = u[:1].expand(x.size(0), -1)
                
                try:
                    no_demo = self.forward(x, torch.zeros_like(u))
                    attributions['demographics'] = baseline - no_demo
                    
                    # 修改：适应8维demographic的详细attribution分析
                    if u.shape[-1] > 1:
                        # 为8维demographic的每一维计算attribution
                        demographic_names = ['gender_0', 'gender_1', 'race_0', 'race_1', 'race_2', 'race_3', 'race_4', 'age']
                        for i in range(min(u.shape[-1], len(demographic_names))):
                            u_modified = u.clone()
                            u_modified[..., i] = 0
                            modified_output = self.forward(x, u_modified)
                            attr_name = demographic_names[i] if i < len(demographic_names) else f'demo_attr_{i}'
                            attributions[attr_name] = baseline - modified_output
                            
                        # 额外计算组合效应
                        # Gender效应 (前2维)
                        if u.shape[-1] >= 2:
                            u_no_gender = u.clone()
                            u_no_gender[..., :2] = 0
                            modified_output = self.forward(x, u_no_gender)
                            attributions['gender_combined'] = baseline - modified_output
                            
                        # Race效应 (第3-7维)
                        if u.shape[-1] >= 7:
                            u_no_race = u.clone()
                            u_no_race[..., 2:7] = 0
                            modified_output = self.forward(x, u_no_race)
                            attributions['race_combined'] = baseline - modified_output
                            
                except Exception as e:
                    print(f"Warning: Could not compute demographic attributions: {e}")
                    attributions['demographics'] = torch.zeros_like(baseline)
            
            try:
                no_image = self.forward(torch.zeros_like(x), u)
                attributions['image_features'] = baseline - no_image
            except Exception as e:
                print(f"Warning: Could not compute image feature attributions: {e}")
                attributions['image_features'] = torch.zeros_like(baseline)
            
            return attributions
        except Exception as e:
            print(f"Error in get_causal_attribution: {e}")
            # 返回零填充的默认attribution
            return {'demographics': torch.zeros_like(x), 'image_features': torch.zeros_like(x)}


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, u_dim=0, hidden_dim=512,  # 修改：默认u_dim=0
                 dropout_v=0.0, nonlinear=True, passing_v=False, causal=False, convDepth=1):
        super(BClassifier, self).__init__()
        self.causal = causal
        self.u_dim = u_dim  # 新增：保存u_dim
        hidden_size = 64

        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)

        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # 修改：只有在causal=True且u_dim>0时才创建StructuralEquationModel
        if causal and u_dim > 0:
            self.graph = StructuralEquationModel(
                x_dim=input_size, 
                u_dim=u_dim,
                hidden_dim=hidden_dim,
                depth=convDepth
            )
            
            # 只有在使用causal时才创建demographic_decoder
            self.demographic_decoder = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, u_dim),  # 输出维度对应u_dim
                nn.Sigmoid()
            )
            
            # 使用causal path的输出分类器
            self.fcc = nn.Linear(hidden_dim, output_class)
        else:
            # 不使用causal时，直接使用传统MIL
            self.graph = None
            self.demographic_decoder = None
            # 直接从attention pooling的结果分类
            self.fcc = nn.Linear(input_size, output_class)

        self.attention_weights = None

    def forward(self, feats, c, u=None):
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)

        result = {}
        
        # 注意力机制计算（无论是否causal都需要）
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=feats.device)), 0)
        B = torch.mm(A.transpose(0, 1), V)
        self.attention_weights = A.detach()

        epsilon = B.squeeze()
        if len(epsilon.shape) == 1:
            epsilon = epsilon.view([1, -1])

        # 修改：根据causal和u_dim决定是否使用因果模型
        if self.causal and self.u_dim > 0 and u is not None and self.graph is not None:
            # 使用因果模型路径
            # 修复：确保u的维度处理正确
            if u.dim() == 2 and u.size(0) == feats.size(0):
                u_processed = u[0].unsqueeze(0)
            elif u.dim() == 1:
                u_processed = u.unsqueeze(0)
            else:
                u_processed = u
                
            # 添加维度检查，确保u符合预期的u_dim
            if u_processed.size(-1) != self.u_dim:
                print(f"Warning: Expected {self.u_dim}-dim demographic input, got {u_processed.size(-1)}-dim. Adjusting...")
                if u_processed.size(-1) > self.u_dim:
                    u_processed = u_processed[..., :self.u_dim]  # 截取前u_dim维
                else:
                    # 如果维度不足，用0填充
                    padding = torch.zeros(u_processed.size(0), self.u_dim - u_processed.size(-1), device=u_processed.device)
                    u_processed = torch.cat([u_processed, padding], dim=-1)
                
            z = self.graph(epsilon, u_processed)
            result['Z'] = z
            
            if hasattr(self, 'demographic_decoder') and self.demographic_decoder is not None:
                decoded_demographics = self.demographic_decoder(z)
                result['decoded_demographics'] = decoded_demographics
            
            fcc_output = self.fcc(z)
            disease_classes = torch.mean(fcc_output, dim=0, keepdim=True)
            
            # 修复：causal attribution的安全调用
            if hasattr(self.graph, 'get_causal_attribution'):
                try:
                    # 为attribution准备正确维度的u
                    if u_processed.size(0) == 1 and epsilon.size(0) > 1:
                        u_for_attribution = u_processed.expand(epsilon.size(0), -1)
                    else:
                        u_for_attribution = u_processed
                    result['causal_attributions'] = self.graph.get_causal_attribution(epsilon, u_for_attribution)
                except Exception as e:
                    print(f"Warning: Could not compute causal attributions: {e}")
                    result['causal_attributions'] = {}
        else:
            # 使用传统MIL路径（不使用因果模型）
            fcc_output = self.fcc(epsilon)
            disease_classes = torch.mean(fcc_output, dim=0, keepdim=True)

        result['A'] = A
        result['B'] = B
        result['disease_classes'] = disease_classes
        result['using_causal'] = self.causal and self.u_dim > 0 and u is not None and self.graph is not None
        return disease_classes, result

    def get_attention_maps(self):
        return self.attention_weights


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x, u=None):
        feats, classes = self.i_classifier(x)
        
        # 修复：只有在b_classifier支持causal且u不为None时才处理u
        u_processed = None
        if hasattr(self.b_classifier, 'causal') and self.b_classifier.causal and u is not None:
            if u.dim() == 2 and u.size(0) > 1:
                u_processed = u[0].unsqueeze(0)
            elif u.dim() == 1:
                u_processed = u.unsqueeze(0)
            else:
                u_processed = u
            
            # 添加维度检查
            if hasattr(self.b_classifier, 'u_dim') and u_processed.size(-1) != self.b_classifier.u_dim:
                print(f"Warning in MILNet: Expected {self.b_classifier.u_dim}-dim demographic, got {u_processed.size(-1)}-dim")
        
        prediction_bag, result = self.b_classifier(feats, classes, u_processed)
        return classes, prediction_bag, result

    def get_interpretability_outputs(self, x, u=None):
        """修复：增强interpretability输出的健壮性"""
        try:
            feats, classes = self.i_classifier(x)
            
            # 修复：只有在支持causal时才处理u
            u_processed = None
            if hasattr(self.b_classifier, 'causal') and self.b_classifier.causal and u is not None:
                if u.dim() == 2 and u.size(0) > 1:
                    u_processed = u[0].unsqueeze(0)
                elif u.dim() == 1:
                    u_processed = u.unsqueeze(0)
                else:
                    u_processed = u
                    
                # 确保维度正确
                if hasattr(self.b_classifier, 'u_dim') and u_processed.size(-1) != self.b_classifier.u_dim:
                    print(f"Warning: Adjusting demographic dim from {u_processed.size(-1)} to {self.b_classifier.u_dim}")
                    if u_processed.size(-1) > self.b_classifier.u_dim:
                        u_processed = u_processed[..., :self.b_classifier.u_dim]
                    else:
                        padding = torch.zeros(u_processed.size(0), self.b_classifier.u_dim - u_processed.size(-1), device=u_processed.device)
                        u_processed = torch.cat([u_processed, padding], dim=-1)
            
            prediction_bag, result = self.b_classifier(feats, classes, u_processed)
            
            interpretability_outputs = {
                'attention_weights': self.b_classifier.get_attention_maps(),
                'feature_maps': feats,
                'causal_attributions': result.get('causal_attributions', {}),
                'intermediate_representations': result.get('Z'),
                'prediction_bag': prediction_bag,
                'instance_predictions': classes,
                'using_causal': result.get('using_causal', False),
                'demographic_input_dim': u_processed.size(-1) if u_processed is not None else None
            }
            
            return interpretability_outputs
            
        except Exception as e:
            print(f"Error in get_interpretability_outputs: {e}")
            # 返回安全的默认输出
            return {
                'attention_weights': None,
                'feature_maps': None,
                'causal_attributions': {},
                'intermediate_representations': None,
                'prediction_bag': None,
                'instance_predictions': None,
                'using_causal': False,
                'error': str(e),
                'demographic_input_dim': None
            }