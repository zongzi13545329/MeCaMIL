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


class CausalGraphAttention(nn.Module):
    """Graph attention layer for causal relationship modeling"""
    def __init__(self, in_dim, out_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
    def forward(self, x, adjacency_mask):
        """
        x: [batch, num_nodes, in_dim]
        adjacency_mask: [num_nodes, num_nodes] - adjacency matrix of causal graph
        """
        B, N, _ = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply structural constraints from causal graph
        adjacency_mask = adjacency_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        attn = attn.masked_fill(adjacency_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)
        
        return out, attn


class StructuralEquationModel(nn.Module):
    """Structural equation model with graph neural networks"""
    def __init__(self, x_dim, u_dim, hidden_dim=256, depth=1, use_graph=True):
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_graph = use_graph
        
        # Causal graph structure: X -> Z <- U, Z -> Y
        # Node indices: 0=X, 1=U, 2=Z
        self.register_buffer('causal_adjacency', self._build_causal_graph())
        
        # Node embedding layers
        self.node_embeddings = nn.ModuleDict({
            'x_embed': nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            ),
            'u_embed': nn.Sequential(
                nn.Linear(u_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            ),
            'z_init': nn.Linear(1, hidden_dim)  # Z initialization
        })
        
        if use_graph:
            # Graph attention network for causal propagation
            self.graph_layers = nn.ModuleList([
                CausalGraphAttention(hidden_dim, hidden_dim, num_heads=4)
                for _ in range(depth)
            ])
        else:
            # Fallback to simple structural equations (for ablation)
            self.structural_equations = nn.ModuleDict({
                'x_to_z': nn.Linear(hidden_dim, hidden_dim),
                'u_to_z': nn.Linear(hidden_dim, hidden_dim),
                'combine': nn.Linear(hidden_dim * 2, hidden_dim)
            })
        
        # Learnable intervention vectors for do-intervention
        self.intervention_vectors = nn.ParameterDict({
            'u_intervention': nn.Parameter(torch.randn(1, hidden_dim))
        })
        
    def _build_causal_graph(self):
        """Construct adjacency matrix of causal graph"""
        adjacency = torch.zeros(3, 3)
        
        # Allow bidirectional propagation
        adjacency[2, 0] = 1  # Z → X (allow backward information flow)
        adjacency[2, 1] = 1  # Z → U
        
        # Self-loops (must be retained)
        adjacency[0, 0] = 1  # X self-loop
        adjacency[1, 1] = 1  # U self-loop
        adjacency[2, 2] = 1  # Z self-loop
        
        return adjacency
    
    def contrastive_loss(self, x_embed, u_embed, temperature=0.5):
        """Ensure X and U representations are relatively independent (as exogenous variables)"""
        # Normalization
        x_norm = F.normalize(x_embed, dim=-1)
        u_norm = F.normalize(u_embed, dim=-1)
        
        # Cosine similarity
        similarity = torch.mm(x_norm, u_norm.t()) / temperature
        
        # Minimize similarity between X and U
        loss = torch.mean(torch.exp(similarity))
        
        return loss
    
    def handle_missing_demographics(self, x, u):
        """Handle missing demographic data"""
        if u is None:
            return torch.zeros(x.size(0), self.u_dim, device=x.device), 0.0
            
        if u.size(0) == 1 and x.size(0) > 1:
            u = u.expand(x.size(0), -1)
            
        mask = torch.isnan(u)
        if mask.any():
            u = torch.where(mask, torch.zeros_like(u), u)
            uncertainty = 0.5  
        else:
            uncertainty = 1.0
        return u, uncertainty

    def forward_graph(self, x, u=None):
        """Causal propagation using graph neural networks"""
        batch_size = x.size(0)
        
        # Prepare node features
        x_embed = self.node_embeddings['x_embed'](x)
        
        if u is not None:
            u, uncertainty = self.handle_missing_demographics(x, u)
            u_embed = self.node_embeddings['u_embed'](u) * uncertainty
        else:
            u_embed = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Initialize Z as zero vector (Z is collider, determined by X and U)
        z_embed = self.node_embeddings['z_init'](torch.zeros(batch_size, 1, device=x.device))
        
        # Construct node feature matrix [batch, 3, hidden_dim]
        nodes = torch.stack([x_embed, u_embed, z_embed], dim=1)
        
        # Propagate information through graph attention layers
        for layer in self.graph_layers:
            nodes_new, attn = layer(nodes, self.causal_adjacency)
            nodes = nodes + nodes_new  # Residual connection
        
        # Extract Z node representation
        z = nodes[:, 2, :]
        
        return z, {'attention': attn, 'node_representations': nodes}

    def forward_structural(self, x, u=None):
        """Simplified version using structural equations (for ablation)"""
        x_contrib = self.node_embeddings['x_embed'](x)
        
        if u is not None:
            u, uncertainty = self.handle_missing_demographics(x, u)
            u_contrib = self.node_embeddings['u_embed'](u) * uncertainty
        else:
            u_contrib = torch.zeros_like(x_contrib)
        
        # Linear combination
        combined = torch.cat([x_contrib, u_contrib], dim=-1)
        z = self.structural_equations['combine'](combined)
        z = torch.relu(z)
        
        return z, {}

    def forward(self, x, u=None):
        """Unified forward propagation interface"""
        if self.use_graph:
            return self.forward_graph(x, u)
        else:
            return self.forward_structural(x, u)
    
    def get_causal_attribution(self, x, u):
        """Causal attribution analysis"""
        with torch.no_grad():
            # Full model output
            z_full, _ = self.forward(x, u)
            
            # X-only contribution
            z_x_only, _ = self.forward(x, None)
            
            # U-only contribution (requires zero input)
            x_zero = torch.zeros_like(x)
            z_u_only, _ = self.forward(x_zero, u)
            
            attributions = {
                'x_contribution': torch.norm(z_x_only, dim=-1).mean().item(),
                'u_contribution': torch.norm(z_u_only, dim=-1).mean().item(),
                'interaction': torch.norm(z_full - z_x_only - z_u_only, dim=-1).mean().item()
            }
            
        return attributions
    
    def sensitivity_analysis(self, x, u, num_samples=10):
        """Sensitivity analysis: perturb U and observe Z changes"""
        results = []
        
        with torch.no_grad():
            # Baseline
            z_baseline, _ = self.forward(x, u)
            
            for i in range(num_samples):
                # Add noise to U
                noise = torch.randn_like(u) * 0.1
                u_perturbed = u + noise
                
                z_perturbed, _ = self.forward(x, u_perturbed)
                
                # Compute change magnitude
                delta_z = torch.norm(z_perturbed - z_baseline, dim=-1).mean().item()
                results.append(delta_z)
        
        return {
            'mean_sensitivity': np.mean(results),
            'std_sensitivity': np.std(results),
            'all_deltas': results
        }


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, u_dim=0, hidden_dim=128,
                 dropout_v=0.0, nonlinear=True, passing_v=False, 
                 causal=False, convDepth=1, use_causal_graph=True):
        super(BClassifier, self).__init__()
        
        self.causal = causal
        self.u_dim = u_dim
        
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), 
                                  nn.Linear(128, 128), nn.Tanh())
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
        
        # Causal model components
        if causal and u_dim > 0:
            self.graph = StructuralEquationModel(
                x_dim=input_size, 
                u_dim=u_dim,
                hidden_dim=hidden_dim,
                depth=convDepth,
                use_graph=use_causal_graph
            )
            
            self.demographic_decoder = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, u_dim),
                nn.Sigmoid()
            )
            
            self.fcc = nn.Linear(hidden_dim, output_class)
        else:
            self.graph = None
            self.demographic_decoder = None
            self.fcc = nn.Linear(input_size, output_class)

        self.attention_weights = None

    def forward(self, feats, c, u=None):
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)

        result = {}
        
        # Attention mechanism
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

        if self.causal and self.u_dim > 0 and u is not None and self.graph is not None:
            # Process u dimensions
            if u.dim() == 2 and u.size(0) == feats.size(0):
                u_processed = u[0].unsqueeze(0)
            elif u.dim() == 1:
                u_processed = u.unsqueeze(0)
            else:
                u_processed = u
                
            if u_processed.size(-1) != self.u_dim:
                if u_processed.size(-1) > self.u_dim:
                    u_processed = u_processed[..., :self.u_dim]
                else:
                    padding = torch.zeros(u_processed.size(0), self.u_dim - u_processed.size(-1), 
                                        device=u_processed.device)
                    u_processed = torch.cat([u_processed, padding], dim=-1)
                
            # Use improved causal model
            z, causal_info = self.graph(epsilon, u_processed)
            result['Z'] = z
            result['causal_info'] = causal_info
            
            if hasattr(self, 'demographic_decoder') and self.demographic_decoder is not None:
                decoded_demographics = self.demographic_decoder(z)
                result['decoded_demographics'] = decoded_demographics
            
            fcc_output = self.fcc(z)
            disease_classes = torch.mean(fcc_output, dim=0, keepdim=True)
            
            # Causal attribution
            if hasattr(self.graph, 'get_causal_attribution'):
                try:
                    if u_processed.size(0) == 1 and epsilon.size(0) > 1:
                        u_for_attribution = u_processed.expand(epsilon.size(0), -1)
                    else:
                        u_for_attribution = u_processed
                    result['causal_attributions'] = self.graph.get_causal_attribution(
                        epsilon, u_for_attribution
                    )
                except Exception as e:
                    print(f"Warning: Could not compute causal attributions: {e}")
                    result['causal_attributions'] = {}
        else:
            fcc_output = self.fcc(epsilon)
            disease_classes = torch.mean(fcc_output, dim=0, keepdim=True)

        result['A'] = A
        result['B'] = B
        result['disease_classes'] = disease_classes
        result['using_causal'] = self.causal and self.u_dim > 0 and u is not None and self.graph is not None
        return disease_classes, result

    def get_attention_maps(self):
        return self.attention_weights
    
    def perform_sensitivity_analysis(self, feats, c, u):
        """Perform sensitivity analysis to verify causal assumptions"""
        if not (self.causal and self.graph is not None):
            return None
            
        # Get aggregated features through attention
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=feats.device)), 0)
        B = torch.mm(A.transpose(0, 1), V)
        epsilon = B.squeeze()
        if len(epsilon.shape) == 1:
            epsilon = epsilon.view([1, -1])
        
        # Process u
        if u.dim() == 2 and u.size(0) == feats.size(0):
            u_processed = u[0].unsqueeze(0)
        elif u.dim() == 1:
            u_processed = u.unsqueeze(0)
        else:
            u_processed = u
            
        # Execute sensitivity analysis
        return self.graph.sensitivity_analysis(epsilon, u_processed)


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x, u=None):
        feats, classes = self.i_classifier(x)
        
        u_processed = None
        if hasattr(self.b_classifier, 'causal') and self.b_classifier.causal and u is not None:
            if u.dim() == 2 and u.size(0) > 1:
                u_processed = u[0].unsqueeze(0)
            elif u.dim() == 1:
                u_processed = u.unsqueeze(0)
            else:
                u_processed = u
            
            if hasattr(self.b_classifier, 'u_dim') and u_processed.size(-1) != self.b_classifier.u_dim:
                print(f"Warning in MILNet: Expected {self.b_classifier.u_dim}-dim demographic, got {u_processed.size(-1)}-dim")
        
        prediction_bag, result = self.b_classifier(feats, classes, u_processed)
        return classes, prediction_bag, result, result['B']

    def get_interpretability_outputs(self, x, u=None):
        """Enhanced interpretability outputs"""
        try:
            feats, classes = self.i_classifier(x)
            
            u_processed = None
            if hasattr(self.b_classifier, 'causal') and self.b_classifier.causal and u is not None:
                if u.dim() == 2 and u.size(0) > 1:
                    u_processed = u[0].unsqueeze(0)
                elif u.dim() == 1:
                    u_processed = u.unsqueeze(0)
                else:
                    u_processed = u
                    
                if hasattr(self.b_classifier, 'u_dim') and u_processed.size(-1) != self.b_classifier.u_dim:
                    if u_processed.size(-1) > self.b_classifier.u_dim:
                        u_processed = u_processed[..., :self.b_classifier.u_dim]
                    else:
                        padding = torch.zeros(u_processed.size(0), 
                                            self.b_classifier.u_dim - u_processed.size(-1), 
                                            device=u_processed.device)
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
                'demographic_input_dim': u_processed.size(-1) if u_processed is not None else None,
                'causal_graph_info': result.get('causal_info', {})
            }
            
            return interpretability_outputs
            
        except Exception as e:
            print(f"Error in get_interpretability_outputs: {e}")
            return {
                'attention_weights': None,
                'feature_maps': None,
                'causal_attributions': {},
                'intermediate_representations': None,
                'prediction_bag': None,
                'instance_predictions': None,
                'using_causal': False,
                'error': str(e),
                'demographic_input_dim': None,
                'causal_graph_info': {}
            }
    
    def perform_sensitivity_analysis(self, x, u):
        """Perform sensitivity analysis at model level"""
        if not hasattr(self.b_classifier, 'perform_sensitivity_analysis'):
            return None
            
        feats, classes = self.i_classifier(x)
        return self.b_classifier.perform_sensitivity_analysis(feats, classes, u)
