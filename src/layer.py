import torch
import torch.nn as nn

# LogSumExp(-x)=-log(sum(exp(-x))) is a smooth approximation to min(x)
def smoothmin(x, dim=None, sharpness=10):
    return -torch.log(torch.sum(torch.exp(-sharpness * x), dim=dim)) / sharpness

# LogSumExp(x)=log(sum(exp(x))) is a smooth approximation to max(x)
# output any real number
def smoothmax(x, dim=None, sharpness=10):
    return torch.log(torch.sum(torch.exp(sharpness * x), dim=dim)) / sharpness

# Softmax for attention weights - preserves shape for probability distributions
# softmax: exp(x_i) / sum(exp(x_j)), Probability distribution normalization
# Outputs probabilities [0,1] that sum to 1
def attention_softmax(x, dim=None):
    """Standard softmax for attention weights that preserves tensor shape"""
    return torch.softmax(x, dim=dim)

class TemporalLayer(nn.Module):
    def __init__(self, num_predicates):
        super().__init__()
        # Weights for selecting between [G, F, Identity] for each predicate [cite: 5, 207]
        self.selection_weights = nn.Parameter(torch.randn(num_predicates, 3))

    def forward(self, x):
        # x shape: (batch, num_predicates, time_steps)
        
        # Temporal operators are min/max over the time dimension
        g_x = smoothmin(x, dim=2) # Globally (G) [cite: 3, 136]
        f_x = smoothmax(x, dim=2) # Finally (F) [cite: 3, 136]
        id_x = x[:, :, 0]       # Evaluate at t=0 (approximates identity for evaluation)

        # Stack operators for selection
        # Shape: (batch, num_predicates, 3)
        operators = torch.stack([g_x, f_x, id_x], dim=2)

        # Apply soft attention to select operator [cite: 5, 207, 208]
        # selection_probs shape: (num_predicates, 3)
        selection_probs = attention_softmax(self.selection_weights, dim=1)
        
        # Debug shapes
        print(f"operators.shape: {operators.shape}")
        print(f"selection_probs.shape: {selection_probs.shape}")
        print(f"selection_probs.unsqueeze(0).shape: {selection_probs.unsqueeze(0).shape}")

        # Output shape: (batch, num_predicates)
        output = torch.sum(operators * selection_probs.unsqueeze(0), dim=2)
        return output

class PropositionalLayer(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        # Create all unique pairs of inputs
        self.indices = torch.combinations(torch.arange(num_inputs))
        num_pairs = self.indices.shape[0]

        # Weights for selecting between [AND, OR] for each pair
        self.selection_weights = nn.Parameter(torch.randn(num_pairs, 2))
        
        # Learnable weights for negation gate [cite: 5, 214]
        self.negation_weights = nn.Parameter(torch.randn(num_pairs, 2))

    def forward(self, x):
        # x shape: (batch, num_inputs)
        
        # Get pairs of inputs
        # pair_a, pair_b shapes: (batch, num_pairs)
        pair_a = x[:, self.indices[:, 0]]
        pair_b = x[:, self.indices[:, 1]]

        # Apply negation gates
        neg_a = torch.tanh(self.negation_weights[:, 0]) * pair_a
        neg_b = torch.tanh(self.negation_weights[:, 1]) * pair_b

        # Logical operators are min/max
        op_and = smoothmin(torch.stack([neg_a, neg_b]), dim=0) # AND (∧) [cite: 3, 109]
        op_or = smoothmax(torch.stack([neg_a, neg_b]), dim=0)  # OR (∨) [cite: 3, 109]
        
        # Stack for selection
        operators = torch.stack([op_and, op_or], dim=2)
        selection_probs = attention_softmax(self.selection_weights, dim=1)

        # Output shape: (batch, num_pairs)
        output = torch.sum(operators * selection_probs.unsqueeze(0), dim=2)
        return output

class AggregationLayer(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        # Weights for selecting between [AND, OR] to connect inputs
        self.selection_weights = nn.Parameter(torch.randn(num_inputs - 1, 2))

    def forward(self, x):
        # x shape: (batch, num_inputs)
        
        score = x[:, 0]
        selection_probs = attention_softmax(self.selection_weights, dim=1)

        for i in range(x.shape[1] - 1):
            op_and = smoothmin(torch.stack([score, x[:, i+1]]), dim=0)
            op_or = smoothmax(torch.stack([score, x[:, i+1]]), dim=0)
            
            # Choose between AND/OR based on learned weights
            score = selection_probs[i, 0] * op_and + selection_probs[i, 1] * op_or
            
        return score.unsqueeze(1)  # Shape: (batch, 1)