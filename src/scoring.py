import torch
import torch.nn as nn
from layer import TemporalLayer, PropositionalLayer, AggregationLayer

class ScoringLogicNetwork(nn.Module):
    def __init__(self, predicate_list, num_temporal_layers=1):
        super().__init__()
        self.predicates = nn.ModuleList(predicate_list)
        num_predicates = len(self.predicates)

        # Only one temporal layer is needed to process the time dimension
        self.temporal_layer = TemporalLayer(num_predicates)
        
        # The output of the temporal layer becomes the input for the propositional layer
        self.propositional_layer = PropositionalLayer(num_predicates)
        
        num_prop_outputs = torch.combinations(torch.arange(num_predicates)).shape[0]
        self.aggregation_layer = AggregationLayer(num_prop_outputs)

    def forward(self, batch_of_plan_data):
        # batch_of_plan_data: list of dicts, one for each item in the batch
        
        # Evaluate all predicates over the time horizon for the whole batch
        # This part requires significant data processing from NuPlan logs
        # Let's assume predicate_outputs is shape: (batch, num_predicates, time_steps)
        predicate_outputs = self._evaluate_predicates(batch_of_plan_data)
        
        temporal_out = self.temporal_layer(predicate_outputs)
        propositional_out = self.propositional_layer(temporal_out)
        final_score = self.aggregation_layer(propositional_out)
        
        return final_score

    def _evaluate_predicates(self, batch_of_plan_data):
        """
        Evaluate all predicates on a batch of trajectory data.
        Returns predicate outputs over time for the whole batch.
        """
        batch_size = len(batch_of_plan_data)
        time_steps = len(batch_of_plan_data[0]['velocity']) if batch_of_plan_data else 80
        num_predicates = len(self.predicates)
        
        # Initialize output tensor
        predicate_outputs = torch.zeros(batch_size, num_predicates, time_steps)
        
        # Evaluate each predicate on each trajectory at each time step
        for batch_idx, plan_data in enumerate(batch_of_plan_data):
            for pred_idx, predicate in enumerate(self.predicates):
                # For each time step, evaluate the predicate causally
                for t in range(time_steps):
                    # Extract data up to time t for causal evaluation
                    causal_data = {}
                    for key, value in plan_data.items():
                        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) > 1:
                            causal_data[key] = value[:t+1]
                        elif hasattr(value, '__len__') and not torch.is_tensor(value) and len(value) > 1:
                            causal_data[key] = value[:t+1]
                        else:
                            causal_data[key] = value
                    
                    # Evaluate predicate
                    pred_value = predicate(causal_data)
                    predicate_outputs[batch_idx, pred_idx, t] = pred_value
        
        return predicate_outputs

    def clamp_predicate_params(self):
        """Helper to clamp all predicate parameters during training."""
        for p in self.predicates:
            if hasattr(p, 'clamp_T'):
                p.clamp_T()