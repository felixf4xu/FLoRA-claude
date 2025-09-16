"""
Complete FLoRA implementation example.
This script demonstrates the full pipeline from data processing to training.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

# Import all the components
from predicate import *
from layer import TemporalLayer, PropositionalLayer, AggregationLayer  
from scoring import ScoringLogicNetwork
from data_processing import PlanDataProcessor, FLoRADataLoader
from train import train_flora

class FLoRAPredicateFactory:
    """Factory for creating all FLoRA predicates."""
    
    @staticmethod
    def create_all_predicates() -> List[Predicate]:
        """Create all implemented predicates for FLoRA."""
        predicates = []
        
        # Action Predicates
        predicates.extend([
            AccelerateNormal(),
            AccelerateHard(), 
            DecelerateNormal(),
            DecelerateHard(),
            FastCruise(),
            SlowCruise(),
            TurnLeft(),
            TurnRight(),
            Start(),
            Stop(),
            ChangeLaneLeft(),
            ChangeLaneRight(),
            KeepLane(),
            CenterInLane(),
            SmoothSteering(),
            FollowDistance(),
            SmoothFollowing(),
        ])
        
        # Dual-Purpose Predicates
        predicates.extend([
            InDrivableArea(),
            Comfortable(),
            Overtaking(),
            SafeTTC(),
        ])
        
        # Condition Predicates
        predicates.extend([
            CanChangeLaneLeft(),
            CanChangeLaneRight(),
            TrafficLightGreen(),
            TrafficLightRed(),
            VRUCrossing(),
            VRUInPath(),
        ])
        
        return predicates

class FLoRAModel(nn.Module):
    """
    Complete FLoRA model that combines all components.
    """
    
    def __init__(self, num_temporal_layers: int = 2):
        super().__init__()
        
        # Create all predicates
        self.predicates = nn.ModuleList(FLoRAPredicateFactory.create_all_predicates())
        self.num_predicates = len(self.predicates)
        
        print(f"Initialized FLoRA with {self.num_predicates} predicates:")
        for i, pred in enumerate(self.predicates):
            print(f"  {i}: {pred.__class__.__name__}")
        
        # Create the scoring logic network
        self.scoring_network = ScoringLogicNetwork(
            self.predicates, 
            num_temporal_layers=num_temporal_layers
        )
    
    def forward(self, batch_of_plan_data: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the complete FLoRA model.
        
        Args:
            batch_of_plan_data: List of processed trajectory data
            
        Returns:
            Batch of scores for each trajectory
        """
        return self.scoring_network(batch_of_plan_data)
    
    def evaluate_predicates_on_batch(self, batch_of_plan_data: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Evaluate all predicates on a batch of trajectory data.
        
        Returns:
            Tensor of shape (batch_size, num_predicates, time_steps)
        """
        batch_size = len(batch_of_plan_data)
        time_steps = len(batch_of_plan_data[0]['velocity'])  # Assume all have same length
        
        # Initialize output tensor
        predicate_outputs = torch.zeros(batch_size, self.num_predicates, time_steps)
        
        # Evaluate each predicate on each trajectory at each time step
        for batch_idx, plan_data in enumerate(batch_of_plan_data):
            for pred_idx, predicate in enumerate(self.predicates):
                # For each time step, evaluate the predicate
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
        """Clamp all predicate parameters to their valid ranges."""
        for predicate in self.predicates:
            if hasattr(predicate, 'clamp_T'):
                predicate.clamp_T()

def demo_flora_pipeline():
    """
    Demonstration of the complete FLoRA pipeline.
    """
    print("=== FLoRA Pipeline Demonstration ===\n")
    
    # 1. Initialize data processing
    print("1. Initializing data processor...")
    data_processor = PlanDataProcessor(time_horizon=4.0, sampling_rate=20.0)
    data_loader = FLoRADataLoader(data_processor, batch_size=8)
    print(f"   Created data loader with {len(data_loader)} batches\n")
    
    # 2. Initialize model
    print("2. Initializing FLoRA model...")
    model = FLoRAModel(num_temporal_layers=2)
    print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # 3. Process a sample batch
    print("3. Processing sample batch...")
    sample_batch = next(iter(data_loader))
    print(f"   Batch size: {len(sample_batch)}")
    print(f"   Time steps: {len(sample_batch[0]['velocity'])}")
    print(f"   Data keys: {list(sample_batch[0].keys())}\n")
    
    # 4. Evaluate predicates
    print("4. Evaluating predicates...")
    with torch.no_grad():
        predicate_outputs = model.evaluate_predicates_on_batch(sample_batch)
        print(f"   Predicate outputs shape: {predicate_outputs.shape}")
        print(f"   Sample predicate values: {predicate_outputs[0, :5, 0]}\n")
    
    # 5. Forward pass through full model
    print("5. Computing final scores...")
    with torch.no_grad():
        scores = model(sample_batch)
        print(f"   Scores shape: {scores.shape}")
        print(f"   Sample scores: {scores[:3]}\n")
    
    # 6. Training step demonstration
    print("6. Demonstration training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass
    scores = model(sample_batch)
    loss = -torch.mean(scores)  # Maximize scores
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Show gradients
    print(f"   Loss: {loss.item():.4f}")
    total_grad_norm = torch.norm(torch.stack([
        p.grad.norm() for p in model.parameters() if p.grad is not None
    ]))
    print(f"   Total gradient norm: {total_grad_norm:.4f}")
    
    # Update parameters
    optimizer.step()
    model.clamp_predicate_params()
    
    print("   Parameter update completed\n")
    
    print("7. Model summary:")
    print(f"   Number of predicates: {model.num_predicates}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model, data_loader

def full_training_example():
    """
    Example of full training loop.
    """
    print("\n=== Full Training Example ===\n")
    
    # Initialize model and data
    model, data_loader = demo_flora_pipeline()
    
    # Train for a few epochs
    print("Starting training...")
    train_flora(model.scoring_network, data_loader, epochs=3, lr=1e-4)
    
    print("Training completed!")

if __name__ == "__main__":
    # Run demonstrations
    model, data_loader = demo_flora_pipeline()
    
    # Uncomment to run full training
    # full_training_example()
