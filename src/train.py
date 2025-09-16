import torch
from layer import PropositionalLayer, AggregationLayer

def train_flora(model, data_loader, epochs, lr=1e-4, alpha=1e-5, beta=1e-3, w_max=5.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            # Objective: Maximize the score for good demonstrations [cite: 3, 117]
            scores = model(batch)
            loss = -torch.mean(scores) # We minimize -score to maximize score

            # Update weights and predicate parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Regularization Step (from Algorithm 1) ---
            with torch.no_grad():
                # 1. Regularize predicate parameters (θ) [cite: 5, 264]
                for p in model.predicates:
                    for param in p.parameters():
                        if param.grad is not None:
                            # This approximates the sign-based update
                            grad_sign = torch.sign(param.grad)
                            param.data -= alpha * grad_sign
                
                # Clamp predicate parameters
                model.clamp_predicate_params()

                # 2. Regularize logic structure (encourage AND over OR) [cite: 5, 266]
                for layer in model.modules():
                    if isinstance(layer, (PropositionalLayer, AggregationLayer)):
                        # Increase the weight for the AND operator (index 0)
                        layer.selection_weights.data[:, 0] += beta
                        layer.selection_weights.data[:, 0].clamp_(max=w_max)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}, Avg Score: {-avg_loss:.4f}")