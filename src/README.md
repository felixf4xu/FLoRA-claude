# (Claude generated) FLoRA Implementation

This project contains a **claude-generated** implementation of the FLoRA (Framework for Learning Scoring Rules in Autonomous Driving Planning Systems) (https://arxiv.org/html/2502.11352v1) method for autonomous vehicle planning evaluation as described in the paper. **WARNING**: LLM hallucinations exist in the code.

(The original author has not released the full code yet, which will be updated at https://github.com/ZikangXiong/FLoRA . `docs/subpages/supplementary.md` is from the original repo: https://github.com/ZikangXiong/FLoRA/blob/master/docs/subpages/supplementary.md)

## Files Overview

### Core Components

- **`predicate.py`** - Implements all predicates from the supplementary document:
  - **Action Predicates**: AccelerateNormal, AccelerateHard, DecelerateNormal, DecelerateHard, FastCruise, SlowCruise, TurnLeft, TurnRight, Start, Stop, ChangeLaneLeft, ChangeLaneRight, KeepLane, CenterInLane, SmoothSteering, FollowDistance, SmoothFollowing
  - **Condition Predicates**: CanChangeLaneLeft, CanChangeLaneRight, TrafficLightGreen, TrafficLightRed, VRUCrossing, VRUInPath
  - **Dual-Purpose Predicates**: InDrivableArea, Comfortable, Overtaking, SafeTTC

- **`layer.py`** - Neural network layers for temporal logic:
  - **TemporalLayer**: Implements temporal operators (Globally G, Finally F, Identity)
  - **PropositionalLayer**: Implements logical operators (AND, OR) with negation gates
  - **AggregationLayer**: Combines all logical expressions into final score

- **`scoring.py`** - Main scoring logic network that combines all components

- **`data_processing.py`** - Data preprocessing pipeline:
  - **PlanDataProcessor**: Converts trajectory data into predicate inputs
  - **FLoRADataLoader**: Batches data for training

- **`train.py`** - Training loop with regularization as described in Algorithm 1

- **`main.py`** - Complete demonstration and example usage

## Key Features Implemented

### 1. All Predicates from Supplementary Document
Each predicate implements the exact mathematical formula from the supplementary material:

```python
# Example: Decelerate Normal
p_dec_n = tanh(k(T + max(a_long(t))))
```

### 2. Temporal Logic Operations
- **Globally (G)**: `softmin` over time dimension
- **Finally (F)**: `softmax` over time dimension  
- **Soft attention**: Learned selection between temporal operators

### 3. Propositional Logic
- **AND**: `softmin` between predicates
- **OR**: `softmax` between predicates
- **Negation gates**: Learnable negation weights

### 4. Learnable Parameters
- **Predicate thresholds (T)**: Each predicate has learnable threshold parameters with specified ranges
- **Logic structure**: Weights for selecting between AND/OR operations
- **Temporal selection**: Weights for selecting between G/F/Identity operators

### 5. Regularization (Algorithm 1)
- **Sign-based regularization**: Encourages sparse predicate parameters
- **AND preference**: Encourages conjunctive logic structures
- **Parameter clamping**: Keeps parameters within valid ranges

## Usage Example

```python
from main import demo_flora_pipeline

# Run complete demonstration
model, data_loader = demo_flora_pipeline()

# This will:
# 1. Create all 23+ predicates
# 2. Process synthetic trajectory data
# 3. Evaluate predicates over time
# 4. Compute final scores through temporal/propositional layers
# 5. Demonstrate training step with regularization
```

## Mathematical Foundations

### Predicate Evaluation
Each predicate follows the pattern:
```
p_name = tanh(k * (T - metric(trajectory_data)))
```

Where:
- `k`: Fixed scaling parameter (default 1.0)
- `T`: Learnable threshold parameter with specified range
- `metric()`: Specific calculation for each predicate

### Temporal Operators
```python
G(p) = softmin(p, dim=time)  # Must hold globally
F(p) = softmax(p, dim=time)  # Must hold finally
```

### Logical Operators
```python
p1 ∧ p2 = softmin([p1, p2])  # Both must hold
p1 ∨ p2 = softmax([p1, p2])  # Either must hold
```

## Training

The training objective maximizes scores for expert demonstrations:
```python
loss = -mean(scores)  # Maximize scores
```

With regularization:
1. **L1-like regularization** on predicate parameters
2. **Structural regularization** favoring AND operations
3. **Parameter clamping** to valid ranges

## Extensions

To add new predicates:

1. Inherit from `Predicate` base class
2. Implement `forward()` method with mathematical formula
3. Add to `FLoRAPredicateFactory.create_all_predicates()`
4. Update `PlanDataProcessor` to provide required data fields

## Data Requirements

The implementation expects trajectory data with fields like:
- `a_long`: Longitudinal acceleration
- `velocity`: Vehicle velocity  
- `yaw_rate`: Yaw rate
- `d_lat`: Lateral distance to lane center
- `d_long`: Distance to lead vehicle
- And many more (see `data_processing.py` for complete list)

In practice, these would be extracted from NuPlan or similar datasets.

## Paper Correspondence

This implementation follows the paper's methodology:
- **Section 3**: Predicate formulations (all implemented)
- **Section 4**: Temporal logic neural networks (TemporalLayer)
- **Section 5**: Training with regularization (Algorithm 1)
- **Supplementary**: All mathematical formulas implemented exactly

The code provides a complete, runnable implementation of the FLoRA method suitable for research and development in autonomous vehicle planning evaluation.