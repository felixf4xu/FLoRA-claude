import torch
import torch.nn as nn

# A small constant to avoid division by zero, as mentioned in SafeTTC predicate
EPSILON = 0.001

class Predicate(nn.Module):
    """Base class for a differentiable predicate."""
    def __init__(self, T, T_range, k=1.0):
        super().__init__()
        # T is the learnable threshold parameter
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        self.T_range = T_range
        self.k = k # The fixed 'k' parameter, needs to be tuned.

    def forward(self, plan_data):
        # Each subclass must implement its specific logic
        raise NotImplementedError
    
    def __call__(self, plan_data):
        return self.forward(plan_data)

    def clamp_T(self):
        """Clamps the learnable parameter T within its specified range."""
        self.T.data.clamp_(min=self.T_range[0], max=self.T_range[1])

class DecelerateNormal(Predicate):
    def __init__(self):
        # T: initialized with 2.0 [m/s²], range [1.0, 3.0]
        super().__init__(T=2.0, T_range=(1.0, 3.0))

    def forward(self, plan_data):
        """
        Calculates p_dec_n = tanh(k(T + max(a_long(t))))
        plan_data['a_long'] is a tensor of longitudinal accelerations over the plan.
        """
        if 'a_long' not in plan_data:
            return torch.tensor(0.0)
        
        max_a_long = torch.max(plan_data['a_long'])
        return torch.tanh(self.k * (self.T + max_a_long))

# Example Dual-Purpose Predicate: Comfortable
class Comfortable(Predicate):
    def __init__(self):
        super().__init__(T=1.0, T_range=(0.5, 2.0))  # Add dummy T for base class
        # These are the learnable thresholds for acceleration in different directions
        # θ_af: [0.0, 5.0], θ_ab: [0.0, 5.0], θ_al: [0.0, 3.0], θ_ar: [0.0, 3.0]
        self.theta_af = nn.Parameter(torch.tensor(1.0))
        self.theta_ab = nn.Parameter(torch.tensor(1.0))
        self.theta_al = nn.Parameter(torch.tensor(0.5))
        self.theta_ar = nn.Parameter(torch.tensor(0.5))
        self.ranges = {'af': (0.0, 5.0), 'ab': (0.0, 5.0), 'al': (0.0, 3.0), 'ar': (0.0, 3.0)}

    def forward(self, plan_data):
        """
        Calculates tanh(min(θ_i - max_acc_i(τ)))
        plan_data should contain max accelerations: 'max_acc_f', 'max_acc_b', etc.
        """
        vals = [
            self.theta_af - plan_data.get('max_acc_f', 0),
            self.theta_ab - plan_data.get('max_acc_b', 0),
            self.theta_al - plan_data.get('max_acc_l', 0),
            self.theta_ar - plan_data.get('max_acc_r', 0)
        ]
        min_val = torch.min(torch.stack(vals))
        return torch.tanh(min_val)

    def clamp_T(self):
        super().clamp_T()
        self.theta_af.data.clamp_(*self.ranges['af'])
        self.theta_ab.data.clamp_(*self.ranges['ab'])
        self.theta_al.data.clamp_(*self.ranges['al'])
        self.theta_ar.data.clamp_(*self.ranges['ar'])

# ====================== ACTION PREDICATES ======================

class AccelerateNormal(Predicate):
    def __init__(self):
        super().__init__(T=2.0, T_range=(1.0, 3.0))
    
    def forward(self, plan_data):
        if 'a_long' not in plan_data:
            return torch.tensor(0.0)
        min_a_long = torch.min(plan_data['a_long'])
        return torch.tanh(self.k * (self.T - min_a_long))

class AccelerateHard(Predicate):
    def __init__(self):
        super().__init__(T=4.0, T_range=(3.0, 5.0))
    
    def forward(self, plan_data):
        if 'a_long' not in plan_data:
            return torch.tensor(0.0)
        min_a_long = torch.min(plan_data['a_long'])
        return torch.tanh(self.k * (self.T - min_a_long))

class DecelerateHard(Predicate):
    def __init__(self):
        super().__init__(T=4.0, T_range=(3.0, 5.0))
    
    def forward(self, plan_data):
        if 'a_long' not in plan_data:
            return torch.tensor(0.0)
        max_a_long = torch.max(plan_data['a_long'])
        return torch.tanh(self.k * (self.T + max_a_long))

class FastCruise(Predicate):
    def __init__(self):
        super().__init__(T=0.5, T_range=(0.3, 1.0))
    
    def forward(self, plan_data):
        if 'a_long' not in plan_data:
            return torch.tensor(0.0)
        max_abs_a_long = torch.max(torch.abs(plan_data['a_long']))
        return torch.tanh(self.k * (self.T - max_abs_a_long))

class SlowCruise(Predicate):
    def __init__(self):
        super().__init__(T=0.5, T_range=(0.3, 1.0))
    
    def forward(self, plan_data):
        if 'a_long' not in plan_data:
            return torch.tensor(0.0)
        max_abs_a_long = torch.max(torch.abs(plan_data['a_long']))
        return torch.tanh(self.k * (self.T - max_abs_a_long))

class TurnLeft(Predicate):
    def __init__(self):
        super().__init__(T=0.3, T_range=(0.1, 0.5))
    
    def forward(self, plan_data):
        if 'yaw_rate' not in plan_data:
            return torch.tensor(0.0)
        min_yaw_rate = torch.min(plan_data['yaw_rate'])
        return torch.tanh(self.k * (self.T - min_yaw_rate))

class TurnRight(Predicate):
    def __init__(self):
        super().__init__(T=0.3, T_range=(0.1, 0.5))
    
    def forward(self, plan_data):
        if 'yaw_rate' not in plan_data:
            return torch.tensor(0.0)
        max_yaw_rate = torch.max(plan_data['yaw_rate'])
        return torch.tanh(self.k * (self.T + max_yaw_rate))

class Start(Predicate):
    def __init__(self):
        super().__init__(T=3.5, T_range=(1.5, 6.0))
    
    def forward(self, plan_data):
        if 'velocity' not in plan_data:
            return torch.tensor(0.0)
        v_max = torch.max(plan_data['velocity'])
        v_min = torch.min(plan_data['velocity'])
        velocity_range = v_max - v_min
        return torch.tanh(self.k * (self.T - velocity_range))

class Stop(Predicate):
    def __init__(self):
        super().__init__(T=0.95, T_range=(0.9, 1.0))
    
    def forward(self, plan_data):
        if 'velocity' not in plan_data:
            return torch.tensor(0.0)
        max_velocity = torch.max(plan_data['velocity'])
        return torch.tanh(self.k * (self.T - max_velocity))

class ChangeLaneLeft(Predicate):
    def __init__(self):
        super().__init__(T=0.95, T_range=(0.9, 1.0))
        self.w = 3.5  # lane width
        self.v_lat = 0.7  # lateral velocity threshold
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['d_lat', 'heading', 'lateral_velocity']):
            return torch.tensor(0.0)
        
        # Evaluate at end time t_1
        d_lat_t1 = plan_data['d_lat'][-1]
        heading_t1 = plan_data['heading'][-1]
        lat_vel_t1 = plan_data['lateral_velocity'][-1]
        
        terms = [
            torch.abs(d_lat_t1) / self.w,
            torch.abs(heading_t1),
            torch.abs(lat_vel_t1) / self.v_lat
        ]
        min_term = torch.min(torch.stack(terms))
        return torch.tanh(self.k * (self.T - min_term))

class ChangeLaneRight(Predicate):
    def __init__(self):
        super().__init__(T=0.95, T_range=(0.9, 1.0))
        self.w = 3.5  # lane width
        self.v_lat = 0.7  # lateral velocity threshold
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['d_lat', 'heading', 'lateral_velocity']):
            return torch.tensor(0.0)
        
        # Evaluate at end time t_1
        d_lat_t1 = plan_data['d_lat'][-1]
        heading_t1 = plan_data['heading'][-1]
        lat_vel_t1 = plan_data['lateral_velocity'][-1]
        
        terms = [
            torch.abs(d_lat_t1) / self.w,
            torch.abs(heading_t1),
            torch.abs(lat_vel_t1) / self.v_lat
        ]
        min_term = torch.min(torch.stack(terms))
        return torch.tanh(self.k * (self.T - min_term))

class KeepLane(Predicate):
    def __init__(self):
        super().__init__(T=0.2, T_range=(0.05, 0.4))
    
    def forward(self, plan_data):
        if 'lateral_position' not in plan_data:
            return torch.tensor(0.0)
        
        y_t0 = plan_data['lateral_position'][0]
        max_deviation = torch.max(torch.abs(plan_data['lateral_position'] - y_t0))
        return torch.tanh(self.k * (self.T - max_deviation))

class CenterInLane(Predicate):
    def __init__(self):
        super().__init__(T=0.2, T_range=(0.1, 0.3))
    
    def forward(self, plan_data):
        if 'd_lat' not in plan_data:
            return torch.tensor(0.0)
        
        max_d_lat = torch.max(torch.abs(plan_data['d_lat']))
        return torch.tanh(self.k * (self.T - max_d_lat))

class SmoothSteering(Predicate):
    def __init__(self):
        super().__init__(T=0.3, T_range=(0.2, 0.4))
    
    def forward(self, plan_data):
        if 'yaw_acceleration' not in plan_data:
            return torch.tensor(0.0)
        
        max_yaw_acc = torch.max(torch.abs(plan_data['yaw_acceleration']))
        return torch.tanh(self.k * (self.T - max_yaw_acc))

class FollowDistance(Predicate):
    def __init__(self):
        super().__init__(T=0.5, T_range=(0.3, 0.7))
        self.t_desired = 2.0  # desired time headway
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['d_long', 'velocity']):
            return torch.tensor(0.0)
        
        time_headway = plan_data['d_long'] / (plan_data['velocity'] + EPSILON)
        max_headway_error = torch.max(torch.abs(time_headway - self.t_desired))
        return torch.tanh(self.k * (self.T - max_headway_error))

class SmoothFollowing(Predicate):
    def __init__(self):
        super().__init__(T=0.3, T_range=(0.2, 0.4))
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['a_long', 'a_lead']):
            return torch.tensor(0.0)
        
        acc_ratio = plan_data['a_long'] / (plan_data['a_lead'] + EPSILON)
        max_ratio_error = torch.max(torch.abs(acc_ratio - 1.0))
        return torch.tanh(self.k * (self.T - max_ratio_error))

# ====================== DUAL-PURPOSE PREDICATES ======================

class InDrivableArea(Predicate):
    def __init__(self):
        super().__init__(T=0.3, T_range=(0.2, 0.5))
    
    def forward(self, plan_data):
        if 'd_drivable' not in plan_data:
            return torch.tensor(0.0)
        
        max_d_drivable = torch.max(plan_data['d_drivable'])
        return torch.tanh(self.k * (self.T - max_d_drivable))

class Overtaking(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.85, 0.95))
        self.v_min = 2.0
        self.w = 3.5
        self.d_min = 10.0
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['v_rel', 'd_lat_overtaken', 'd_long_overtaken']):
            return torch.tensor(0.0)
        
        v_rel_t = plan_data['v_rel']
        d_lat_t1 = plan_data['d_lat_overtaken'][-1]
        d_long_t0 = plan_data['d_long_overtaken'][0]
        
        terms = [
            v_rel_t / self.v_min,
            d_lat_t1 / self.w,
            d_long_t0 / self.d_min
        ]
        min_term = torch.min(torch.stack(terms))
        return torch.tanh(self.k * (self.T - min_term))

class SafeTTC(Predicate):
    def __init__(self):
        super().__init__(T=3.0, T_range=(2.0, 4.0))
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['distances', 'relative_velocities']):
            return torch.tensor(0.0)
        
        # Calculate TTC for all vehicles over all time steps
        distances = plan_data['distances']  # shape: (num_vehicles, time_steps)
        rel_velocities = plan_data['relative_velocities']  # shape: (num_vehicles, time_steps)
        
        ttc = distances / (torch.abs(rel_velocities) + EPSILON)
        min_ttc = torch.min(ttc)
        return torch.tanh(self.k * (self.T - min_ttc))

# ====================== CONDITION PREDICATES ======================

class CanChangeLaneLeft(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.85, 0.95))
        self.w_safe = 1.0
        self.d_lead_min = 15.0
        self.d_follow_min = 10.0
        self.v_rel_max = 5.0
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['d_lat_safe', 'd_long_lead', 'd_long_follow', 'v_rel_lead', 'v_rel_follow']):
            return torch.tensor(0.0)
        
        terms = [
            plan_data['d_lat_safe'] / self.w_safe,
            plan_data['d_long_lead'] / self.d_lead_min,
            plan_data['d_long_follow'] / self.d_follow_min,
            torch.abs(plan_data['v_rel_lead']) / self.v_rel_max,
            torch.abs(plan_data['v_rel_follow']) / self.v_rel_max
        ]
        min_term = torch.min(torch.stack(terms))
        return torch.tanh(self.k * (self.T - min_term))

class CanChangeLaneRight(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.85, 0.95))
        self.w_safe = 1.0
        self.d_lead_min = 15.0
        self.d_follow_min = 10.0
        self.v_rel_max = 5.0
    
    def forward(self, plan_data):
        if not all(k in plan_data for k in ['d_lat_safe', 'd_long_lead', 'd_long_follow', 'v_rel_lead', 'v_rel_follow']):
            return torch.tensor(0.0)
        
        terms = [
            plan_data['d_lat_safe'] / self.w_safe,
            plan_data['d_long_lead'] / self.d_lead_min,
            plan_data['d_long_follow'] / self.d_follow_min,
            torch.abs(plan_data['v_rel_lead']) / self.v_rel_max,
            torch.abs(plan_data['v_rel_follow']) / self.v_rel_max
        ]
        min_term = torch.min(torch.stack(terms))
        return torch.tanh(self.k * (self.T - min_term))

class TrafficLightGreen(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.8, 1.0))
    
    def forward(self, plan_data):
        if 'traffic_light_state' not in plan_data:
            return torch.tensor(0.0)
        
        # Assuming traffic_light_state is 1 for green, 0 for red
        green_state = plan_data['traffic_light_state']
        return torch.tanh(self.k * (self.T - (1.0 - green_state)))

class TrafficLightRed(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.8, 1.0))
    
    def forward(self, plan_data):
        if 'traffic_light_state' not in plan_data:
            return torch.tensor(0.0)
        
        # Assuming traffic_light_state is 1 for green, 0 for red
        red_state = 1.0 - plan_data['traffic_light_state']
        return torch.tanh(self.k * (self.T - (1.0 - red_state)))

class VRUCrossing(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.8, 1.0))
    
    def forward(self, plan_data):
        if 'vru_crossing' not in plan_data:
            return torch.tensor(0.0)
        
        crossing_detected = plan_data['vru_crossing']
        return torch.tanh(self.k * (self.T - (1.0 - crossing_detected)))

class VRUInPath(Predicate):
    def __init__(self):
        super().__init__(T=0.9, T_range=(0.8, 1.0))
    
    def forward(self, plan_data):
        if 'vru_in_path' not in plan_data:
            return torch.tensor(0.0)
        
        vru_detected = plan_data['vru_in_path']
        return torch.tanh(self.k * (self.T - (1.0 - vru_detected)))