"""
CHAOS GROK (FIXED)
==================
1. Fixes the Seaborn import error.
2. Adds 'Attractor Stability' test (Does it stay on the butterfly?).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns # Ensure this is installed: pip install seaborn

# ============================================================================
# 1. THE CHAOS GENERATOR
# ============================================================================
def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_chaos(seq_len=10000, dt=0.01):
    t = np.arange(0, seq_len * dt, dt)
    initial_state = [1.0, 1.0, 1.0]
    states = odeint(lorenz_system, initial_state, t)
    # Normalize
    states = (states - states.mean(axis=0)) / states.std(axis=0)
    return torch.tensor(states, dtype=torch.float32)

# ============================================================================
# 2. THE RESONANT CORTEX
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.1)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.1)

    def forward(self, z):
        real_out = self.fc_r(z.real) - self.fc_i(z.imag)
        imag_out = self.fc_r(z.imag) + self.fc_i(z.real)
        return torch.complex(real_out, imag_out)

class ResonantColumn(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mixer = ComplexLinear(width, width)
        self.bias_phase = nn.Parameter(torch.randn(width) * 0.1)

    def forward(self, z):
        z_mixed = self.mixer(z)
        amp = torch.tanh(torch.abs(z_mixed)) 
        phase = torch.angle(z_mixed) + self.bias_phase 
        return torch.polar(amp, phase)

class RCNet_Chaos(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.input_r = nn.Linear(input_dim, hidden_dim)
        self.input_i = nn.Linear(input_dim, hidden_dim)
        self.column1 = ResonantColumn(hidden_dim)
        self.column2 = ResonantColumn(hidden_dim)
        self.readout = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        r = self.input_r(x); i = self.input_i(x)
        z = torch.complex(r, i)
        z = z + self.column1(z)
        z = z + self.column2(z)
        cat = torch.cat([z.real, z.imag], dim=1)
        return self.readout(cat), z

# ============================================================================
# 3. EXPERIMENT
# ============================================================================
def train_chaos(device):
    print("🌪️  Booting Chaos Simulation...")
    data = generate_chaos().to(device)
    X = data[:-1]; Y = data[1:]
    
    model = RCNet_Chaos().to(device)
    opt = optim.AdamW(model.parameters(), lr=0.005)
    crit = nn.MSELoss()
    
    # Train
    for epoch in range(1000):
        model.train(); opt.zero_grad()
        pred, wave_state = model(X)
        loss = crit(pred, Y)
        loss.backward(); opt.step()
        if epoch % 100 == 0: print(f"Ep {epoch} | MSE: {loss.item():.6f}")
            
    # Visualize
    print("\nVisualizing the Attractor...")
    test_len = 2000 # Long run to check stability
    current_state = X[0].unsqueeze(0)
    generated_path = []
    
    with torch.no_grad():
        for _ in range(test_len):
            next_state, _ = model(current_state)
            generated_path.append(next_state.cpu().numpy()[0])
            current_state = next_state 
            
    gen_path = np.array(generated_path)
    true_path = data[:test_len].cpu().numpy()
    
    fig = plt.figure(figsize=(15, 6))
    
    # 3D Phase Space
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(true_path[:,0], true_path[:,1], true_path[:,2], lw=0.5, alpha=0.5, color='blue', label='Reality')
    ax1.plot(gen_path[:,0], gen_path[:,1], gen_path[:,2], lw=0.8, color='red', label='AI Dream')
    ax1.set_title("The Butterfly Effect")
    ax1.legend()
    
    # Phase Locking
    ax2 = fig.add_subplot(1, 2, 2)
    phases = torch.angle(wave_state[:100, :]).detach().cpu().numpy()
    sns.heatmap(phases, cmap='twilight', cbar=True)
    ax2.set_title('Resonant Frequencies (The Hidden Rules)')
    
    plt.tight_layout()
    plt.savefig('chaos_fixed.png')
    print("Saved 'chaos_fixed.png'")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_chaos(device)