"""
MANDELBROT GROK: EMERGENT COMPLEXITY
====================================
Can a Neural Network 'dream' the Mandelbrot Set without ever seeing it?

1. Train: Learn the local physics z -> z^2 + c (Regression on random numbers).
2. Generate: Iterate the Network recursively on a 2D grid.
3. Result: The Fractal emerges from the weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# 1. THE RESONANT CORTEX (Universal Physics Simulator)
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        
        # Initialize for stable recurrence
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.5)

    def forward(self, z):
        # z is complex tensor [Batch, Features]
        # Weights are complex
        # Output = W * z + b
        
        real_out = self.fc_r(z.real) - self.fc_i(z.imag) + self.bias_r
        imag_out = self.fc_r(z.imag) + self.fc_i(z.real) + self.bias_i
        
        return torch.complex(real_out, imag_out)

class ResonantNetwork(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        # Input: z (1 dim complex), c (1 dim complex) -> Total 2 complex features
        self.layer1 = ComplexLinear(2, width)
        self.layer2 = ComplexLinear(width, width)
        self.layer3 = ComplexLinear(width, 1) # Output z'
        
        # Activation: Complex ReLU (modReLU) or similar
        # We need a non-linearity that allows squaring behavior (amplitude expansion)
        # z^2 rotates phase by 2theta and squares magnitude.
        # RCNet naturally rotates. Magnitude squaring needs a non-linear amplitude function.

    def activation(self, z):
        # z * sigmoid(|z|) allows non-linear amplitude scaling
        # Standard Complex ReLU: z if angle in [0, pi/2] etc is bad for rotation.
        # We use Modulus activation: retain phase, warp amplitude.
        mag = torch.abs(z)
        phase = torch.angle(z)
        
        # Allow the network to sharpen the magnitude (approximate squaring)
        new_mag = F.softplus(mag - 0.5) 
        
        # Reconstruct
        return torch.polar(new_mag, phase)

    def forward(self, z, c):
        # Stack inputs: [Batch, 2] complex
        x = torch.stack([z, c], dim=1)
        
        h = self.layer1(x)
        h = self.activation(h)
        
        h = self.layer2(h)
        h = self.activation(h)
        
        out = self.layer3(h)
        return out.squeeze(1)

import torch.nn.functional as F

# ============================================================================
# 2. TRAINING THE LAWS OF PHYSICS
# ============================================================================
def get_physics_batch(batch_size=4096, device='cuda'):
    # Generate RANDOM complex numbers.
    # The network never sees the Mandelbrot set. It only sees random points.
    
    # z: Random state inside [-2, 2] box
    z_real = torch.randn(batch_size, device=device) * 2
    z_imag = torch.randn(batch_size, device=device) * 2
    z = torch.complex(z_real, z_imag)
    
    # c: Random constant inside [-2, 2] box
    c_real = torch.randn(batch_size, device=device) * 2
    c_imag = torch.randn(batch_size, device=device) * 2
    c = torch.complex(c_real, c_imag)
    
    # GROUND TRUTH: The Law of the Fractal
    # z_next = z^2 + c
    target = (z * z) + c
    
    return z, c, target

def train_physics_engine(device):
    print("🌌 Booting Universal Simulator...")
    model = ResonantNetwork(width=256).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    
    # We define a custom complex MSE loss
    def complex_mse(pred, target):
        return torch.mean(torch.abs(pred - target)**2)

    start = time.time()
    for step in range(5000):
        model.train()
        opt.zero_grad()
        
        z, c, target = get_physics_batch(device=device)
        
        pred = model(z, c)
        loss = complex_mse(pred, target)
        
        loss.backward()
        opt.step()
        
        if step % 500 == 0:
            print(f"Step {step} | Physics Loss: {loss.item():.6f}")

    print(f"Physics compiled in {time.time()-start:.2f}s")
    return model

# ============================================================================
# 3. DREAMING THE FRACTAL
# ============================================================================
def render_fractal(model, device, resolution=1000):
    print(f"\n🎨 Dreaming the Mandelbrot Set ({resolution}x{resolution})...")
    model.eval()
    
    # 1. Setup the Complex Plane
    # x: -2.0 to 0.5, y: -1.2 to 1.2
    y, x = np.ogrid[-1.2:1.2:resolution*1j, -2.0:0.8:resolution*1j]
    c_grid_np = x + y*1j
    c_grid = torch.tensor(c_grid_np, dtype=torch.complex64, device=device)
    
    # Flatten for batch processing
    c_flat = c_grid.flatten()
    z_flat = torch.zeros_like(c_flat) # Start at 0
    
    # 2. Iterate the Neural Network Recursively
    # The Network IS the iterator.
    iters = 50
    escape_map = torch.zeros_like(c_flat, dtype=torch.float32)
    
    batch_size = 500000 # Process in chunks to save VRAM
    
    with torch.no_grad():
        for i in range(0, len(c_flat), batch_size):
            end = min(i + batch_size, len(c_flat))
            c_chunk = c_flat[i:end]
            z_chunk = z_flat[i:end]
            mask_chunk = torch.zeros(len(c_chunk), dtype=torch.bool, device=device)
            steps_chunk = torch.zeros(len(c_chunk), dtype=torch.float32, device=device)
            
            for it in range(iters):
                # THE NEURAL UPDATE
                # z = model(z, c) instead of z = z*z + c
                z_chunk = model(z_chunk, c_chunk)
                
                # Check escape (|z| > 2)
                # We use 4.0 as escape radius squared
                abs_z2 = z_chunk.real**2 + z_chunk.imag**2
                escaped = (abs_z2 > 4.0) & (~mask_chunk)
                
                # Record escape time
                steps_chunk[escaped] = it
                mask_chunk |= escaped
                
                # Optimization: Stop processing escaped points (clamp them)
                # z_chunk[mask_chunk] = 0 # Optional stability hack
                
            # Fill remaining with max iters
            steps_chunk[~mask_chunk] = iters
            escape_map[i:end] = steps_chunk
            
            print(f"  Rendered chunk {i//batch_size + 1}/{(len(c_flat)-1)//batch_size + 1}", end='\r')

    # 3. Reshape and Visualize
    img = escape_map.view(resolution, resolution).cpu().numpy()
    
    # 4. Compare with Ground Truth (CPU Math)
    print("\n  Calculating Ground Truth for comparison...")
    c_cpu = c_grid_np
    z_cpu = np.zeros_like(c_cpu)
    truth_map = np.zeros(c_cpu.shape)
    mask = np.full(c_cpu.shape, True, dtype=bool)
    
    for i in range(iters):
        z_cpu[mask] = z_cpu[mask]**2 + c_cpu[mask]
        escaped = np.abs(z_cpu) > 2
        truth_map[mask & escaped] = i
        mask &= ~escaped

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(truth_map, cmap='twilight_shifted', extent=[-2.0, 0.8, -1.2, 1.2])
    axes[0].set_title("Ground Truth (Hard Math)")
    axes[0].axis('off')
    
    axes[1].imshow(img, cmap='twilight_shifted', extent=[-2.0, 0.8, -1.2, 1.2])
    axes[1].set_title("Neural Hallucination (Learned Physics)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("mandelbrot_grok.png")
    print("\n✅ Saved 'mandelbrot_grok.png'.")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_physics_engine(device)
    render_fractal(model, device)