"""
RESONANT CORTEX (RCNet)
=======================
A Self-Organizing Complex-Valued Neural Network.

Core Physics:
1. Neurons are Oscillators (Complex Numbers).
2. Computation is Wave Interference.
3. Architecture is Dynamic: It grows new columns when frustrated.

Hardware: Optimized for RTX 3060 (12GB VRAM).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import time

# ============================================================================
# 1. THE RESONANT NEURON (Complex Valued Layer)
# ============================================================================
class ComplexLinear(nn.Module):
    """
    A linear layer that operates on complex numbers.
    Simulates interference patterns between neurons.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # We model complex weights as two real matrices for explicit control
        # (Real * Real) - (Imag * Imag) = Real Part
        # (Real * Imag) + (Imag * Real) = Imag Part
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        
        # Initialize phase randomly (0 to 2pi)
        nn.init.xavier_uniform_(self.fc_r.weight)
        nn.init.xavier_uniform_(self.fc_i.weight)

    def forward(self, z):
        # z is a complex tensor
        r = z.real
        i = z.imag
        
        # Complex Multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_out = self.fc_r(r) - self.fc_i(i)
        imag_out = self.fc_r(i) + self.fc_i(r)
        
        return torch.complex(real_out, imag_out)

class ResonantColumn(nn.Module):
    """
    A single 'cortical column' of oscillators.
    Non-linearity is applied to Amplitude (modulus) only, preserving Phase info.
    """
    def __init__(self, width):
        super().__init__()
        self.mixer = ComplexLinear(width, width)
        # Learnable natural frequency for this column
        self.bias_phase = nn.Parameter(torch.randn(width) * 0.1)

    def forward(self, z):
        # 1. Linear Mixing (Interference)
        z_mixed = self.mixer(z)
        
        # 2. Resonant Activation (The "Non-Linearity")
        # We act on the Phase and Amplitude separately
        
        # Amplitude: Sigmoid/Tanh (Firing Rate)
        amp = torch.abs(z_mixed)
        amp = torch.tanh(amp) # Saturate amplitude
        
        # Phase: Rotate by natural frequency (The "Logic")
        # We normalize phase to keep it purely rotational
        phase = torch.angle(z_mixed) + self.bias_phase
        
        # Reconstruct
        return torch.polar(amp, phase)

# ============================================================================
# 2. THE GROWING BRAIN (RCNet)
# ============================================================================
class RCNet(nn.Module):
    def __init__(self, vocab_size=97, dim=128):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Input Embedding: Map numbers to Frequencies
        # We learn an embedding, then treat it as a complex vector
        self.embedding = nn.Embedding(vocab_size, dim * 2) # Real + Imag parts
        
        # The Cortex: A list of Resonant Columns
        # We start with 1 column. We will grow more.
        self.columns = nn.ModuleList([ResonantColumn(dim)])
        
        # Readout: Map complex resonance back to probabilities
        self.readout = nn.Linear(dim * 2, vocab_size)

    def forward(self, x):
        # x: [Batch, 2] (Two numbers to add)
        batch_size = x.shape[0]
        
        # 1. Embed Inputs
        # We sum the embeddings of the two inputs to create the initial "Wave"
        emb_a = self.embedding(x[:, 0])
        emb_b = self.embedding(x[:, 1])
        
        # Superposition of inputs
        combined = emb_a + emb_b 
        
        # Convert to Complex: First half real, second half imaginary
        z = torch.complex(combined[:, :self.dim], combined[:, self.dim:])
        
        # 2. Propagate through Resonant Columns
        # It's a ResNet-style skip connection architecture (Wave Propagation)
        for col in self.columns:
            # Interference: New wave adds to existing wave
            z_new = col(z)
            z = z + z_new # Constructive/Destructive Interference
            
        # 3. Readout
        # Convert back to real for classification
        # We use both Magnitude and Phase info for the final decision
        cat = torch.cat([z.real, z.imag], dim=1)
        logits = self.readout(cat)
        
        return logits, z # Return z for visualization

    def grow(self):
        """Physically adds a new layer to the network."""
        new_col = ResonantColumn(self.dim).to(next(self.parameters()).device)
        self.columns.append(new_col)
        return len(self.columns)

# ============================================================================
# 3. THE LAB (Training & Physics)
# ============================================================================
def get_beta(activations):
    """
    Measures the 'Roughness' or 'Complexity' of the wave state.
    High Beta = Complex, differentiated patterns (Crystals).
    Low Beta = Smooth, uniform patterns (Sludge).
    """
    # Simple metric: Standard deviation of the phase angles
    # If phases are locked (low entropy), this might be distinct.
    # If phases are random noise, this is high.
    # We want "Structured Complexity".
    
    # Let's use the variance of the Amplitudes as a proxy for "Selection"
    # If the network has decided on a path, some amps should be 1, others 0.
    amp = torch.abs(activations)
    return torch.std(amp).item()

def train_resonant_cortex():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧠 Booting Resonant Cortex on {device}...")
    
    P = 97 # Prime number for modular arithmetic
    dim = 128 # Hidden dimension
    model = RCNet(vocab_size=P, dim=dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    # 2. Data (Full Table)
    X = []
    Y = []
    for i in range(P):
        for j in range(P):
            X.append([i, j])
            Y.append((i + j) % P)
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    
    # 3. Training Loop with Growth
    history = {'loss': [], 'acc': [], 'beta': [], 'size': []}
    loss_window = deque(maxlen=200)
    
    start_time = time.time()
    
    print("\n--- PHASE 1: RESONANCE TUNING ---")
    print("Goal: Learn (a + b) % 97")
    print("Mechanism: Auto-Growth upon Frustration\n")
    
    growth_cooldown = 0
    
    for epoch in range(5001): # 5000 Epochs
        model.train()
        
        # Full batch gradient descent (for "Physics" stability)
        # Or massive batches. X is small (97*97 = 9409), fits in VRAM easily.
        opt.zero_grad()
        logits, wave_state = model(X)
        loss = crit(logits, Y)
        loss.backward()
        opt.step()
        
        # Metrics
        acc = (logits.argmax(dim=1) == Y).float().mean().item()
        beta = get_beta(wave_state)
        
        loss_val = loss.item()
        loss_window.append(loss_val)
        
        history['loss'].append(loss_val)
        history['acc'].append(acc)
        history['beta'].append(beta)
        history['size'].append(len(model.columns))
        
        # --- THE SIEVE (Growth Logic) ---
        # If we are stuck (loss not improving) AND not perfect yet
        if epoch > 500 and growth_cooldown == 0 and acc < 0.99:
            # Check for plateau
            recent_avg = np.mean(list(loss_window)[-100:])
            older_avg = np.mean(list(loss_window)[:100])
            
            # If improvement is tiny (< 1%)
            if older_avg - recent_avg < 0.01:
                print(f"⚠️  FRUSTRATION DETECTED (Loss stuck at {loss_val:.4f})")
                new_size = model.grow()
                
                # IMPORTANT: Add new parameters to optimizer
                opt = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
                
                print(f"🌱  GROWTH TRIGGERED: Cortex Expanded to {new_size} Columns.")
                growth_cooldown = 500 # Give it time to tune the new column
                
                # Little shake to break symmetry
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(torch.randn_like(p) * 0.01)
        
        if growth_cooldown > 0:
            growth_cooldown -= 1
            
        if epoch % 500 == 0:
            print(f"Ep {epoch:<5} | Loss: {loss_val:.4f} | Acc: {acc:.1%} | β (Cryst): {beta:.3f} | Size: {len(model.columns)}")
            
        if acc == 1.0:
            print(f"\n🚀  GROKKED at Epoch {epoch}! Perfect Accuracy.")
            break

    total_time = time.time() - start_time
    print(f"Training Complete in {total_time:.2f}s")
    
    # =================================================================       
    # 4. VISUALIZATION
    # =================================================================
    print("\nVisualizing Resonance...")
    
    plt.figure(figsize=(15, 5))
    
    # Accuracy & Size
    plt.subplot(1, 3, 1)
    plt.plot(history['acc'], label='Accuracy', color='green')
    ax2 = plt.gca().twinx()
    ax2.plot(history['size'], label='Network Size', color='blue', linestyle='--')
    plt.title('Performance & Growth')
    plt.xlabel('Epoch')
    
    # Beta (Crystallization)
    plt.subplot(1, 3, 2)
    plt.plot(history['beta'], color='purple')
    plt.title('Beta (Structural Crystallization)')
    plt.xlabel('Epoch')
    
    # Phase Heatmap (The "Brain Scan")
    plt.subplot(1, 3, 3)
    # Visualize the phases of the final wave state for the first 100 inputs
    # This shows if the network "locked" onto specific frequencies
    phases = torch.angle(wave_state[:100, :]).detach().cpu().numpy()
    sns.heatmap(phases, cmap='twilight', cbar=True)
    plt.title('Phase Alignment (First 100 Samples)')
    plt.xlabel('Neuron Dimension')
    plt.ylabel('Sample Index')
    
    plt.tight_layout()
    plt.savefig('resonant_cortex_results.png')
    print("Saved 'resonant_cortex_results.png'")

if __name__ == '__main__':
    train_resonant_cortex()