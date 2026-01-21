"""
THE NEURAL PAINTER
==================
A Real-Time GUI to watch RCNet learn the geometry of an image.

1. Load Image.
2. RCNet learns the mapping (x, y) -> (r, g, b) via Phase Interference.
3. Watch it evolve from chaos to order.
4. "Dream" button to extrapolate beyond the canvas.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time

# ============================================================================
# 1. THE RESONANT BRAIN (Coordinate Network)
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.2)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.2)

    def forward(self, z):
        real = self.fc_r(z.real) - self.fc_i(z.imag) + self.bias_r
        imag = self.fc_r(z.imag) + self.fc_i(z.real) + self.bias_i
        return torch.complex(real, imag)

class RCNet_Painter(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        # Input: (x, y) complex coordinate
        self.layer1 = ComplexLinear(1, width)
        self.layer2 = ComplexLinear(width, width)
        self.layer3 = ComplexLinear(width, width)
        self.layer4 = ComplexLinear(width, width) # Deep for detail
        self.out = ComplexLinear(width, 3) # RGB (Real part)

    def forward(self, z):
        # Sine Activation (Periodic, good for images)
        z = self.layer1(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        
        z = self.layer2(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        
        z = self.layer3(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        
        z = self.layer4(z)
        z = torch.sin(z.real) + 1j*torch.sin(z.imag)
        
        out = self.out(z)
        return torch.sigmoid(out.real) # RGB 0-1

# ============================================================================
# 2. THE GUI
# ============================================================================
class PainterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Painter (RCNet)")
        self.master.geometry("1000x600")
        self.master.configure(bg="#222")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_training = False
        self.model = None
        self.target_img = None
        self.coords = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Left Panel (Controls)
        self.panel_left = tk.Frame(self.master, width=200, bg="#333")
        self.panel_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        btn_style = {"bg": "#444", "fg": "white", "font": ("Arial", 12), "relief": "flat", "width": 15}
        
        tk.Label(self.panel_left, text="NEURAL PAINTER", bg="#333", fg="#0f0", font=("Arial", 14, "bold")).pack(pady=20)
        
        self.btn_load = tk.Button(self.panel_left, text="Load Image", command=self.load_image, **btn_style)
        self.btn_load.pack(pady=10)
        
        self.btn_train = tk.Button(self.panel_left, text="Start Dreaming", command=self.toggle_train, state='disabled', **btn_style)
        self.btn_train.pack(pady=10)
        
        self.lbl_loss = tk.Label(self.panel_left, text="Loss: ---", bg="#333", fg="gray")
        self.lbl_loss.pack(pady=20)
        
        # Right Panel (Canvas)
        self.panel_right = tk.Frame(self.master, bg="#222")
        self.panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Two Canvases: Target vs Dream
        self.lbl_target = tk.Label(self.panel_right, text="Reality", bg="#222", fg="white")
        self.lbl_target.grid(row=0, column=0)
        self.canvas_target = tk.Label(self.panel_right, bg="black")
        self.canvas_target.grid(row=1, column=0, padx=10)
        
        self.lbl_dream = tk.Label(self.panel_right, text="Neural Hallucination", bg="#222", fg="white")
        self.lbl_dream.grid(row=0, column=1)
        self.canvas_dream = tk.Label(self.panel_right, bg="black")
        self.canvas_dream.grid(row=1, column=1, padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path:
            # Load and Resize
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            
            self.target_img = torch.tensor(img / 255.0, dtype=torch.float32).to(self.device)
            
            # Display Target
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.canvas_target.config(image=img_tk)
            self.canvas_target.image = img_tk
            
            # Init Model
            self.model = RCNet_Painter(width=128).to(self.device)
            self.opt = optim.Adam(self.model.parameters(), lr=0.01)
            
            # Init Coordinates (Complex Plane)
            # x + iy
            y, x = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
            z = x + 1j*y
            self.coords = torch.tensor(z, dtype=torch.cfloat).unsqueeze(-1).to(self.device) # [H, W, 1]
            
            self.btn_train.config(state='normal', bg="#005500")

    def toggle_train(self):
        if self.is_training:
            self.is_training = False
            self.btn_train.config(text="Resume Dreaming", bg="#005500")
        else:
            self.is_training = True
            self.btn_train.config(text="Stop Dreaming", bg="#550000")
            Thread(target=self.train_loop, daemon=True).start()

    def train_loop(self):
        while self.is_training:
            self.model.train()
            self.opt.zero_grad()
            
            # Forward Pass (Predict RGB from Coordinates)
            pred = self.model(self.coords)
            
            # Loss (Compare to Pixel)
            loss = nn.MSELoss()(pred, self.target_img)
            
            loss.backward()
            self.opt.step()
            
            # Update UI (Every few steps)
            if np.random.rand() < 0.2:
                loss_val = loss.item()
                self.lbl_loss.config(text=f"Loss: {loss_val:.5f}")
                
                # Render Dream
                with torch.no_grad():
                    dream = pred.cpu().numpy()
                    dream = (np.clip(dream, 0, 1) * 255).astype(np.uint8)
                    dream_pil = Image.fromarray(dream)
                    dream_tk = ImageTk.PhotoImage(dream_pil)
                    
                    # Update in main thread
                    self.master.after(0, lambda img=dream_tk: self.update_canvas(img))
            
            time.sleep(0.01)

    def update_canvas(self, img):
        self.canvas_dream.config(image=img)
        self.canvas_dream.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = PainterApp(root)
    root.mainloop()