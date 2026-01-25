"""
LBM Fluid Simulation - Animation Generator
Author: Enio Carlos
Output: von_karman.gif
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# --- CLASSE DE SIMULAÇÃO  ---
class WindTunnelLBM:
    def __init__(self, height=100, width=400, reynolds=220.0):
        self.Ny = height
        self.Nx = width
        self.u_max = 0.1
        self.L_obst = height // 5
        self.nu = (self.u_max * self.L_obst) / reynolds
        self.tau = 3.0 * self.nu + 0.5
        
        self.NL = 9
        self.idxs = np.arange(self.NL)
        self.cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # Inicialização com leve ruído
        self.F = np.ones((self.Ny, self.Nx, self.NL)) + 0.01 * np.random.randn(self.Ny, self.Nx, self.NL)
        self.F[:, :, 1] += 2.0 * self.u_max
        self.cylinder = np.full((self.Ny, self.Nx), False)
        self._create_obstacle()

    def _create_obstacle(self):
        y_start = self.Ny // 2 - self.L_obst // 2
        y_end = self.Ny // 2 + self.L_obst // 2
        x_start = self.Nx // 5
        x_end = x_start + self.L_obst
        self.cylinder[y_start:y_end, x_start:x_end] = True

    def step(self):
        # Streaming
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)
        
        # Fronteiras (Bounce-back)
        bndryF = self.F[self.cylinder, :]
        bndryF = bndryF[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
        self.F[self.cylinder, :] = bndryF
        
        # Macroscópicas
        rho = np.sum(self.F, 2)
        ux  = np.sum(self.F * self.cxs, 2) / rho
        uy  = np.sum(self.F * self.cys, 2) / rho
        
        # Entrada/Saída forçada (simples)
        ux[self.cylinder] = 0
        uy[self.cylinder] = 0
        
        # Colisão
        Feq = np.zeros(self.F.shape)
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            cu = 3 * (cx * ux + cy * uy)
            usqr = ux**2 + uy**2
            Feq[:, :, i] = rho * self.weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * usqr)
        
        self.F += -(1.0 / self.tau) * (self.F - Feq)
        return ux, uy

    def get_curl(self, ux, uy):
        duy_dx = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
        dux_dy = np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
        return duy_dx - dux_dy

# --- CONFIGURAÇÃO DA ANIMAÇÃO ---
print(">>> PREPARANDO O ESTÚDIO DE GRAVAÇÃO <<<")

# Parâmetros
width = 400
height = 100
sim = WindTunnelLBM(height=height, width=width, reynolds=150)
total_frames = 300  # Quantos quadros terá o vídeo (300 frames @ 30fps = 10s)
steps_per_frame = 10 # Acelera a simulação (calcula 10 passos para mostrar 1 imagem)

# Configura a Figura
fig, ax = plt.subplots(figsize=(12, 4))
# Inicializa a imagem com zeros
curl_field = np.zeros((height, width))
curl_field[sim.cylinder] = np.nan

# Configuração de cor (Alto Contraste)
im = ax.imshow(curl_field, cmap='seismic', origin='lower', vmin=-0.08, vmax=0.08, animated=True)
ax.contour(sim.cylinder, levels=[0.5], colors='black', linewidths=2)
ax.set_title("Simulação CFD: Vórtices de Von Kármán (Inicializando...)")

def init():
    """Função de inicialização limpa."""
    return im,

def update(frame):
    """ função é chamada repetidamente para criar cada quadro."""
    # Roda a física 'steps_per_frame' vezes antes de desenhar (para o vídeo não ficar lento)
    for _ in range(steps_per_frame):
        ux, uy = sim.step()
    
    # Calcula vorticidade
    curl = sim.get_curl(ux, uy)
    curl[sim.cylinder] = np.nan
    
    # Atualiza a imagem
    im.set_array(curl)
    ax.set_title(f"Simulação CFD: Passo {frame * steps_per_frame}")
    
    # Barra de progresso no terminal
    if frame % 10 == 0:
        print(f"Renderizando quadro {frame}/{total_frames}...")
        
    return im,

# Cria a animação
ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)

print(">>> GRAVANDO VÍDEO (Isso pode levar alguns minutos)...")
# Salva como GIF usando Pillow (não precisa de ffmpeg)
# fps=30 garante fluidez
ani.save('von_karman_simulation.gif', writer='pillow', fps=30)

print("\n✅ VÍDEO PRONTO: 'von_karman_simulation.gif'")
print("Abra o arquivo na pasta para ver o resultado!")