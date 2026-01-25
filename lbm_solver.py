"""
LBM Fluid Simulation - Von Kármán Vortex Street
Author: Enio Carlos
Date: 2026
Description: Simulação 2D de fluxo de fluidos usando o método Lattice Boltzmann (D2Q9).
             O objetivo é visualizar a esteira de vórtices gerada por um obstáculo rígido.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

class WindTunnelLBM:
    def __init__(self, height=100, width=400, reynolds=220.0):
        """
        Inicializa o túnel de vento digital.
        
        Parâmetros:
        height (int): Altura do domínio (Ny)
        width (int): Largura do domínio (Nx)
        reynolds (float): Número de Reynolds desejado (Define a turbulência)
        """
        self.Ny = height
        self.Nx = width
        
        # --- Parâmetros Físicos ---
        # No LBM, a velocidade do som cs^2 é 1/3
        # A viscosidade cinemática 'nu' relaciona-se com o tempo de relaxamento 'tau'
        # Re = (U * L) / nu
        
        self.u_max = 0.1          # Velocidade máxima de entrada (para estabilidade < 0.15)
        self.L_obst = height // 5 # Tamanho característico do obstáculo
        
        # Calculando a viscosidade necessária para atingir o Reynolds desejado
        self.nu = (self.u_max * self.L_obst) / reynolds
        self.tau = 3.0 * self.nu + 0.5
        
        print(f"--- Configuração LBM ---")
        print(f"Reynolds: {reynolds}")
        print(f"Viscosidade (nu): {self.nu:.4f}")
        print(f"Relaxamento (tau): {self.tau:.4f}")
        
        # --- Pesos e Direções D2Q9 ---
        self.NL = 9
        self.idxs = np.arange(self.NL)
        self.cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # --- Inicialização dos Campos ---
        # F: Função distribuição de partículas
        # Adicionamos ruído aleatório para quebrar a simetria e iniciar os vórtices mais rápido
        self.F = np.ones((self.Ny, self.Nx, self.NL)) + 0.01 * np.random.randn(self.Ny, self.Nx, self.NL)
        
        # Ajustando a velocidade inicial para a direita
        self.F[:, :, 1] += 2.0 * self.u_max
        
        # Define o obstáculo
        self.cylinder = np.full((self.Ny, self.Nx), False)
        self._create_obstacle()

    def _create_obstacle(self):
        """Cria um obstáculo quadrado no fluxo."""
        # Posição centralizada no Y, deslocada para esquerda no X
        y_start = self.Ny // 2 - self.L_obst // 2
        y_end = self.Ny // 2 + self.L_obst // 2
        x_start = self.Nx // 5
        x_end = x_start + self.L_obst
        
        self.cylinder[y_start:y_end, x_start:x_end] = True

    def step(self):
        """Executa um passo de tempo (Iteração LBM)."""
        
        # 1. STREAMING (Advecção)
        # Movemos as partículas para as células vizinhas
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)
        
        # 2. CONDIÇÕES DE CONTORNO (Boundary Conditions)
        # A. Paredes Sólidas (Bounce-Back no obstáculo)
        bndryF = self.F[self.cylinder, :]
        # Inverte as direções: o que ia pra direita (1) volta pra esquerda (3), etc.
        bndryF = bndryF[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]
        self.F[self.cylinder, :] = bndryF
        
        # 3. CÁLCULO DAS VARIÁVEIS MACROSCÓPICAS
        rho = np.sum(self.F, 2)
        ux  = np.sum(self.F * self.cxs, 2) / rho
        uy  = np.sum(self.F * self.cys, 2) / rho
        
        # Força condicao de entrada (Inflow) constante à esquerda
        # e saída livre (Outflow) à direita é assumida pelo loop periódico do np.roll,
        # mas podemos forçar a velocidade no início para estabilizar
        ux[self.cylinder] = 0
        uy[self.cylinder] = 0
        
        # 4. COLISÃO (Relaxamento BGK)
        # Calcula a Distribuição de Equilíbrio (Feq)
        Feq = np.zeros(self.F.shape)
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            cu = 3 * (cx * ux + cy * uy)
            usqr = ux**2 + uy**2
            Feq[:, :, i] = rho * self.weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * usqr)
        
        # Aplica o relaxamento
        self.F += -(1.0 / self.tau) * (self.F - Feq)
        
        return ux, uy

    def get_curl(self, ux, uy):
        """Calcula a Vorticidade (Curl) para visualização dos redemoinhos."""
        # Curl = d(uy)/dx - d(ux)/dy
        duy_dx = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
        dux_dy = np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
        return duy_dx - dux_dy

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Configuração
    sim = WindTunnelLBM(height=100, width=400, reynolds=150)
    TOTAL_STEPS = 3000
    
    print(f"Iniciando simulação de {TOTAL_STEPS} passos...")
    
    # Loop de Simulação
    # Vamos pular os primeiros passos para chegar logo no regime estacionário
    for t in range(TOTAL_STEPS):
        ux, uy = sim.step()
        
        if t % 500 == 0:
            print(f"Passo {t}/{TOTAL_STEPS}")

    # --- PLOTAGEM FINAL ---
    # --- 5. VISUALIZAÇÃO DE ALTO CONTRASTE (EXCELÊNCIA) ---
    print("Gerando visualização de Vorticidade...")
    
    # Recalcula o Curl final
    curl = sim.get_curl(ux, uy)
    curl[sim.cylinder] = np.nan
    
    # --- TRUQUE DE ENGENHARIA: Ajuste Automático de Contraste ---
    # Pegamos o valor máximo de giro, mas ignoramos picos de ruído (usando percentil)
    # Isso garante que a cor não fique "estourada" por causa de um único pixel errado
    max_val = np.nanpercentile(np.abs(curl), 98) 
    print(f"Vorticidade típica detectada: {max_val:.4f}")
    
    # Definimos o limite para saturar um pouco e deixar as cores vivas
    limit = max_val * 0.8  
    
    plt.figure(figsize=(14, 5)) # Imagem mais larga (Cinemática)
    
    # cmap='seismic': Azul Escuro (Giro -) <-> Branco <-> Vermelho Escuro (Giro +)
    plt.imshow(curl, cmap='seismic', origin='lower', vmin=-limit, vmax=limit)
    
    cbar = plt.colorbar()
    cbar.set_label('Vorticidade (1/s)', rotation=270, labelpad=15)
    
    plt.title(f'Análise CFD: Esteira de Von Kármán (Re={150})', fontsize=14, fontweight='bold')
    plt.xlabel('Domínio Longitudinal (x)', fontsize=12)
    plt.ylabel('Domínio Transversal (y)', fontsize=12)
    
    # Desenha o Prédio (Obstáculo) com contorno forte
    plt.contour(sim.cylinder, levels=[0.5], colors='black', linewidths=3)
    
    plt.tight_layout()
    plt.savefig('analise_cfd_final_HD.png', dpi=300) # Salva em Alta Definição
    print("✅ Imagem de Alta Definição salva: 'analise_cfd_final_HD.png'")
    plt.show()