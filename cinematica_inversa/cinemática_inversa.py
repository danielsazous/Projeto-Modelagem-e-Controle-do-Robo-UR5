import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERRO] Instale com: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

plt.style.use("seaborn-v0_8-whitegrid")

PASTA_SAIDA = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(PASTA_SAIDA, exist_ok=True)

JOINT_NAMES = [f"/UR5_joint{i}" for i in range(1, 7)]
EEF_NAME    = "/UR5_connection"
BASE_NAME   = "/UR5"

CHAIN_ZERO = [
    ([0.0,      0.0,      0.0085  ], [0.0,      0.0,       0.0,      1.0     ]),
    ([-0.070317,-1.4e-05, 0.066039], [6.8e-05, -0.707159,  6.8e-05,  0.707054]),
    ([0.425105, -9.1e-05, 1.9e-05 ], [-0.0,    -1.5e-05,  -0.000107, 1.0     ]),
    ([0.392149, -7.6e-05, 2e-06   ], [-0.0,    -0.0,      -9.7e-05,  1.0     ]),
    ([0.045573, -1e-05,   0.03971 ], [7.5e-05,  0.707098, -7.5e-05,  0.707116]),
    ([0.014424, -4e-06,   0.049176], [-8.5e-05,-0.707107, -8.5e-05,  0.707107]),
]
EEF_ZERO = ([-0.0, -0.0, 0.089375], [-0.0, -0.0, -9.7e-05, 1.0])

# 10 configurações distintas das do código original
TEST_CONFIGS_DEG = [
    [ 20,  -70,   80,  -50,  -80,   20],
    [-20,  -50,   70,  -40,  -70,  -20],
    [ 50,  -40,   55,  -70,  -55,   50],
    [-50,  -70,   65,  -55,  -65,  -50],
    [ 75,  -55,   40,  -75,  -40,   75],
    [-75,  -40,   50,  -65,  -50,  -75],
    [ 35,  -65,   85,  -35,  -85,   35],
    [-35,  -80,   35,  -50,  -35,  -35],
    [ 55,  -25,   25,  -80,  -25,   55],
    [-55,  -35,   75,  -25,  -75,  -55],
]


# ── Funções matemáticas auxiliares ───────────────────────────────────────────

def _quat_to_rot(q):
    qx,qy,qz,qw = q[0],q[1],q[2],q[3]
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])

def _Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0,0]=c; T[0,1]=-s; T[1,0]=s; T[1,1]=c
    return T

def _chain_T(pos, quat):
    T = np.eye(4)
    T[:3,:3] = _quat_to_rot(quat)
    T[:3, 3] = pos
    return T

def _rot_to_rotvec(R):
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if theta < 1e-10:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        diag = np.array([R[0,0]+1, R[1,1]+1, R[2,2]+1]) / 2
        axis = np.sqrt(np.maximum(diag, 0))
        axis[0] *= np.sign(R[0,1] + R[1,0] + 1e-10)
        axis[1] *= np.sign(R[0,2] + R[2,0] + 1e-10)
        axis /= (np.linalg.norm(axis) + 1e-10)
        return theta * axis
    return theta/(2*np.sin(theta)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


def cinematica_direta_cop(thetas):
    T = np.eye(4)
    for i, (pos, quat) in enumerate(CHAIN_ZERO):
        T = T @ _chain_T(pos, quat) @ _Rz(thetas[i])
    return T @ _chain_T(*EEF_ZERO)


def _jacobiano_numerico(thetas, delta=1e-6):
    T0 = cinematica_direta_cop(thetas)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]
    J  = np.zeros((6, 6))
    for i in range(6):
        t_plus = thetas.copy()
        t_plus[i] += delta
        T_plus = cinematica_direta_cop(t_plus)
        J[:3, i] = (T_plus[:3, 3] - p0) / delta
        J[3:, i] = _rot_to_rotvec(T_plus[:3,:3] @ R0.T) / delta
    return J

def _ik_uma_solucao(T_alvo, theta_init, max_iter=300, tol=1e-8):
    thetas = theta_init.copy()
    p_alvo = T_alvo[:3, 3]
    R_alvo = T_alvo[:3, :3]
    for it in range(max_iter):
        T_atual = cinematica_direta_cop(thetas)
        ep = p_alvo - T_atual[:3, 3]
        eo = _rot_to_rotvec(R_alvo @ T_atual[:3,:3].T)
        erro = np.concatenate([ep, eo])
        if np.linalg.norm(erro) < tol: break
        J = _jacobiano_numerico(thetas)
        thetas = thetas + 0.3 * np.linalg.pinv(J) @ erro
        thetas = (thetas + np.pi) % (2*np.pi) - np.pi
    T_f = cinematica_direta_cop(thetas)
    ep_f = np.linalg.norm(T_f[:3,3] - p_alvo) * 1000
    eo_f = np.linalg.norm(_rot_to_rotvec(R_alvo @ T_f[:3,:3].T))
    return thetas, it+1, ep_f, eo_f

def _solucao_ja_existe(nova, solucoes, tol_ang=0.15):
    for s in solucoes:
        if np.linalg.norm(nova - s) < tol_ang:
            return True
    return False

_CONFIGS_INICIAIS = np.radians([
    [   0,  -45,   90,    0,   90,   0],
    [   0,  -45,   90,    0,  -90,   0],
    [   0,  -90,  -90,    0,   90,   0],
    [   0,  -90,  -90,    0,  -90,   0],
    [ 180,   45,   90,    0,   90,   0],
    [ 180,   45,   90,    0,  -90,   0],
    [ 180,   90,  -90,    0,   90,   0],
    [ 180,   90,  -90,    0,  -90,   0],
    [  90,  -60,   60,  -30,   60,   0],
    [ -90,  -60,   60,  -30,  -60,   0],
    [  45,  -30,   45,   45,   45,   0],
    [ -45,  -30,   45,  -45,  -45,   0],
])

def cinematica_inversa_numerica(T_alvo, theta_ref=None, max_iter=200, tol=1e-6):
    if theta_ref is None:
        theta_ref = np.zeros(6)

    pontos = list(_CONFIGS_INICIAIS) + [theta_ref]
    solucoes = []
    iters_total = 0
    for p0 in pontos:
        sol, it, ep, eo_f = _ik_uma_solucao(T_alvo, p0, max_iter, tol)
        iters_total += it
        if ep < 0.5 and eo_f < 0.05 and not _solucao_ja_existe(sol, solucoes):
            solucoes.append(sol)

    if not solucoes:
        return [], np.zeros(6), 0, iters_total

    diffs  = [np.linalg.norm(s - theta_ref) for s in solucoes]
    melhor = solucoes[int(np.argmin(diffs))]
    return solucoes, melhor, len(solucoes), iters_total


# Interface CoppeliaSim

class UR5CoppeliaInterface:

    def __init__(self, host='localhost', port=23000):
        print(f"[INFO] Conectando ao CoppeliaSim em {host}:{port}...")
        self.client = RemoteAPIClient(host=host, port=port)
        self.sim    = self.client.require('sim')
        print("[INFO] Conexão estabelecida!")
        self._resolve_handles()

    def _resolve_handles(self):
        self.joint_handles = [self.sim.getObject(n) for n in JOINT_NAMES]
        self.eef_handle    = self.sim.getObject(EEF_NAME)
        self.base_handle   = self.sim.getObject(BASE_NAME)
        print("[INFO] Handles resolvidos.")

    def set_joint_angles(self, thetas_rad):
        for h, theta in zip(self.joint_handles, thetas_rad):
            self.sim.setJointPosition(h, float(theta))
        time.sleep(0.05)

    def get_eef_pose(self):
        pos = np.array(self.sim.getObjectPosition(self.eef_handle, self.base_handle))
        q   = self.sim.getObjectQuaternion(self.eef_handle, self.base_handle)
        R   = _quat_to_rot(q)
        T   = np.eye(4); T[:3,:3]=R; T[:3,3]=pos
        return pos, R, T

    def coletar_ground_truth(self, configs_deg=None):
        if configs_deg is None:
            configs_deg = TEST_CONFIGS_DEG
        N = len(configs_deg)
        thetas_all = np.radians(configs_deg)
        pos_gt = np.zeros((N,3)); R_gt = np.zeros((N,3,3)); T_gt = []

        print(f"\n[INFO] Coletando ground truth — {N} configurações...")
        for i, thetas in enumerate(thetas_all):
            self.set_joint_angles(thetas)
            pos, R, T = self.get_eef_pose()
            pos_gt[i]=pos; R_gt[i]=R; T_gt.append(T)
            print(f"  [{i+1:2d}/{N}] θ={np.degrees(thetas).round(1)} °  "
                  f"→  pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) m")

        print("[INFO] Coleta concluída!\n")
        return {'thetas':thetas_all, 'pos_gt':pos_gt, 'R_gt':R_gt, 'T_gt':T_gt}


def _rot_to_euler_xyz(R):
    """Converte matriz de rotação para ângulos de Euler (Roll, Pitch, Yaw) em graus.
    Usada apenas para visualização nos gráficos de orientação."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.isclose(np.cos(pitch), 0.0, atol=1e-8):
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


# Validação da Cinemática Inversa

def calcular_erros_inversa(gt, interface):
    N = len(gt['thetas'])
    ep    = np.zeros(N); eo    = np.zeros(N)
    ep_x  = np.zeros(N); ep_y  = np.zeros(N); ep_z = np.zeros(N)
    n_sol = np.zeros(N, dtype=int)
    thetas_ik = np.zeros((N, 6))
    pos_ik_all = np.zeros((N, 3))
    R_ik_all   = np.zeros((N, 3, 3))

    print(f"\n[INFO] Validando Cinemática Inversa — {N} configurações...")
    iters_total_geral = 0
    for i in range(N):
        T_alvo      = gt['T_gt'][i]
        thetas_orig = gt['thetas'][i]

        solucoes, melhor, n_sol_i, iters_config = cinematica_inversa_numerica(
            T_alvo, theta_ref=thetas_orig
        )
        iters_total_geral += iters_config
        n_sol[i]     = n_sol_i
        thetas_ik[i] = melhor

        interface.set_joint_angles(melhor)
        pos_ik, R_ik, _ = interface.get_eef_pose()

        pos_ik_all[i] = pos_ik
        R_ik_all[i]   = R_ik

        diff   = pos_ik - gt['pos_gt'][i]
        ep[i]  = np.linalg.norm(diff)
        ep_x[i] = abs(diff[0]); ep_y[i] = abs(diff[1]); ep_z[i] = abs(diff[2])
        eo[i]  = np.linalg.norm(R_ik - gt['R_gt'][i], 'fro')

        print(f"  [{i+1:2d}/{N}] {n_sol_i} soluções  iters={iters_config}  "
              f"→  ΔPos={ep[i]*1000:.4f} mm  ΔOri={eo[i]:.2e}")

    print(f"\n[INFO] Total de iterações Newton-Raphson: {iters_total_geral}")

    return ep, ep_x, ep_y, ep_z, eo, n_sol, thetas_ik, pos_ik_all, R_ik_all


# =============================================================================
# GRÁFICOS
# =============================================================================

def grafico_curvas_juntas(gt, caminho):
    """Gráfico 1: Ângulos das juntas nas configurações de teste."""
    n   = len(gt['thetas'])
    idx = np.arange(1, n + 1)
    juntas_deg = np.degrees(gt['thetas'])

    fig, ax = plt.subplots(figsize=(10, 6))
    marcadores = ["o", "s", "^", "D", "v", "P"]
    for j in range(6):
        ax.plot(idx, juntas_deg[:, j],
                marker=marcadores[j], label=f"Junta {j+1}",
                linewidth=2.2, markersize=7)

    ax.set_title("Configurações de Teste — Ângulos das Juntas",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Índice da Configuração de Teste", fontsize=12)
    ax.set_ylabel("Ângulo da Junta (graus)", fontsize=12)
    ax.set_xticks(idx)
    ax.legend(ncol=3, fontsize=10, loc='best')
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 1 (Juntas) salvo → {caminho}")


def grafico_curvas_posicao(gt, pos_ik_all, caminho):
    """Gráfico 2: Comparação de posição Ground Truth vs IK."""
    n   = len(gt['thetas'])
    idx = np.arange(1, n + 1)

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    eixos = ["X", "Y", "Z"]
    cores = ["tab:blue", "tab:green", "tab:red"]

    for k in range(3):
        axs[k].plot(idx, gt['pos_gt'][:, k], "o-",
                    color=cores[k], linewidth=2.2, markersize=6,
                    label="Ground Truth (CoppeliaSim)")
        axs[k].plot(idx, pos_ik_all[:, k], "x--",
                    color="black", linewidth=2.2, markersize=7,
                    label="Cinemática Inversa (Python)")
        axs[k].set_ylabel(f"Posição {eixos[k]} (m)", fontsize=12)
        axs[k].legend(fontsize=11)
        axs[k].grid(True, linestyle=":", alpha=0.8)

    axs[-1].set_xlabel("Índice da Configuração de Teste", fontsize=12)
    axs[-1].set_xticks(idx)
    fig.suptitle("Comparação de Posição do Efetuador\nGround Truth × Cinemática Inversa",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 2 (Posição) salvo → {caminho}")


def grafico_curvas_orientacao(gt, R_ik_all, caminho):
    """Gráfico 3: Comparação de orientação (Euler) Ground Truth vs IK."""
    n   = len(gt['thetas'])
    idx = np.arange(1, n + 1)

    euler_gt = np.array([_rot_to_euler_xyz(gt['R_gt'][i]) for i in range(n)])
    euler_ik = np.array([_rot_to_euler_xyz(R_ik_all[i])   for i in range(n)])

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    nomes = ["Roll", "Pitch", "Yaw"]
    cores = ["tab:purple", "tab:orange", "tab:cyan"]

    for k in range(3):
        axs[k].plot(idx, euler_gt[:, k], "o-",
                    color=cores[k], linewidth=2.2, markersize=6,
                    label="Ground Truth (CoppeliaSim)")
        axs[k].plot(idx, euler_ik[:, k], "x--",
                    color="black", linewidth=2.2, markersize=7,
                    label="Cinemática Inversa (Python)")
        axs[k].set_ylabel(f"{nomes[k]} (°)", fontsize=12)
        axs[k].legend(fontsize=11)
        axs[k].grid(True, linestyle=":", alpha=0.8)

    axs[-1].set_xlabel("Índice da Configuração de Teste", fontsize=12)
    axs[-1].set_xticks(idx)
    fig.suptitle("Comparação de Orientação do Efetuador\n(Ground Truth × Cinemática Inversa)",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 3 (Orientação) salvo → {caminho}")


def grafico_erros(ep, eo, caminho):
    """Gráfico 4: Erros totais de posição e orientação."""
    n   = len(ep)
    idx = np.arange(1, n + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    axs[0].plot(idx, ep * 1000, "o-", color="firebrick", linewidth=2, markersize=7)
    axs[0].axhline(np.mean(ep) * 1000, color="gray", linestyle="--", label="Média")
    axs[0].set_title("Erro de Posição", fontsize=13, fontweight='bold')
    axs[0].set_xlabel("Configuração de Teste")
    axs[0].set_ylabel("Erro (mm)")
    axs[0].set_xticks(idx)
    axs[0].legend(fontsize=11)
    axs[0].grid(True, linestyle=":")

    axs[1].plot(idx, eo, "s-", color="darkslateblue", linewidth=2, markersize=7)
    axs[1].axhline(np.mean(eo), color="gray", linestyle="--", label="Média")
    axs[1].set_title("Erro de Orientação", fontsize=13, fontweight='bold')
    axs[1].set_xlabel("Configuração de Teste")
    axs[1].set_ylabel("Erro (Norma Frobenius)")
    axs[1].set_xticks(idx)
    axs[1].legend(fontsize=11)
    axs[1].grid(True, linestyle=":")

    fig.suptitle("Validação Numérica — Erros entre Cinemática Inversa e Ground Truth",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 4 (Erros) salvo → {caminho}")


def grafico_dispersao_3d(gt, pos_ik_all, caminho):
    """Gráfico 5: Dispersão 3D das posições Ground Truth vs IK."""
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")

    pg, pc = gt['pos_gt'], pos_ik_all

    ax.scatter(*pg.T, c="tab:blue", s=80,
               label="Ground Truth (CoppeliaSim)", depthshade=False)
    ax.scatter(*pc.T, c="tab:red", s=50, marker="^",
               label="Cinemática Inversa (Python)", depthshade=False)
    for a, b in zip(pg, pc):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="gray", linewidth=0.8, alpha=0.5)

    ax.set_title("Dispersão 3D das Posições do Efetuador",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 5 (3D) salvo → {caminho}")


def grafico_erros_por_eixo(ep_x, ep_y, ep_z, caminho):
    """Gráfico 6: Erro de posição separado por eixo X, Y, Z."""
    n   = len(ep_x)
    idx = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    ax.bar(idx - width, ep_x * 1000, width=width, label='Erro X (mm)', color='tab:blue',  alpha=0.9)
    ax.bar(idx,         ep_y * 1000, width=width, label='Erro Y (mm)', color='tab:green', alpha=0.9)
    ax.bar(idx + width, ep_z * 1000, width=width, label='Erro Z (mm)', color='tab:red',   alpha=0.9)

    ax.set_title("Erro de Posição por Eixo (X, Y, Z)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Configuração de Teste")
    ax.set_ylabel("Erro (mm)")
    ax.set_xticks(idx)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 6 (Erros por Eixo) salvo → {caminho}")


def grafico_num_solucoes(n_sol, caminho):
    """Gráfico 7: Número de soluções encontradas por configuração."""
    n   = len(n_sol)
    idx = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(idx, n_sol, color='mediumseagreen', alpha=0.9, edgecolor='darkgreen')
    ax.axhline(8, color='red', linestyle='--', linewidth=1.5, label='Máximo teórico = 8')
    ax.set_title("Número de Soluções Encontradas pela Cinemática Inversa",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Configuração de Teste")
    ax.set_ylabel("Nº de soluções")
    ax.set_xticks(idx)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 7 (Soluções) salvo → {caminho}")


def grafico_erros_juntas(gt, thetas_ik, caminho):
    """Gráfico 8: Erro angular por junta (GT vs IK), em graus."""
    n   = len(gt['thetas'])
    idx = np.arange(1, n + 1)

    erros_deg = np.degrees(np.abs(
        ((thetas_ik - gt['thetas']) + np.pi) % (2 * np.pi) - np.pi
    ))

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axs = axs.flatten()
    cores = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for j in range(6):
        axs[j].bar(idx, erros_deg[:, j], color=cores[j], alpha=0.85, edgecolor='black', linewidth=0.5)
        axs[j].axhline(np.mean(erros_deg[:, j]), color='gray', linestyle='--', linewidth=1.4, label=f"Média: {np.mean(erros_deg[:, j]):.2f}°")
        axs[j].set_title(f"Junta {j+1}", fontsize=13, fontweight='bold')
        axs[j].set_ylabel("Erro angular (°)", fontsize=11)
        axs[j].set_xticks(idx)
        axs[j].legend(fontsize=10)
        axs[j].grid(True, linestyle="--", alpha=0.6)

    for ax in axs:
        ax.set_xlabel("Configuração de Teste", fontsize=11)

    fig.suptitle("Erro Angular por Junta — Ground Truth × Cinemática Inversa",
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Gráfico 8 (Erros por Junta) salvo → {caminho}")


def imprimir_relatorio(gt, ep, eo, n_sol):
    N = len(gt['thetas'])
    print("\n" + "=" * 65)
    print("  RELATÓRIO — CINEMÁTICA INVERSA  (Python vs. CoppeliaSim)")
    print("=" * 65)
    print(f"\n  Configurações testadas: {N}")
    print(f"\n  {'Config':<8}{'ΔPos (mm)':<14}{'ΔOri (Frob)':<16}{'Soluções'}")
    print("  " + "-" * 50)
    for i in range(N):
        print(f"  C{i+1:<7}{ep[i]*1e3:<14.4f}{eo[i]:<16.2e}{n_sol[i]}")
    print(f"\n  Erro de posição    — máx: {np.max(ep)*1e3:.4f} mm | médio: {np.mean(ep)*1e3:.4f} mm")
    print(f"  Erro de orientação — máx: {np.max(eo):.2e} | médio: {np.mean(eo):.2e}")
    print(f"  Soluções encontradas: min={int(np.min(n_sol))}  max={int(np.max(n_sol))}")
    print("=" * 65 + "\n")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        interface = UR5CoppeliaInterface()
    except Exception as e:
        print(f"\n[ERRO] Não foi possível conectar: {e}")
        sys.exit(1)

    gt = interface.coletar_ground_truth()

    ep, ep_x, ep_y, ep_z, eo, n_sol, thetas_ik, pos_ik_all, R_ik_all = \
        calcular_erros_inversa(gt, interface)

    imprimir_relatorio(gt, ep, eo, n_sol)

    grafico_curvas_juntas(gt,
        os.path.join(PASTA_SAIDA, "grafico_1_curvas_juntas.png"))
    grafico_curvas_posicao(gt, pos_ik_all,
        os.path.join(PASTA_SAIDA, "grafico_2_posicao_garra.png"))
    grafico_curvas_orientacao(gt, R_ik_all,
        os.path.join(PASTA_SAIDA, "grafico_3_orientacao_garra.png"))
    grafico_erros(ep, eo,
        os.path.join(PASTA_SAIDA, "grafico_4_erros.png"))
    grafico_dispersao_3d(gt, pos_ik_all,
        os.path.join(PASTA_SAIDA, "grafico_5_dispersao_3d.png"))
    grafico_erros_por_eixo(ep_x, ep_y, ep_z,
        os.path.join(PASTA_SAIDA, "grafico_6_erros_por_eixo.png"))
    grafico_num_solucoes(n_sol,
        os.path.join(PASTA_SAIDA, "grafico_7_num_solucoes.png"))
    grafico_erros_juntas(gt, thetas_ik,
        os.path.join(PASTA_SAIDA, "grafico_8_erros_juntas.png"))

    print(f"\n✅ Validação concluída!")
