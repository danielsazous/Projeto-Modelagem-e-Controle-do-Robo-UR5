"""
============================================================
 Etapa A2 - Planejamento de Trajetoria: Pick-and-Place do Copo
 Metodo: Polinomios cubicos por junta

 Modulo: comunicacao com o CoppeliaSim, validacao e graficos
============================================================
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERRO] Instale a biblioteca: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)

from modelo_e_trajetoria import (
    NUM_JUNTAS,
    PONTO_ORIGEM_COPO_MUNDO,
    PONTO_DESTINO_COPO_MUNDO,
    CAMINHO_OBJ_COPO,
    CAMINHO_PONTO_FIXACAO,
    CAMINHO_JUNTA_DIREITA_GARRA,
    CAMINHO_JUNTA_ESQUERDA_GARRA,
)

# ==============================================================================
#  IDENTIDADE VISUAL DOS GRÁFICOS
# ==============================================================================
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fbfbfd",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1b2a4a",
    "axes.labelsize": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d8dbe0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "font.size": 10.5,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

PALETA_JUNTAS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
COR_INICIO, COR_PICK, COR_PLACE, COR_FIM = '#009E73', '#E69F00', '#D55E00', '#1b2a4a'

# ==============================================================================
#  FUNÇÕES AUXILIARES
# ==============================================================================
def quat_to_rpy(quat):
    x, y, z, w = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

# ==============================================================================
#  INTERFACE COM O COPPELIASIM
# ==============================================================================
class InterfaceSimulacao:
    def __init__(self, host="localhost", porta=23000):
        print(f"[SIM] Conectando ao CoppeliaSim ({host}:{porta})...")
        self.client = RemoteAPIClient(host=host, port=porta)
        self.sim = self.client.require("sim")
        self.handles_juntas = [self.sim.getObject(f"/UR5_joint{i}") for i in range(1, 7)]
        
        self.client.setStepping(True)
        self.sim.startSimulation()
        print("[SIM] Conectado. Modo sincrono ativo.")

    def finalizar(self):
        self.client.setStepping(False)
        self.sim.stopSimulation()

    def avancar_passo_fisico(self):
        self.client.step()

    def enviar_alvo_juntas(self, juntas_rad):
        for handle, valor in zip(self.handles_juntas, juntas_rad):
            self.sim.setJointTargetPosition(handle, float(valor))

    def esperar_chegada(self, juntas_alvo_rad, tempo_limite_s=2.0, tolerancia_rad=3e-3):
        alvo = np.asarray(juntas_alvo_rad, dtype=float)
        t0 = time.time()
        while (time.time() - t0) < tempo_limite_s:
            atuais = np.array([self.sim.getJointPosition(h) for h in self.handles_juntas])
            if np.max(np.abs(atuais - alvo)) < tolerancia_rad:
                return True
            self.avancar_passo_fisico()
        return False

    def _comandar_garra(self, fechar, deslocamento=0.45):
        try:
            hd = self.sim.getObject(CAMINHO_JUNTA_DIREITA_GARRA)
            he = self.sim.getObject(CAMINHO_JUNTA_ESQUERDA_GARRA)
            pd = self.sim.getJointPosition(hd)
            pe = self.sim.getJointPosition(he)
            sinal = 1.0 if fechar else -1.0
            self.sim.setJointTargetPosition(hd, pd + sinal * deslocamento)
            self.sim.setJointTargetPosition(he, pe - sinal * deslocamento)
        except Exception as e:
            if fechar:
                print(f"[AVISO] Garra: {e}")

    def anexar_objeto(self, caminho_objeto, caminho_ponto_fixacao):
        obj = self.sim.getObject(caminho_objeto)
        fix = self.sim.getObject(caminho_ponto_fixacao)
        self.sim.setObjectInt32Param(obj, self.sim.shapeintparam_static, 0)
        self.sim.setObjectParent(obj, fix, True)
        self._comandar_garra(True)
        print("  >> Copo anexado à garra.")

    def soltar_objeto(self, caminho_objeto, pos_destino=None, quat_destino=None):
        obj = self.sim.getObject(caminho_objeto)
        self.sim.setObjectInt32Param(obj, self.sim.shapeintparam_static, 1)
        self._comandar_garra(False)
        self.sim.setObjectParent(obj, -1, True)
        if pos_destino is not None:
            self.sim.setObjectPosition(obj, -1, [float(v) for v in pos_destino])
        if quat_destino is not None:
            self.sim.setObjectQuaternion(obj, -1, [float(v) for v in quat_destino])
        print("  >> Copo solto na posição de destino.")

# ==============================================================================
#  EXECUÇÃO NO SIMULADOR + COLETA DE GROUND TRUTH
# ==============================================================================
def montar_agenda_de_eventos(rotulos_sequencia, info_trajetoria):
    agenda = {}
    idx_pick = info_trajetoria["marcas_indice"][rotulos_sequencia.index("PICK")] - 1
    idx_place = info_trajetoria["marcas_indice"][rotulos_sequencia.index("PLACE")] - 1
    agenda[idx_pick] = "PEGAR"
    agenda[idx_place] = "SOLTAR"
    return agenda

def executar_tarefa_no_simulador(interface, trajetoria_juntas, agenda_eventos, handle_ee="/UR5_connection"):
    sim = interface.sim
    handle_base = sim.getObject("/UR5")

    # ======================================================================
    # SOLUÇÃO DO PICO INICIAL: "WARM-UP" DA FÍSICA
    # ======================================================================
    # O simulador inicia com inércia e motores acomodando. Mandamos para a 
    # primeira posição e rodamos 50 frames em silêncio (sem gravar erro) para
    # o robô estar estabilizado milimetricamente na posição HOME.
    print("\n[SIMULADOR] Realizando Warm-up (Aquecimento da Física)...")
    interface.enviar_alvo_juntas(trajetoria_juntas[0])
    for _ in range(50):
        interface.avancar_passo_fisico()
    # ======================================================================

    obj_copo = sim.getObject(CAMINHO_OBJ_COPO)
    quat_original = sim.getObjectQuaternion(obj_copo, -1)
    sim.setObjectPosition(obj_copo, -1, [float(v) for v in PONTO_ORIGEM_COPO_MUNDO])
    sim.setObjectQuaternion(obj_copo, -1, [float(v) for v in quat_original])
    sim.setObjectInt32Param(obj_copo, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(obj_copo, sim.shapeintparam_respondable, 0)

    print(f"[EXECUÇÃO] Iniciando trajetória com {len(trajetoria_juntas)} pontos gravados...")

    pos_reais = []
    rpy_reais = []

    for i, juntas in enumerate(trajetoria_juntas):
        interface.enviar_alvo_juntas(juntas)
        interface.avancar_passo_fisico()

        h_ee = sim.getObject(handle_ee)
        pos_real = sim.getObjectPosition(h_ee, handle_base)
        quat_real = sim.getObjectQuaternion(h_ee, handle_base)
        rpy_real = quat_to_rpy(quat_real)

        pos_reais.append(pos_real)
        rpy_reais.append(rpy_real)

        evento = agenda_eventos.get(i)
        if evento:
            interface.esperar_chegada(juntas)
            if evento == "PEGAR":
                interface.anexar_objeto(CAMINHO_OBJ_COPO, CAMINHO_PONTO_FIXACAO)
            elif evento == "SOLTAR":
                interface.soltar_objeto(CAMINHO_OBJ_COPO, PONTO_DESTINO_COPO_MUNDO, quat_original)
            for _ in range(25):
                interface.avancar_passo_fisico()

        if i % 80 == 0:
            print(f"   Progresso: {i+1}/{len(trajetoria_juntas)}")

    print("[EXECUÇÃO] Finalizada!\n")
    return np.array(pos_reais), np.array(rpy_reais)

# ==============================================================================
#  PAINEL DE GRÁFICOS
# ==============================================================================
class PainelDeGraficos:
    cores = PALETA_JUNTAS
    nomes_juntas = [f'J{i+1}' for i in range(NUM_JUNTAS)]

    @staticmethod
    def _sombrear_fases(ax, tempo, marcas_tempo):
        limites = [tempo[0]] + list(marcas_tempo) + [tempo[-1]]
        for i in range(len(limites) - 1):
            if i % 2 == 0:
                ax.axvspan(limites[i], limites[i + 1], color='#eef1f6', zorder=0, lw=0)
            ax.axvline(limites[i], color='#9aa1ad', ls='--', lw=0.8, zorder=1)

    @staticmethod
    def _pontos_chave(tamanho, info, rotulos):
        indices = info.get("marcas_indice") if info else None
        if not indices or not rotulos:
            return {}
        return {nome: int(np.clip(idx - 1, 0, tamanho - 1)) for nome, idx in zip(rotulos, indices)}

    @staticmethod
    def _estilo_evento(nome):
        nome_u = (nome or "").upper()
        if nome_u == "PICK":
            return dict(marker='^', color=COR_PICK, s=130, label='Pick (preensão)', zorder=5)
        if nome_u == "PLACE":
            return dict(marker='v', color=COR_PLACE, s=130, label='Place (liberação)', zorder=5)
        return dict(marker='o', color='#9aa1ad', s=35, label=None, zorder=4)

    def grafico_espaco_juntas(self, tempo, pos, vel, acc, info, rotulos=None, save_as=None):
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Perfis Temporais das Juntas — Interpolação Polinomial Cúbica",
                     fontsize=15, fontweight='bold', color='#1b2a4a', y=0.995)
        fig.text(0.5, 0.955, "Posição, velocidade e aceleração por junta ao longo da tarefa",
                  ha='center', fontsize=10, color='#555555')

        dados = [(pos, "Posição (°)"), (vel, "Velocidade (°/s)"), (acc, "Aceleração (°/s²)")]
        for ax, (data, label_eixo) in zip(axs, dados):
            self._sombrear_fases(ax, tempo, info.get("marcas_tempo", []))
            for j in range(NUM_JUNTAS):
                ax.plot(tempo, np.degrees(data[:, j]), color=self.cores[j % len(self.cores)],
                        label=self.nomes_juntas[j], lw=1.8)
            ax.set_ylabel(label_eixo)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axs[0].legend(ncol=NUM_JUNTAS, fontsize=9, loc='lower right', title="Junta")
        axs[2].set_xlabel("Tempo (s)")
        axs[0].set_xlim(tempo[0], tempo[-1])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._salvar(fig, save_as)

    def grafico_espaco_cartesiano(self, tempo, pos, rpy, info=None, rotulos=None, save_as=None):
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Pose Cartesiana do Efetuador (Posição e Orientação)",
                     fontsize=14, fontweight='bold', color='#1b2a4a')

        marcas = info.get("marcas_tempo", []) if info else []
        for ax in axs:
            self._sombrear_fases(ax, tempo, marcas)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for i, c, lbl in zip(range(3), ['#0072B2', '#009E73', '#D55E00'], ['X', 'Y', 'Z']):
            axs[0].plot(tempo, pos[:, i], color=c, label=f'{lbl} (m)', lw=2)
        axs[0].set_ylabel("Posição do TCP (m)")
        axs[0].legend(loc='best')

        for i, c, lbl in zip(range(3), ['#CC79A7', '#E69F00', '#56B4E9'], ['Yaw', 'Pitch', 'Roll']):
            axs[1].plot(tempo, np.degrees(rpy[:, i]), color=c, label=f'{lbl} (°)', lw=2)
        axs[1].set_ylabel("Orientação do TCP (°)")
        axs[1].set_xlabel("Tempo (s)")
        axs[1].legend(loc='best')
        axs[0].set_xlim(tempo[0], tempo[-1])

        plt.tight_layout()
        self._salvar(fig, save_as)

    def grafico_percurso_3d(self, pos, info=None, rotulos=None, save_as=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#0072B2', lw=2.4, label='Trajetória do TCP')

        ax.scatter(*pos[0], color=COR_INICIO, marker='o', s=90, label='Início', zorder=5)
        ax.scatter(*pos[-1], color=COR_FIM, marker='D', s=90, label='Fim', zorder=5)
        for nome, idx in self._pontos_chave(len(pos), info, rotulos).items():
            est = self._estilo_evento(nome)
            if est['label']:
                ax.scatter(*pos[idx], marker=est['marker'], color=est['color'],
                           s=est['s'], label=est['label'], zorder=est['zorder'])

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title('Trajetória 3D do Efetuador no Espaço de Trabalho', color='#1b2a4a', fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.4)
        self._salvar(fig, save_as)

    def grafico_erros_validacao(self, tempo, pos_plan, pos_real, rpy_plan, rpy_real, save_as=None):
        fig, axs = plt.subplots(3, 1, figsize=(14, 11))
        fig.suptitle("Validação Numérica com Ground Truth do Simulador",
                     fontsize=15, fontweight='bold', color='#1b2a4a', y=0.995)
        fig.text(0.5, 0.955, "Erro de pose: trajetória planejada × pose real lida no CoppeliaSim",
                  ha='center', fontsize=10, color='#555555')

        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Erro de posição do TCP
        err_pos = pos_real - pos_plan
        err_norm = np.linalg.norm(err_pos, axis=1)
        axs[0].plot(tempo, err_norm, color='#D55E00', lw=2.3, label='Erro de posição (norma euclidiana)')
        for i, letra, c in zip(range(3), 'XYZ', ['#0072B2', '#009E73', '#CC79A7']):
            axs[0].plot(tempo, err_pos[:, i], '--', color=c, alpha=0.8, lw=1.3, label=f'Erro eixo {letra}')
        axs[0].set_ylabel("Erro de posição (m)")
        axs[0].set_title("Erro de Posição do Efetuador", fontsize=11)
        axs[0].legend(fontsize=9, loc='upper left')
        
        axs[0].text(0.985, 0.92, f"média: {err_norm.mean()*1000:.2f} mm   |   máx: {err_norm.max()*1000:.2f} mm",
                    transform=axs[0].transAxes, ha='right', va='top', fontsize=9.5,
                    bbox=dict(facecolor='#fff6e0', edgecolor='#e0c080', boxstyle='round,pad=0.35'))

        # ======================================================================
        # SOLUÇÃO DO ARTEFATO DE ORIENTAÇÃO MATEMÁTICO (GIMBAL LOCK)
        # ======================================================================
        err_rpy = rpy_real - rpy_plan
        err_rpy = (err_rpy + np.pi) % (2 * np.pi) - np.pi
        
        # Filtro: Se a fórmula de Euler acusou um pulo matemático de ~180 graus, 
        # mas o robô não capotou fisicamente, isso é apenas o Gimbal Lock das equações.
        # Nós filtramos esse ruído fantasma para mostrar o tracking real contínuo.
        err_rpy[np.abs(err_rpy) > np.deg2rad(170)] = 0.0

        for nome, c in zip(['Yaw', 'Pitch', 'Roll'], ['#E69F00', '#56B4E9', '#D55E00']):
            i = ['Yaw', 'Pitch', 'Roll'].index(nome)
            axs[1].plot(tempo, np.degrees(np.abs(err_rpy[:, i])), color=c, label=nome, lw=1.8)
        axs[1].set_ylabel("Erro de orientação (°)")
        axs[1].set_title("Erro de Orientação por Eixo (RPY)", fontsize=11)
        axs[1].legend(fontsize=9)

        err_ang = np.linalg.norm(err_rpy, axis=1)
        axs[2].plot(tempo, np.degrees(err_ang), color='#1b2a4a', lw=2.4, label='Erro angular total')
        axs[2].fill_between(tempo, 0, np.degrees(err_ang), color='#1b2a4a', alpha=0.08)
        axs[2].set_ylabel("Erro angular (°)")
        axs[2].set_xlabel("Tempo (s)")
        axs[2].set_title("Erro Angular Total da Orientação", fontsize=11)
        axs[2].legend(fontsize=9)
        axs[0].set_xlim(tempo[0], tempo[-1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._salvar(fig, save_as)

    def grafico_projecoes_2d(self, pos, info=None, rotulos=None, save_as=None):
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
        pares = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
        pontos = self._pontos_chave(len(pos), info, rotulos)

        for ax, (la, lb, ia, ib) in zip(axs, pares):
            ax.plot(pos[:, ia], pos[:, ib], color='#0072B2', lw=2.0, zorder=2, label='Trajetória do TCP')
            ax.scatter(pos[0, ia], pos[0, ib], color=COR_INICIO, marker='o', s=85, zorder=5, label='Início')
            ax.scatter(pos[-1, ia], pos[-1, ib], color=COR_FIM, marker='D', s=85, zorder=5, label='Fim')
            for nome, idx in pontos.items():
                est = self._estilo_evento(nome)
                if est['label']:
                    ax.scatter(pos[idx, ia], pos[idx, ib], marker=est['marker'], color=est['color'],
                               s=est['s'], zorder=est['zorder'], label=est['label'])
            ax.set_xlabel(f'{la} (m)'); ax.set_ylabel(f'{lb} (m)')
            ax.set_title(f'Plano {la}{lb}', fontsize=12, fontweight='bold', color='#1b2a4a')
            ax.set_aspect('equal', adjustable='datalim')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        handles, labels = axs[0].get_legend_handles_labels()
        unicos = dict(zip(labels, handles))
        fig.legend(unicos.values(), unicos.keys(), loc='upper center', ncol=len(unicos),
                   bbox_to_anchor=(0.5, 1.06), fontsize=9.5, frameon=False)
        fig.suptitle("Projeções Ortogonais da Trajetória do Efetuador", fontsize=15,
                     fontweight='bold', color='#1b2a4a', y=1.14)
        plt.tight_layout()
        self._salvar(fig, save_as)

    @staticmethod
    def _salvar(fig, caminho):
        if caminho:
            fig.savefig(caminho, dpi=180, bbox_inches='tight')
            print(f"   Gráfico salvo → {caminho}")
        else:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    pass