# controlador_atracao_repulsao_p3dx.py
# Versão ajustada: forças no referencial do robô, transformação das leituras, normalização de angulos

import time
import numpy as np
from math import sin, cos, atan2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROBOT_NAME = "Pioneer_p3dx"      
LEFT_MOTOR_NAME = ROBOT_NAME + "_leftMotor"
RIGHT_MOTOR_NAME = ROBOT_NAME + "_rightMotor"
# padrão de sensores 
SENSOR_PATTERN = "Pioneer_p3dx_ultrasonicSensor{}"  

# Cinêmica / físicos (use os valores que correspondam ao seu modelo na cena)
L = 0.381      # distância entre rodas 
r = 0.0975     # raio da roda

# parametros a serem ajustados para calibrar
K_att = 1.0
K_rep = 0.8
kv = 1.5
kw = 3.5

# limites e parâmetros do comportamento
d0 = 1.0        # raio da repulsão
v_max = 0.6
w_max = 1.5

#  coordenadas do mnundo -> alterar posteriormente para a posicao do goal
x_goal_world = np.array([2.0, 2.0])

 # condição de parada 30 cm do objetivo
DIST_GOAL_STOP = 0.3 


def angle_diff(a, b):
    """Retorna diferença angular (a - b) normalizada para [-pi, pi]."""
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))

def clip(x, a, b):
    return max(a, min(b, x))

def transform_point_matrix(sim, mat4, point3):

    M = np.array(mat4).reshape((3,4))
    p = np.array([point3[0], point3[1], point3[2], 1.0])
    res = M @ p
    return res  # 3-vector

# ---------------------------------------------------- Conexão com CoppeliaSim ---------------------------------------------------------------
client = RemoteAPIClient()
sim = client.require("sim")
sim.setStepping(True)


robotHandle = sim.getObject("/" + ROBOT_NAME) if sim.getObject("/" + ROBOT_NAME) != -1 else sim.getObject(ROBOT_NAME)
if robotHandle == -1:
    raise RuntimeError(f"Handle do robô não encontrado: verifique ROBOT_NAME='{ROBOT_NAME}' na sua cena.")

def get_handle_try(name):
    h = sim.getObject("/" + name)
    if h == -1:
        h = sim.getObject(name)
    return h


l_wheel = get_handle_try(LEFT_MOTOR_NAME)
r_wheel = get_handle_try(RIGHT_MOTOR_NAME)
if l_wheel == -1 or r_wheel == -1:
    raise RuntimeError("Handles das rodas não encontrados. Ajuste LEFT_MOTOR_NAME e RIGHT_MOTOR_NAME.")

# sensores
sensorHandles = []
for i in range(1, 17):
    name = SENSOR_PATTERN.format(i)
    h = get_handle_try(name)
    if h != -1:
        sensorHandles.append(h)
if len(sensorHandles) == 0:
    print("AVISO: nenhum sensor encontrado com o padrão. Verifique SENSOR_PATTERN / nomes na cena.")
else:
    print(f"{len(sensorHandles)} sensores detectados e serão usados.")

# checar existencia de um objeto goal na cena (opcional)
goal_handle = sim.getObject("/goal")
if goal_handle == -1:
    # não encontrou via nome absoluto; tenta por 'goal'
    goal_handle = sim.getObject("goal")
if goal_handle == -1:
    print("AVISO: objeto '/goal' não encontrado na cena — usando x_goal_world fixo.")
else:
    # atualiza x_goal_world usando objeto goal
    gp = sim.getObjectPosition(goal_handle, sim.handle_world)
    x_goal_world = np.array([gp[0], gp[1]])
    print("goal encontrado na cena; usando posição do goal:", x_goal_world)

# ----  TRANSFORMAR leitura dos sensores para o frame DO ROBO ----
def proximity_detect_in_robot_frame(sensorHandle):
    """
    Lê proximity sensor; se detectou, transforma o ponto detectado -> frame do robô.
    Retorna (detected, obs_in_robot (x,y,z), dist) ou (False, None, None).
    """
    try:
        # readProximitySensor costuma retornar (state, detectedPoint)
        state, detectedPoint = sim.readProximitySensor(sensorHandle)
    except Exception:
        # algumas variações retornam 3 valores
        res = sim.readProximitySensor(sensorHandle)
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            state, detectedPoint = res[0], res[1]
        else:
            return False, None, None

    if not state:
        return False, None, None


    # primeiro: tentamos usar getObjectMatrix do sensor e do robô
    try:
        sensor_mat = sim.getObjectMatrix(sensorHandle)  
        robot_mat = sim.getObjectMatrix(robotHandle)

        robot_mat_inv = sim.getMatrixInverse(robot_mat)
        rel = sim.multiplyMatrices(robot_mat_inv, sensor_mat)  # 3x4 matrix
        obs_in_robot_3 = sim.multiplyVector(rel, detectedPoint)
        obs_in_robot = np.array(obs_in_robot_3)
        dist = np.linalg.norm(obs_in_robot)
        return True, obs_in_robot, dist
    except Exception:
        # pega posição do sensor no mundo e posição do robo no mundo
        sensor_pos = np.array(sim.getObjectPosition(sensorHandle, sim.handle_world))
        sensor_ori = np.array(sim.getObjectOrientation(sensorHandle, sim.handle_world))  # rx,... em rad
        robot_pos = np.array(sim.getObjectPosition(robotHandle, sim.handle_world))
        robot_ori = np.array(sim.getObjectOrientation(robotHandle, sim.handle_world))

        yaw_s = sensor_ori[2]
        cs, ss = cos(yaw_s), sin(yaw_s)
        R_s_world = np.array([[cs, -ss, 0],[ss, cs, 0],[0,0,1]])
        point_world = sensor_pos + R_s_world @ np.array(detectedPoint)

        yaw_r = robot_ori[2]
        cr, sr = cos(yaw_r), sin(yaw_r)
        R_r_world = np.array([[cr, -sr, 0],[sr, cr, 0],[0,0,1]])
        R_world_r = R_r_world.T
        obs_in_robot = R_world_r @ (point_world - robot_pos)
        dist = np.linalg.norm(obs_in_robot)
        return True, obs_in_robot, dist

# ------------------------------------------Forca de repusao no frame do robo ----------------------------------------------------------------------
def compute_repulsion_force_robot_frame(sensor_handles, k_rep=K_rep, d0_local=d0):
    F_rep = np.zeros(2)
    for s in sensor_handles:
        detected, obs_in_robot, dist = proximity_detect_in_robot_frame(s)
        if not detected:
            continue
        if dist <= 0 or np.isnan(dist) or np.isinf(dist):
            continue
        if dist <= d0_local:
            dir_vec = -obs_in_robot[:2]
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir == 0:
                continue
            dir_vec = dir_vec / norm_dir
            mag = k_rep * (1.0 / dist - 1.0 / d0_local) / (dist**2 + 1e-6)
            F_rep += mag * dir_vec
    return F_rep

#------------------------------------------Forca de atracao no frame do robo ----------------------------------------------------------------------
def compute_attraction_force_robot_frame(goal_world, k_att=K_att):
    # pega pos atual real do robô (para transformar)
    rob_pos_world = np.array(sim.getObjectPosition(robotHandle, sim.handle_world))
    rob_ori = np.array(sim.getObjectOrientation(robotHandle, sim.handle_world))
    robot_theta = rob_ori[2]
    rel_goal_world = np.array([goal_world[0], goal_world[1]]) - rob_pos_world[:2]
    # world -> robot rotation
    c, s = cos(robot_theta), sin(robot_theta)
    R_inv = np.array([[c, s], [-s, c]])
    rel_goal_robot = R_inv @ rel_goal_world
    F_att_robot = k_att * rel_goal_robot
    return F_att_robot, rob_pos_world, robot_theta

#--------------------------------------------Resultante das forcas ------------------------------------------------------------------------------
def control_from_force_robot_frame(F_total, robot_theta, k_v=kv, k_w=kw, vlim=v_max, wlim=w_max):
    Fx, Fy = F_total[0], F_total[1]
    v = k_v * Fx
    w = k_w * Fy
    v = float(np.clip(v, -vlim, vlim))
    w = float(np.clip(w, -wlim, wlim))
    return v, w

# --------------------------------------------Inicialização simulação ----------------------------------------------------------------------------
if sim.getSimulationState() != 0:
    sim.stopSimulation()
    time.sleep(0.5)

sim.startSimulation()
sim.step()

# odometria estimada (inicial)
q = np.array([0.0, 0.0, 0.0])  
last_sim_time = sim.getSimulationTime()

print("Controle iniciado. Objetivo (world):", x_goal_world)

# loop principal
try:
    while True:
        sim_time = sim.getSimulationTime()
        dt = sim_time - last_sim_time if sim_time - last_sim_time > 0 else 1e-3
        last_sim_time = sim_time

        # ---- calcula forças no frame do robo ----
        F_att_robot, rob_pos_world, robot_theta = compute_attraction_force_robot_frame(x_goal_world)
        F_rep_robot = compute_repulsion_force_robot_frame(sensorHandles)
        F_total_robot = F_att_robot + F_rep_robot

        # evita estouros -> mag 1
        normF = np.linalg.norm(F_total_robot)
        if normF > 1e-6:
            F_total_robot = F_total_robot / normF

        # controlar (v,w) a partir de F_total em frame do robo 
        v, w = control_from_force_robot_frame(F_total_robot, robot_theta)

        # converte para velocidades das rodas 
        wr = (2.0 * v + w * L) / (2.0 * r)
        wl = (2.0 * v - w * L) / (2.0 * r)


        sim.setJointTargetVelocity(l_wheel, wl)
        sim.setJointTargetVelocity(r_wheel, wr)

        #  mesma matriz  de antes (Mdir @ [wr, wl])
        Mdir = np.array([
            [r * np.cos(q[2]) / 2.0, r * np.cos(q[2]) / 2.0],
            [r * np.sin(q[2]) / 2.0, r * np.sin(q[2]) / 2.0],
            [r / L, -r / L]
        ])
        u = np.array([wr, wl])
        q = q + (Mdir @ u) * dt

        gt_pos = sim.getObjectPosition(robotHandle, sim.handle_world)
        gt_ori = sim.getObjectOrientation(robotHandle, sim.handle_world)
        gt_xy = np.array(gt_pos[:2])
        gt_theta = gt_ori[2]

        dist_goal = np.linalg.norm(x_goal_world - gt_xy)

        print(f"[t={sim_time:.2f}] GT_pos={gt_xy.round(3)} θ={np.degrees(gt_theta):.1f}° | odom={q[:2].round(3)} θ={np.degrees(q[2]):.1f}° | dist_goal={dist_goal:.2f} | v={v:.3f} w={w:.3f}")

        if dist_goal <= DIST_GOAL_STOP:
            print("Chegou perto do objetivo — parando.")
            break

        sim.step()


    sim.setJointTargetVelocity(l_wheel, 0.0)
    sim.setJointTargetVelocity(r_wheel, 0.0)
    time.sleep(0.2)
    sim.stopSimulation()
    print("Simulação parada.")

except KeyboardInterrupt:
    sim.setJointTargetVelocity(l_wheel, 0.0)
    sim.setJointTargetVelocity(r_wheel, 0.0)
    sim.stopSimulation()
    print("Interrompido pelo usuário; sim parada.")
except Exception as e:
    print("Erro durante execução:", e)
    try:
        sim.setJointTargetVelocity(l_wheel, 0.0)
        sim.setJointTargetVelocity(r_wheel, 0.0)
        sim.stopSimulation()
    except:
        pass
