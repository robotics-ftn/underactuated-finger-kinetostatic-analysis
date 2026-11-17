import csv
from datetime import datetime

import numpy as np


def e_theta(theta):
    e = np.array([[np.cos(theta)],
                  [np.sin(theta)]])
    return e


def e_perp_theta(theta):
    e_perp = np.array([[-np.sin(theta)],
                       [np.cos(theta)]])
    return e_perp


Q_LIMIT_DEG = [
    (0, 75),
    (0, 110),
    (0, 90)
]
NUMBER_OF_Q_STEPS = (75, 110, 90)

K_COEF_LIMIT = [
    (0.1, 1.0),
    (0.1, 1.0),
    (0.1, 1.0)
]
NUMBER_OF_K_STEPS = (10, 10, 10)

L_1 = 51/1000
L_2 = 51/1000
L_3 = 40/1000
a_1 = 30/1000
a_2 = 25/1000
b_1 = 48.75/1000
b_2 = 48.5/1000
c_1 = 15/1000
c_2 = 10/1000
PSI_2_DEG = 55
PSI_3_DEG = 118
THETA_0_DEG = 55
s_1 = 19/1000
s_2 = 22/1000
L_s1_0 = 19.9/1000
L_s2_0 = 21.3/1000
k_s1 = 1.24*1000
k_s2 = 0.53*1000

MOT_TRQ_MAX = 5.67
GEAR_RATIO = 1/4
GEAR_EFFICIENCY = 1  # 0.885
MOT_TRQ_OVERRIDE = 20
MOT_OVERRIDE = True

psi_2 = np.radians(PSI_2_DEG)
psi_3 = np.radians(PSI_3_DEG)

q = np.zeros((3, 1))

q_max = np.array([[Q_LIMIT_DEG[0][1]],
                  [Q_LIMIT_DEG[1][1]],
                  [Q_LIMIT_DEG[2][1]]])
q_max = np.radians(q_max)

q_min = np.array([[Q_LIMIT_DEG[0][0]],
                  [Q_LIMIT_DEG[1][0]],
                  [Q_LIMIT_DEG[2][0]]])
q_min = np.radians(q_min)

q_steps = np.array([[NUMBER_OF_Q_STEPS[0]],
                    [NUMBER_OF_Q_STEPS[1]],
                    [NUMBER_OF_Q_STEPS[2]]])
q_step = (q_max - q_min) / q_steps

theta_0 = np.array([[THETA_0_DEG],
                    [THETA_0_DEG],
                    [THETA_0_DEG]])
theta_0 = np.radians(theta_0)

T = np.array([[1, 0, 0],
              [1, 1, 0],
              [1, 1, 1]])

k = np.zeros((3, 1))

k_max = np.array([[K_COEF_LIMIT[0][1]],
                  [K_COEF_LIMIT[1][1]],
                  [K_COEF_LIMIT[2][1]]])

k_min = np.array([[K_COEF_LIMIT[0][0]],
                  [K_COEF_LIMIT[1][0]],
                  [K_COEF_LIMIT[2][0]]])

k_steps = np.array([[NUMBER_OF_K_STEPS[0]],
                    [NUMBER_OF_K_STEPS[1]],
                    [NUMBER_OF_K_STEPS[2]]])
k_step = (k_max - k_min) / (k_steps - 1)

count = 0
all_count = 0
data = {}

header = [
    'No.',
    'q1',
    'q2',
    'q3',
    'theta1',
    'theta2',
    'theta3',
    'IsPoseFeasible?',
    'k1',
    'k2',
    'k3',
    'lambda1',
    'lambda2',
    'lambda3',
    'tau_eta1',
    'tau_eta2',
    'tau_eta3',
    'tau_theta1',
    'tau_theta2',
    'tau_theta3',
    'F_N1',
    'F_N2',
    'F_N3',
    'IsForcePositice?',
    'mf',
    'Rf',
    'Nf',
    'Nv',
    'min-singular-value_Jalpha-theta',
    'min-singular-value_Jeta-theta',
    'min-singular-value_Jnc',
    'condition-number_kJalpha-theta',
    'condition-number_kJeta-theta',
    'condition-number_kJnc',
    'detJalpha-theta',
    'detJeta-theta',
    'detJnc'
]

now = datetime.now()
formatted = now.strftime("%Y-%m-%d-%H-%M")
name = f"output_data/data_{formatted}.csv"
file = open(name, 'w', newline='', encoding='utf-8')
writer = csv.DictWriter(file, fieldnames=header)
writer.writeheader()

for i_0 in range(0, q_steps[0, 0] + 1):
    q[0] = q_min[0] + i_0 * q_step[0]

    for i_1 in range(0, q_steps[1, 0] + 1):
        q[1] = q_min[1] + i_1 * q_step[1]

        for i_2 in range(0, q_steps[2, 0] + 1):
            q[2] = q_min[2] + i_2 * q_step[2]

            print(f"q\u2081: {np.degrees(q[0, 0]):.2f}° | q\u2082: {np.degrees(q[1, 0]):.2f}° | q\u2083: {np.degrees(q[2, 0]):.2f}°")
            print("---")

            for i_3 in np.arange(k_min[0, 0], k_max[0, 0] + k_step[0, 0], k_step[0, 0]):
                k[0, 0] = L_1 * i_3

                for i_4 in np.arange(k_min[1, 0], k_max[1, 0] + k_step[1, 0], k_step[1, 0]):
                    k[1, 0] = L_2 * i_4

                    for i_5 in np.arange(k_min[2, 0], k_max[2, 0] + k_step[2, 0], k_step[2, 0]):
                        k[2, 0] = L_3 * i_5

                        print(f"k\u2081: {k[0, 0]:.2f} | k\u2082: {k[1, 0]:.2f} | k\u2083: {k[2, 0]:.2f}")

                        data['No.'] = all_count

                        data['q1'] = np.degrees(q[0, 0])
                        data['q2'] = np.degrees(q[1, 0])
                        data['q3'] = np.degrees(q[2, 0])

                        theta = T @ q + theta_0
                        data['theta1'] = np.degrees(theta[0, 0])
                        data['theta2'] = np.degrees(theta[1, 0])
                        data['theta3'] = np.degrees(theta[2, 0])

                        if theta[0, 0] > theta[1, 0] or theta[1, 0] > theta[2, 0]:
                            data['IsPoseFeasible?'] = 'No'

                        else:
                            data['IsPoseFeasible?'] = 'Yes'

                        data['k1'] = k[0, 0]
                        data['k2'] = k[1, 0]
                        data['k3'] = k[2, 0]

                        data['lambda1'] = k[0, 0]/L_1
                        data['lambda2'] = k[1, 0]/L_2
                        data['lambda3'] = k[2, 0]/L_3

                        x_2 = L_2**2 + c_2**2 + a_2**2 - b_2**2 + 2*L_2*c_2*np.cos(theta[1, 0] - theta[2, 0] + psi_3)
                        m_2 = L_2*np.cos(theta[1, 0]) + c_2*np.cos(theta[2, 0] - psi_3)
                        n_2 = L_2*np.sin(theta[1, 0]) + c_2*np.sin(theta[2, 0] - psi_3)

                        alpha_2_1 = np.arctan2(n_2, m_2) + np.arccos(x_2 / (2*a_2*np.sqrt(n_2**2 + m_2**2)))
                        alpha_2_2 = np.arctan2(n_2, m_2) - np.arccos(x_2 / (2*a_2*np.sqrt(n_2**2 + m_2**2)))

                        x_1 = L_1**2 + c_1**2 + a_1**2 - b_1**2 + 2*L_1*c_1*np.cos(theta[0, 0] - alpha_2_2 + psi_2)
                        m_1 = L_1*np.cos(theta[0, 0]) + c_1*np.cos(alpha_2_2 - psi_2)
                        n_1 = L_1*np.sin(theta[0, 0]) + c_1*np.sin(alpha_2_2 - psi_2)

                        alpha_1_1 = np.arctan2(n_1, m_1) + np.arccos(x_1 / (2*a_1*np.sqrt(n_1**2 + m_1**2)))
                        alpha_1_2 = np.arctan2(n_1, m_1) - np.arccos(x_1 / (2*a_1*np.sqrt(n_1**2 + m_1**2)))

                        alpha = np.array([[alpha_1_2],
                                          [alpha_2_2]])

                        p_P1 = a_1 * e_theta(alpha[0, 0])
                        p_P2 = L_1 * e_theta(theta[0, 0]) + a_2 * e_theta(alpha[1, 0])
                        p_P3 = L_1 * e_theta(theta[0, 0]) + L_2 * e_theta(theta[1, 0]) + c_2 * e_theta(theta[2, 0] - psi_3)
                        p_P2_prime = L_1 * e_theta(theta[0, 0]) + c_1 * e_theta(alpha[1, 0] - psi_2)

                        r1 = p_P2_prime - p_P1
                        r2 = p_P3 - p_P2

                        A_11 = r1.T * L_1 @ e_perp_theta(theta[0, 0])
                        A_11 = A_11[0, 0]
                        A_12 = 0
                        A_13 = 0
                        A_21 = 0
                        A_22 = r2.T * L_2 @ e_perp_theta(theta[1, 0])
                        A_22 = A_22[0, 0]
                        A_23 = r2.T * c_2 @ e_perp_theta(theta[2, 0] - psi_3)
                        A_23 = A_23[0, 0]

                        B_11 = -r1.T * a_1 @ e_perp_theta(alpha[0, 0])
                        B_11 = B_11[0, 0]
                        B_12 = r1.T * c_1 @ e_perp_theta(alpha[1, 0] - psi_2)
                        B_12 = B_12[0, 0]
                        B_21 = 0
                        B_22 = -r2.T * a_2 @ e_perp_theta(alpha[1, 0])
                        B_22 = B_22[0, 0]

                        J_theta = np.array([[A_11, A_12, A_13],
                                            [A_21, A_22, A_23]])

                        J_alpha = np.array([[B_11, B_12],
                                            [B_21, B_22]])

                        det_J_alpha = np.linalg.det(J_alpha)

                        J_alpha_inv = np.linalg.inv(J_alpha)
                        J_alpha_theta = -J_alpha_inv @ J_theta
                        J_eta_theta = np.append(J_alpha_theta, [[0, 0, 1]], axis=0)

                        J_Nc_11 = k[0, 0]
                        J_Nc_12 = 0
                        J_Nc_13 = 0
                        J_Nc_21 = L_1 * np.cos(theta[1, 0] - theta[0, 0])
                        J_Nc_22 = k[1, 0]
                        J_Nc_23 = 0
                        J_Nc_31 = L_1 * np.cos(theta[2, 0] - theta[0, 0])
                        J_Nc_32 = L_2 * np.cos(theta[2, 0] - theta[1, 0])
                        J_Nc_33 = k[2, 0]

                        J_Nc = np.array([[J_Nc_11, J_Nc_12, J_Nc_13],
                                         [J_Nc_21, J_Nc_22, J_Nc_23],
                                         [J_Nc_31, J_Nc_32, J_Nc_33]])

                        svd_J_alpha_theta = np.linalg.svd(J_alpha_theta, compute_uv=False)
                        min_singular_value = svd_J_alpha_theta[-1]
                        data['min-singular-value_Jalpha-theta'] = min_singular_value
                        data['condition-number_kJalpha-theta'] = svd_J_alpha_theta[0]/min_singular_value
                        data['detJalpha-theta'] = np.prod(svd_J_alpha_theta)

                        svd_J_eta_theta = np.linalg.svd(J_eta_theta, compute_uv=False)
                        min_singular_value = svd_J_eta_theta[-1]
                        data['min-singular-value_Jeta-theta'] = min_singular_value
                        data['condition-number_kJeta-theta'] = svd_J_eta_theta[0]/min_singular_value
                        data['detJeta-theta'] = np.linalg.det(J_eta_theta)

                        svd_J_Nc = np.linalg.svd(J_Nc, compute_uv=False)
                        min_singular_value = svd_J_Nc[-1]
                        data['min-singular-value_Jnc'] = min_singular_value
                        data['condition-number_kJnc'] = svd_J_Nc[0]/min_singular_value
                        data['detJnc'] = np.linalg.det(J_Nc)

                        v_x = (L_1 - s_1) * np.cos(theta[0, 0]) + c_1 * np.cos(alpha[1, 0] - psi_2)
                        v_y = (L_1 - s_1) * np.sin(theta[0, 0]) + c_1 * np.sin(alpha[1, 0] - psi_2)
                        L_s1 = np.sqrt(v_x**2 + v_y**2)
                        d_1 = (c_1 * (L_1 - s_1) * np.abs((np.sin(alpha[1, 0] - theta[0, 0] - psi_2)))) / L_s1

                        F_s1 = k_s1 * (L_s1 - L_s1_0)

                        r_B_P2_prime = c_1 * e_theta(alpha[1, 0] - psi_2)
                        r_s1_P2_prime = (L_1 - s_1) * e_theta(theta[0, 0]) + c_1 * e_theta(alpha[1, 0] - psi_2)
                        u_s1 = r_s1_P2_prime / L_s1
                        F_s1_prime = - F_s1 * u_s1

                        tau_s1 = np.cross(np.append(r_B_P2_prime.flatten(), 0), np.append(F_s1_prime.flatten(), 0))
                        tau_alpha2 = 2 * tau_s1[2]

                        F_eq1_vec = 2 * F_s1_prime
                        F_eq1 = np.linalg.norm(F_eq1_vec)

                        v_x = (L_2 - s_2) * np.cos(theta[1, 0]) + c_2 * np.cos(theta[2, 0] - psi_3)
                        v_y = (L_2 - s_2) * np.sin(theta[1, 0]) + c_2 * np.sin(theta[2, 0] - psi_3)
                        L_s2 = np.sqrt(v_x**2 + v_y**2)
                        d_2 = (c_2 * (L_2 - s_2) * np.abs((np.sin(theta[2, 0] - theta[1, 0] - psi_3)))) / L_s2

                        F_s2 = k_s2 * (L_s2 - L_s2_0)

                        r_C_P3 = c_2 * e_theta(theta[2, 0] - psi_3)
                        r_s2_P3 = (L_2 - s_2) * e_theta(theta[1, 0]) + c_2 * e_theta(theta[2, 0] - psi_3)
                        u_s2 = r_s2_P3 / L_s2
                        F_s2_prime = - F_s2 * u_s2

                        tau_s2 = np.cross(np.append(r_C_P3.flatten(), 0), np.append(F_s2_prime.flatten(), 0))
                        tau_theta3 = 2 * tau_s2[2]

                        F_eq2_vec = 2 * F_s2_prime
                        F_eq2 = np.linalg.norm(F_eq2_vec)

                        test_tau_alpha2 = F_eq1 * d_1
                        test_tau_teta3 = F_eq2 * d_2

                        tau_alpha1 = MOT_TRQ_MAX * (1/GEAR_RATIO) * GEAR_EFFICIENCY if not MOT_OVERRIDE else MOT_TRQ_OVERRIDE

                        tau_eta = np.array([[tau_alpha1],
                                            [tau_alpha2],
                                            [tau_theta3]])
                        data['tau_eta1'] = tau_eta[0, 0]
                        data['tau_eta2'] = tau_eta[1, 0]
                        data['tau_eta3'] = tau_eta[2, 0]

                        tau_theta = J_eta_theta.T @ tau_eta
                        data['tau_theta1'] = tau_theta[0, 0]
                        data['tau_theta2'] = tau_theta[1, 0]
                        data['tau_theta3'] = tau_theta[2, 0]

                        F_N3 = -(-tau_theta[2, 0] / k[2, 0])
                        F_N2 = -(-tau_theta[1, 0]*(1/k[1, 0]) + tau_theta[2, 0]*(L_2*np.cos(theta[2, 0] - theta[1, 0]))/(k[1, 0]*k[2, 0]))
                        F_N1 = -(-tau_theta[0, 0]*(1/k[0, 0]) + tau_theta[1, 0]*((L_1*np.cos(theta[1, 0] - theta[0, 0]))/(k[0, 0]*k[1, 0])) - tau_theta[2, 0]*((L_1*(L_2*np.cos(theta[1, 0] - theta[0, 0])*np.cos(theta[2, 0]-theta[1, 0])-k[1, 0]*np.cos(theta[2, 0]-theta[0, 0])))/(k[0, 0]*k[1, 0]*k[2, 0])))

                        F_N = np.array([[F_N1],
                                        [F_N2],
                                        [F_N3]])
                        data['F_N1'] = F_N[0, 0]
                        data['F_N2'] = F_N[1, 0]
                        data['F_N3'] = F_N[2, 0]

                        if F_N[0, 0] < 0 or F_N[1, 0] < 0 or F_N[2, 0] < 0:
                            data['IsForcePositice?'] = 'No'

                        else:
                            count += 1
                            data['IsForcePositice?'] = 'Yes'

                        data['mf'] = min([F_N[0, 0], F_N[1, 0], F_N[2, 0]])

                        if data['IsPoseFeasible?'] == 'Yes':
                            if 'Nv' not in data:
                                data['Nv'] = 0
                            data['Nv'] = data['Nv'] + 1

                        if data['mf'] > 0:
                            if 'Nf' not in data:
                                data['Nf'] = 0
                            data['Nf'] = data['Nf'] + 1

                        J_Nc_inv = np.linalg.inv(J_Nc)
                        F_N_new = J_Nc_inv.T @ tau_theta

                        np.set_printoptions(suppress=True, precision=8)

                        all_count += 1

                        writer.writerows([data])

            print("===")

file.close()
print(f"Total number of data points saved: {all_count}")
