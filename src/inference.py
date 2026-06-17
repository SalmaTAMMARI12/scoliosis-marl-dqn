import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from dataset import ScoliosisDataset
from environment import ZoneEnvironment, DELTAS, STEP_SZ, IMG_W, IMG_H, MAX_STEPS, PATCH_SZ
from agent_dqn import DQNAgent
from scipy.signal import medfilt

# ============================================================
#  ANGLE DE COBB — METHODE STANDARD AVEC LISSAGE
# ============================================================

def vertebra_tilt(centers, i):
    """Calcule l'inclinaison d'une vertèbre avec lissage"""
    n = len(centers)
    if i == 0:
        v = centers[1] - centers[0]
    elif i == n - 1:
        v = centers[n-1] - centers[n-2]
    else:
        v = centers[i+1] - centers[i-1]
    return float(np.degrees(np.arctan2(v[0], v[1])))

def cobb_angle_standard(centers):
    """
    Méthode STANDARD de Cobb avec lissage des inclinaisons.
    """
    n = len(centers)
    if n < 3:
        return 0.0, 0, n-1, np.zeros(n)

    # Calculer les inclinaisons
    tilts = np.array([vertebra_tilt(centers, i) for i in range(n)])

    # Lisser les inclinaisons (filtre médian)
    tilts_smooth = medfilt(tilts, kernel_size=3)

    # Trouver les vertèbres extrêmes sur les inclinaisons lissées
    idx_max = np.argmax(tilts_smooth)
    idx_min = np.argmin(tilts_smooth)

    # S'assurer qu'elles sont espacées d'au moins 3 vertèbres
    if abs(idx_max - idx_min) < 3:
        # Chercher la prochaine meilleure paire
        candidates = []
        for i in range(n):
            for j in range(i+3, n):
                angle = abs(tilts_smooth[i] - tilts_smooth[j])
                candidates.append((angle, i, j))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            angle, i, j = candidates[0]
            # Limiter l'angle à 60° (raisonnable pour une colonne)
            if angle > 60:
                for a, ii, jj in candidates[1:]:
                    if a <= 60:
                        angle, i, j = a, ii, jj
                        break
            return float(angle), min(i, j), max(i, j), tilts

    # Angle de Cobb
    angle = abs(tilts_smooth[idx_max] - tilts_smooth[idx_min])

    # Limiter l'angle à 60° (la plupart des scolioses sont < 60°)
    if angle > 60:
        angle = 60.0

    # Mettre dans l'ordre
    if idx_max < idx_min:
        upper, lower = idx_max, idx_min
    else:
        upper, lower = idx_min, idx_max

    return float(angle), upper, lower, tilts

def treatment(cobb):
    if cobb < 10:  return 'Normal', 'Observation uniquement'
    if cobb < 25:  return 'Legere', 'Controle tous les 6 mois'
    if cobb < 45:  return 'Moderee', 'Corset recommande'
    return 'Severe', 'Chirurgie recommandee'

def run_agent(agent, env, img_idx):
    """Exécute l'agent sur l'environnement"""
    bak = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset(img_idx=img_idx)
    done = False

    while not done:
        a = agent.act(state)
        state, _, done, _ = env.step(a)

    agent.epsilon = bak

    if hasattr(env, '_history'):
        history = env._history.copy()
    else:
        history = []

    return history, []

def predict(img_dir, label_dir, prior_dir, img_idx=0, device='cpu'):
    import os

    # Recherche automatique des fichiers prior
    if not os.path.exists(prior_dir + 'mean_centers.npy'):
        possible_dirs = ['priors/', 'data/priors/', '../priors/']
        for d in possible_dirs:
            if os.path.exists(d + 'mean_centers.npy'):
                prior_dir = d
                break

    mean_c = np.load(prior_dir + 'mean_centers.npy')
    std_c = np.load(prior_dir + 'std_centers.npy')
    ds = ScoliosisDataset(img_dir, label_dir)

    zones = [
        ('HAUT', 0, 6, 'models/top/'),
        ('MILIEU', 6, 12, 'models/mid/'),
        ('BAS', 12, 17, 'models/bot/'),
    ]

    pred_final = np.zeros((17, 2), dtype=np.float32)

    for name, vb_s, vb_e, mdir in zones:
        env = ZoneEnvironment(ds, vb_s, vb_e, mean_c[vb_s,1], mean_c, std_c)
        agent = DQNAgent(device=device, model_dir=mdir)
        agent.load('best')

        history, _ = run_agent(agent, env, img_idx)

        n_vb = vb_e - vb_s
        if len(history) > 0:
            steps_per_vb = max(1, len(history) // n_vb)
            for i in range(n_vb):
                idx_h = min(i * steps_per_vb + steps_per_vb - 1, len(history) - 1)
                pred_final[vb_s + i] = history[idx_h]
        else:
            pred_final[vb_s:vb_e] = mean_c[vb_s:vb_e]

    gt = ds.get_centers(img_idx)
    img = ds.load_image(img_idx)
    l2s = np.linalg.norm(pred_final - gt, axis=1)

    # ============================================================
    # CALCUL DE L'ANGLE AVEC LA METHODE STANDARD LISSÉE
    # ============================================================
    cobb, iu, il, tilts = cobb_angle_standard(pred_final)

    # GT pour comparaison
    cobb_gt, iu_gt, il_gt, tilts_gt = cobb_angle_standard(gt)

    cat, msg = treatment(cobb)
    cat_gt, _ = treatment(cobb_gt)

    print(f'L2 moyen  : {l2s.mean():.1f}px | Max : {l2s.max():.1f}px')
    print(f'Cobb pred : {cobb:.1f} deg  (V{iu+1} -> V{il+1})  [{cat}]')
    print(f'Cobb GT   : {cobb_gt:.1f} deg  (V{iu_gt+1} -> V{il_gt+1})  [{cat_gt}]')
    print(f'Traitement: {msg}')

    # Vérification de la cohérence
    diff = abs(cobb - cobb_gt)
    if diff < 5:
        print(f'✅ Angle très proche du GT (écart: {diff:.1f}°)')
    elif diff < 10:
        print(f'⚠️ Angle modérément différent du GT (écart: {diff:.1f}°)')
    else:
        print(f'❌ Angle très différent du GT (écart: {diff:.1f}°)')

    # ============================================================
    # VISUALISATION
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 14))
    ax.imshow(img, cmap='gray')

    cols_pred = ['lime']*6 + ['yellow']*6 + ['cyan']*5

    # 1. Colonne normale
    ax.plot(mean_c[:,0], mean_c[:,1], '--', color='orange', lw=1.5, alpha=0.7, label='Colonne normale')

    # 2. Prédiction
    for i, (x, y) in enumerate(pred_final):
        ax.plot(x, y, 'o', color=cols_pred[i], ms=7, markeredgecolor='white', markeredgewidth=1)
        if i == iu:
            ax.text(x+5, y-3, f'V{i+1}↑', color='magenta', fontsize=8, fontweight='bold')
        elif i == il:
            ax.text(x+5, y-3, f'V{i+1}↓', color='cyan', fontsize=8, fontweight='bold')
        else:
            ax.text(x+5, y-3, f'V{i+1}', color=cols_pred[i], fontsize=7, fontweight='bold')
    ax.plot(pred_final[:,0], pred_final[:,1], '-', color='white', lw=2, alpha=0.8, label='Prédiction')

    # Tracer les tangentes aux vertèbres extrêmes
    for idx, color in [(iu, 'magenta'), (il, 'cyan')]:
        angle_rad = np.radians(tilts[idx])
        cx, cy = pred_final[idx]
        dx = np.sin(angle_rad) * 40
        dy = np.cos(angle_rad) * 40
        ax.plot([cx-dx, cx+dx], [cy-dy, cy+dy], '-', color=color, lw=2, alpha=0.9)

    # 3. Ground Truth
    for i, (x, y) in enumerate(gt):
        ax.plot(x, y, 'x', color='red', ms=8, markeredgewidth=2, label='GT' if i==0 else '')
    ax.plot(gt[:,0], gt[:,1], '-', color='red', lw=1.5, alpha=0.5, linestyle='--')

    # 4. Informations
    severity, treatment_msg = treatment(cobb)
    if cobb < 10:
        color_text = 'green'
    elif cobb < 25:
        color_text = 'yellowgreen'
    elif cobb < 45:
        color_text = 'orange'
    else:
        color_text = 'red'

    info_text = f"SCOLIOSE {severity.upper()}\n"
    info_text += f"Cobb: {cobb:.1f}° (V{iu+1}→V{il+1})\n"
    info_text += f"GT: {cobb_gt:.1f}° (V{iu_gt+1}→V{il_gt+1})\n"
    info_text += f"Erreur L2: {l2s.mean():.1f}px\n"
    info_text += f"Ecart: {abs(cobb-cobb_gt):.1f}°\n"
    info_text += f"{treatment_msg}"

    ax.text(10, 30, info_text, color=color_text, fontsize=10, weight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.85, edgecolor=color_text))

    # 5. Légende
    legend_elements = [
        plt.Line2D([0], [0], color='lime', marker='o', label='HAUT (V1-V6)'),
        plt.Line2D([0], [0], color='yellow', marker='o', label='MILIEU (V7-V12)'),
        plt.Line2D([0], [0], color='cyan', marker='o', label='BAS (V13-V17)'),
        plt.Line2D([0], [0], color='red', marker='x', label='Ground Truth'),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='Colonne normale'),
        plt.Line2D([0], [0], color='magenta', linestyle='-', label='Tangente sup'),
        plt.Line2D([0], [0], color='cyan', linestyle='-', label='Tangente inf'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    ax.set_title(f'Cobb: {cobb:.1f}° (GT: {cobb_gt:.1f}°) | Image {img_idx}', fontsize=12, weight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'results/pred_{img_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return pred_final, cobb

print("✅ inference.py chargé avec la méthode STANDARD LISSÉE !")