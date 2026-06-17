# MARL Scoliosis — Localisation vertébrale par agents DQN multi-zones

Détection automatique des 17 vertèbres sur une radiographie de colonne vertébrale (vue postéro-antérieure) à l'aide de **3 agents de Deep Q-Learning indépendants** (un par zone anatomique : HAUT / MILIEU / BAS), suivie du calcul automatique de l'**angle de Cobb** pour l'évaluation de la sévérité d'une scoliose.

## Idée générale

Plutôt qu'un seul agent qui doit apprendre à localiser les 17 vertèbres (du thorax haut jusqu'au bas du dos), le problème est découpé en 3 sous-tâches plus simples, chacune confiée à son propre agent :

| Zone | Vertèbres | Modèle |
|---|---|---|
| HAUT | V1 → V6 | `models/top/` |
| MILIEU | V7 → V12 | `models/mid/` |
| BAS | V13 → V17 | `models/bot/` |

Chaque agent est un **Dueling DQN** : il observe un patch de l'image autour de sa position courante, ainsi que sa position relative à la cible et à un "prior anatomique" (la position moyenne attendue de la vertèbre, calculée sur le jeu d'entraînement), puis choisit un déplacement parmi 8 directions pour s'approcher du centre de la vertèbre à localiser.

Une fois les 17 positions prédites, le code calcule l'**angle de Cobb** à partir de l'inclinaison des plateaux vertébraux et propose une orientation clinique indicative (observation / surveillance / corset / chirurgie).

>  Ce projet est un travail de recherche / apprentissage. Il **ne remplace pas un diagnostic médical** et ne doit pas être utilisé en conditions cliniques réelles sans validation par un professionnel de santé.

## Structure du repo

```
scoliosis/
├── notebooks/
│   └── scoliosis.ipynb        # Notebook Colab complet (setup, entraînement, inférence)
├── src/
│   ├── dataset.py              # Chargement des images + labels .mat (coins de vertèbres)
│   ├── build_normal_prior.py   # Calcul du prior anatomique (moyenne/écart-type des centres)
│   ├── environment.py          # Environnement RL (état, actions, récompense) par zone
│   ├── agent_dqn.py             # Réseau Dueling DQN + boucle d'apprentissage
│   ├── train.py                  # Entraînement des 3 agents (HAUT / MILIEU / BAS)
│   └── inference.py              # Prédiction, angle de Cobb, visualisation
├── requirements.txt
└── .gitignore
```

Le notebook est la source de vérité (pensé pour tourner sur Google Colab avec GPU) ; les fichiers dans `src/` sont les mêmes scripts extraits pour pouvoir les lire, versionner et faire des diffs proprement sans ouvrir le `.ipynb`.

## Données attendues

Le projet utilise des images de radiographie + fichiers `.mat` contenant les coordonnées des 4 coins de chacune des 17 vertèbres (format `p2`, 68 points × 2 coordonnées), typique du dataset **AASCE / SpineWeb** (boostnet) pour l'évaluation automatique de la scoliose.

Organisation attendue sur disque :
```
data/
├── train/
│   ├── images/   *.jpg
│   └── labels/   *.jpg.mat
└── test/
    ├── images/
    └── labels/
```

Les données ne sont **pas incluses** dans ce repo (trop volumineuses, et soumises à la licence du dataset d'origine). Le notebook contient une cellule d'import qui prend un `.zip` contenant images + labels, les associe automatiquement, et fait un split train/test 80/20.

## Installation

```bash
git clone https://github.com/<votre-utilisateur>/<votre-repo>.git
cd <votre-repo>
pip install -r requirements.txt
```

GPU recommandé (le notebook est calibré pour ~4h d'entraînement sur GPU Colab).

## Utilisation

### 1. Préparer les données
Placer vos images et fichiers `.mat` dans `data/train/` et `data/test/` selon la structure ci-dessus (ou utiliser la cellule d'import du notebook si vous travaillez sur Colab).

### 2. Calculer le prior anatomique
```bash
python src/build_normal_prior.py
```
Génère `priors/mean_centers.npy` et `priors/std_centers.npy` (position moyenne et dispersion de chaque vertèbre sur le jeu d'entraînement).

### 3. Entraîner les 3 agents
```bash
python src/train.py \
    --img_dir   data/train/images/ \
    --label_dir data/train/labels/ \
    --prior_dir priors/ \
    --episodes  4000 \
    --device    cuda
```
Entraîne séquentiellement HAUT, MILIEU puis BAS. Les meilleurs poids de chaque agent sont sauvegardés dans `models/top/`, `models/mid/`, `models/bot/`.

### 4. Faire une prédiction + calcul de l'angle de Cobb
```python
from src.inference import predict

pred_centers, cobb, cobb_info, cobb_gt_info = predict(
    img_dir='data/test/images/',
    label_dir='data/test/labels/',
    prior_dir='priors/',
    img_idx=0,
    device='cpu'
)
```
Affiche un rapport détaillé (erreur de localisation par vertèbre, angle de Cobb prédit vs ground truth, vertèbres responsables de la courbure) et sauvegarde une visualisation dans `results/`.

## Détails techniques

- **État** : patch 64×64 autour de la position courante de l'agent + position relative à la cible + position relative au prior anatomique + index de la vertèbre courante dans la zone.
- **Actions** : 8 déplacements discrets (haut, bas, gauche, droite, diagonales).
- **Récompense** : bonus de proximité à la cible (forte récompense si erreur < seuil de succès), pénalité/bonus proportionnel au rapprochement, et un petit bonus si la position reste cohérente avec le prior anatomique.
- **Réseau** : Dueling DQN (branche `value` + branche `advantage`), CNN sur le patch image + MLP sur les scalaires de position.
- **Calibration spécifique à la zone HAUT** : la zone V1-V6 est plus étalée verticalement, donc cet agent utilise un nombre de pas plus élevé et un pas de déplacement plus fin que MILIEU/BAS, pour limiter les sauts au-delà de la cible et stabiliser l'apprentissage.

## équipe 
Salma TAMMARI && Hiba HAMDOUNI && Wissal mahboub && Assmaa EL HIDANI 