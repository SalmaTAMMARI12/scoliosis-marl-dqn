import os, cv2, numpy as np, scipy.io as scio

NUM_VERTEBRAE = 17
IMG_W, IMG_H  = 256, 768

class ScoliosisDataset:
    def __init__(self, img_dir, label_dir, max_images=None):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        fnames = sorted([f for f in os.listdir(img_dir)
                         if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.samples = []
        for f in fnames:
            if os.path.isfile(os.path.join(label_dir, f + '.mat')):
                self.samples.append(f)
                if max_images and len(self.samples) >= max_images:
                    break
        assert self.samples, 'Aucune paire image/label trouvee'
        print(f'[Dataset] {len(self.samples)} images')

    def __len__(self): return len(self.samples)

    def load_image(self, idx):
        p   = os.path.join(self.img_dir, self.samples[idx])
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (IMG_W, IMG_H))

    def get_centers(self, idx):
        # retourne (17,2) centres dans coords image resizee
        fname = self.samples[idx]
        mat   = scio.loadmat(os.path.join(self.label_dir, fname + '.mat'))
        p2    = mat['p2'].astype(np.float32)
        img0  = cv2.imread(os.path.join(self.img_dir, fname))
        H0, W0 = img0.shape[:2]
        c = np.zeros((NUM_VERTEBRAE, 2), np.float32)
        for i in range(NUM_VERTEBRAE):
            corners = p2[4*i:4*i+4]
            c[i, 0] = corners[:,0].mean() / W0 * IMG_W
            c[i, 1] = corners[:,1].mean() / H0 * IMG_H
        return c

    def get_corners(self, idx):
        """Retourne les coins (17, 4, 2) pour les 17 vertèbres."""
        fname = self.samples[idx]
        mat = scio.loadmat(os.path.join(self.label_dir, fname + '.mat'))
        p2 = mat['p2'].astype(np.float32)
        img0 = cv2.imread(os.path.join(self.img_dir, fname))
        H0, W0 = img0.shape[:2]

        # p2 a 68 lignes (17 vertèbres * 4 coins) et 2 colonnes (x, y)
        # Il faut les remettre en forme (17, 4, 2)
        corners = p2.reshape((17, 4, 2))

        # Redimensionner les coordonnées à la taille de l'image redimensionnée (IMG_W, IMG_H)
        # (pour être cohérent avec les centres)
        corners[:, :, 0] = corners[:, :, 0] / W0 * IMG_W
        corners[:, :, 1] = corners[:, :, 1] / H0 * IMG_H
        return corners

print("dataset.py OK")