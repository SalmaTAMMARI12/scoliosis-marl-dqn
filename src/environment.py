import numpy as np, cv2

IMG_W, IMG_H   = 256, 768
PATCH_SZ       = 64
STEP_SZ        = 18.0
MAX_STEPS      = 20
SUCCESS_THRESH = 28.0

DELTAS = np.array([
    [ 0,-1],[ 0, 1],[-1, 0],[ 1, 0],
    [-1,-1],[ 1,-1],[-1, 1],[ 1, 1],
], dtype=np.float32) * STEP_SZ

N_ACTIONS = len(DELTAS)


def extract_patch(img_gray, cx, cy, sz=PATCH_SZ):
    half = sz // 2
    x0 = max(0, int(cx)-half);  x1 = min(IMG_W, x0+sz)
    y0 = max(0, int(cy)-half);  y1 = min(IMG_H, y0+sz)
    p  = img_gray[y0:y1, x0:x1]
    if p.shape != (sz, sz):
        p = cv2.resize(p, (sz, sz))
    return p.astype(np.float32) / 255.0


class ZoneEnvironment:
    # Un environnement par zone (HAUT / MILIEU / BAS)
    # max_steps et step_sz peuvent etre surcharges par zone
    # pour regler finement l agent HAUT sans toucher MILIEU/BAS

    def __init__(self, dataset, vb_start, vb_end, start_y, mean_centers, std_centers,
                 max_steps=None, step_sz=None):
        self.dataset   = dataset
        self.vb_start  = vb_start
        self.vb_end    = vb_end
        self.n_vb      = vb_end - vb_start
        self.start_y   = start_y
        self.mean_c    = mean_centers[vb_start:vb_end]
        self.std_c     = std_centers[vb_start:vb_end]
        # parametres ajustables par zone (HAUT en a besoin, MILIEU/BAS gardent les defauts)
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS
        self.step_sz   = step_sz   if step_sz   is not None else STEP_SZ
        self.deltas    = np.array([
            [ 0,-1],[ 0, 1],[-1, 0],[ 1, 0],
            [-1,-1],[ 1,-1],[-1, 1],[ 1, 1],
        ], dtype=np.float32) * self.step_sz
        self._pos      = None

    def reset(self, img_idx=None):
        if img_idx is None:
            img_idx = np.random.randint(0, len(self.dataset))
        img           = self.dataset.load_image(img_idx)
        self._gray    = img[:, :, 0]
        self._centers = self.dataset.get_centers(img_idx)[self.vb_start:self.vb_end]
        self._step    = 0
        self._vb_idx  = 0

        jx = np.random.uniform(-15, 15)
        jy = np.random.uniform(-15, 15)   # jitter Y reduit (etait +/-25, trop large pour HAUT)
        self._pos = np.array([IMG_W/2 + jx, self.start_y + jy], dtype=np.float32)
        self._pos = self._clip(self._pos)
        self._history = []
        return self._state()

    def step(self, action):
        l2_before    = self._l2()
        self._pos    = self._clip(self._pos + self.deltas[action])
        l2_after     = self._l2()
        self._step  += 1
        self._history.append(self._pos.copy())

        reward = self._reward(l2_before, l2_after)

        steps_per_vb = self.max_steps // self.n_vb
        if self._step % steps_per_vb == 0 and self._vb_idx < self.n_vb - 1:
            self._vb_idx += 1

        done = (self._step >= self.max_steps)
        return self._state(), reward, done, l2_after

    def _state(self):
        patch      = extract_patch(self._gray, self._pos[0], self._pos[1])
        target     = self._centers[self._vb_idx]
        prior      = self.mean_c[self._vb_idx]
        rel_target = (target - self._pos) / np.array([IMG_W, IMG_H], dtype=np.float32)
        rel_prior  = (prior  - self._pos) / np.array([IMG_W, IMG_H], dtype=np.float32)
        return {
            'patch':      patch,
            'rel_target': rel_target.astype(np.float32),
            'rel_prior':  rel_prior.astype(np.float32),
            'vb_idx':     np.array([self._vb_idx / self.n_vb], dtype=np.float32),
        }

    def _reward(self, l2_before, l2_after):
        r = 0.0
        if l2_after < SUCCESS_THRESH:
            r += 10.0 * (1.0 - l2_after / SUCCESS_THRESH)
            if l2_after < 15.0:
                r += 5.0
        else:
            r += np.clip((l2_before - l2_after) / self.step_sz, -1.0, 1.0)

        prior = self.mean_c[self._vb_idx]
        std   = np.maximum(self.std_c[self._vb_idx], 5.0)
        if np.all(np.abs(self._pos - prior) < 2 * std):
            r += 0.3
        return float(r)

    def _l2(self):
        return float(np.linalg.norm(self._pos - self._centers[self._vb_idx]))

    def _clip(self, p):
        return np.clip(p, [0,0], [IMG_W-1, IMG_H-1]).astype(np.float32)

print("environment.py OK")
