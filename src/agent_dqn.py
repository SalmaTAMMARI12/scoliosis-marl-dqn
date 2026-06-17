import numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import random, os

LR           = 2e-4
GAMMA        = 0.95
BATCH        = 128
MEM_SIZE     = 20000
TARGET_SYNC  = 200
EPS_START    = 1.0
EPS_END      = 0.08
EPS_DECAY    = 0.997
N_ACT        = 8


class QNet(nn.Module):
    # Dueling DQN : CNN pour le patch + MLP pour les scalaires
    def __init__(self, n_actions=N_ACT):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # scalaires : rel_target(2) + rel_prior(2) + vb_idx(1) = 5
        self.mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.value   = nn.Sequential(nn.Linear(1024+64, 256), nn.ReLU(), nn.Linear(256, 1))
        self.adv     = nn.Sequential(nn.Linear(1024+64, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, patch, scalars):
        v = self.cnn(patch)
        s = self.mlp(scalars)
        x = torch.cat([v, s], dim=-1)
        val = self.value(x)
        adv = self.adv(x)
        return val + adv - adv.mean(dim=-1, keepdim=True)


class DQNAgent:
    def __init__(self, n_actions=N_ACT, device='cuda', model_dir='models/',
                 lr=None, eps_decay=None):
        self.n_actions = n_actions
        self.dev       = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        _lr        = lr        if lr        is not None else LR
        _eps_decay = eps_decay if eps_decay is not None else EPS_DECAY

        self.policy = QNet(n_actions).to(self.dev)
        self.target = QNet(n_actions).to(self.dev)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt       = optim.Adam(self.policy.parameters(), lr=_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=500, gamma=0.5)
        self.memory    = deque(maxlen=MEM_SIZE)
        self.epsilon   = EPS_START
        self.eps_decay = _eps_decay
        self.t         = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            p, s = self._ts(state)
            return self.policy(p, s).argmax().item()

    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def learn(self):
        if len(self.memory) < BATCH:
            return None
        batch  = random.sample(self.memory, BATCH)
        S, A, R, S2, D = zip(*batch)

        def mk(lst, key):
            return torch.stack([torch.FloatTensor(x[key]) for x in lst]).to(self.dev)

        patch  = mk(S,  'patch').unsqueeze(1)
        scal   = torch.stack([
            torch.FloatTensor(np.concatenate([x['rel_target'],x['rel_prior'],x['vb_idx']])) for x in S
        ]).to(self.dev)
        act_t  = torch.LongTensor(A).unsqueeze(1).to(self.dev)
        rew_t  = torch.FloatTensor(R).to(self.dev)
        patch2 = mk(S2, 'patch').unsqueeze(1)
        scal2  = torch.stack([
            torch.FloatTensor(np.concatenate([x['rel_target'],x['rel_prior'],x['vb_idx']])) for x in S2
        ]).to(self.dev)
        done_t = torch.FloatTensor(D).to(self.dev)

        curr_q = self.policy(patch, scal).gather(1, act_t).squeeze(1)
        with torch.no_grad():
            next_a = self.policy(patch2, scal2).argmax(1, keepdim=True)
            next_q = self.target(patch2, scal2).gather(1, next_a).squeeze(1)
            tgt    = rew_t + GAMMA * next_q * (1 - done_t)

        loss = nn.SmoothL1Loss()(curr_q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        self.epsilon = max(EPS_END, self.epsilon * self.eps_decay)
        self.t += 1
        if self.t % TARGET_SYNC == 0:
            self.target.load_state_dict(self.policy.state_dict())
        if self.t % 500 == 0:
            self.scheduler.step()
        return loss.item()

    def save(self, tag):
        torch.save({'policy': self.policy.state_dict(),
                    'target': self.target.state_dict(),
                    'epsilon': self.epsilon, 't': self.t},
                   os.path.join(self.model_dir, f'dqn_{tag}.pth'))

    def load(self, tag):
        p = os.path.join(self.model_dir, f'dqn_{tag}.pth')
        if not os.path.exists(p):
            print(f'  [WARN] Pas de modele: {p}'); return
        ck = torch.load(p, map_location=self.dev)
        self.policy.load_state_dict(ck['policy'])
        self.target.load_state_dict(ck['target'])
        self.epsilon = ck.get('epsilon', EPS_END)
        self.t       = ck.get('t', 0)
        print(f'  Modele charge: {p}')

    def _ts(self, s):
        p  = torch.FloatTensor(s['patch']).unsqueeze(0).unsqueeze(0).to(self.dev)
        sc = torch.FloatTensor(np.concatenate([s['rel_target'],s['rel_prior'],s['vb_idx']])).unsqueeze(0).to(self.dev)
        return p, sc

print("agent_dqn.py OK")
