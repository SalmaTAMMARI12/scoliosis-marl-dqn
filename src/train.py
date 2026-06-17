import argparse, time, numpy as np
from dataset import ScoliosisDataset
from environment import ZoneEnvironment
from agent_dqn import DQNAgent


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir',    default='data/train/images/')
    p.add_argument('--label_dir',  default='data/train/labels/')
    p.add_argument('--prior_dir',  default='priors/')
    p.add_argument('--episodes',   type=int, default=4000)
    p.add_argument('--eval_every', type=int, default=400)
    p.add_argument('--eval_imgs',  type=int, default=30)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--max_images', type=int, default=None)
    return p.parse_args()


def evaluate(env, agent, n=30):
    l2s, succ = [], 0
    bak = agent.epsilon; agent.epsilon = 0.0
    for i in range(min(n, len(env.dataset))):
        state = env.reset(img_idx=i)
        done  = False; last_l2 = 999.
        while not done:
            a = agent.act(state)
            state, _, done, l2 = env.step(a)
            last_l2 = l2
        l2s.append(last_l2)
        if last_l2 < 28.: succ += 1
    agent.epsilon = bak
    return np.mean(l2s), succ / min(n, len(env.dataset))


def train_zone(name, vb_start, vb_end, start_y, ds, mean_c, std_c, args, mdir,
               max_steps=None, step_sz=None, lr=None, eps_decay=None):
    print(f'\n{"="*60}')
    print(f'  Agent {name} | V{vb_start+1}-V{vb_end} | start_y={start_y:.0f}px', end='')
    if max_steps: print(f' | max_steps={max_steps}', end='')
    if step_sz:   print(f' | step_sz={step_sz}', end='')
    print(f'\n{"="*60}')

    env   = ZoneEnvironment(ds, vb_start, vb_end, start_y, mean_c, std_c,
                            max_steps=max_steps, step_sz=step_sz)
    agent = DQNAgent(device=args.device, model_dir=mdir, lr=lr, eps_decay=eps_decay)
    best_l2 = float('inf'); ep_l2 = []; t0 = time.time()

    for ep in range(1, args.episodes+1):
        state = env.reset(); done = False; last_l2 = 999.
        while not done:
            a = agent.act(state)
            ns, r, done, l2 = env.step(a)
            agent.store(state, a, r, ns, done)
            agent.learn()
            state = ns; last_l2 = l2
        ep_l2.append(last_l2)

        if ep % args.eval_every == 0:
            eval_l2, succ = evaluate(env, agent, args.eval_imgs)
            m100 = np.mean(ep_l2[-100:])
            mins = (time.time()-t0)/60
            print(f'  Ep {ep:5d}/{args.episodes} | '
                  f'Train {m100:.1f}px | Eval {eval_l2:.1f}px | '
                  f'Succes {succ:.0%} | eps {agent.epsilon:.3f} | {mins:.1f}min')
            if eval_l2 < best_l2:
                best_l2 = eval_l2
                agent.save('best')
                print(f'    -> Meilleur: {best_l2:.1f}px')

    agent.save('final')
    print(f'  {name} termine | Meilleur: {best_l2:.1f}px')
    return best_l2


def main():
    args   = get_args()
    mean_c = np.load(args.prior_dir + 'mean_centers.npy')
    std_c  = np.load(args.prior_dir + 'std_centers.npy')
    ds     = ScoliosisDataset(args.img_dir, args.label_dir, max_images=args.max_images)

    print('='*60)
    print(f'  MARL SCOLIOSE | {len(ds)} images | {args.episodes} ep/agent')
    print('='*60)

    r = {}

    # ── HAUT : corrections ciblees pour la divergence ───────────
    # max_steps=30 (5 steps/vertebre au lieu de 3)   -> zone HAUT plus etalee en Y
    # step_sz=12   (au lieu de 18)                   -> pas plus fin, evite de sauter la cible
    # lr=1e-4      (au lieu de 2e-4)                 -> apprentissage plus stable, evite divergence
    # eps_decay=0.9985 (au lieu de 0.997)            -> exploration un peu plus longue
    r['HAUT']   = train_zone('HAUT',   0,  6, mean_c[0,1],  ds, mean_c, std_c, args, 'models/top/',
                             max_steps=30, step_sz=12, lr=1e-4, eps_decay=0.9985)

    # ── MILIEU et BAS : INCHANGES, exactement comme avant ───────
    r['MILIEU'] = train_zone('MILIEU', 6, 12, mean_c[6,1],  ds, mean_c, std_c, args, 'models/mid/')
    r['BAS']    = train_zone('BAS',   12, 17, mean_c[12,1], ds, mean_c, std_c, args, 'models/bot/')

    print('\n' + '='*60 + '\n  RESULTATS FINAUX')
    for k, v in r.items():
        s = 'OK' if v < 35 else ('Moyen' if v < 60 else 'A reentrain.')
        print(f'  {k:<8}: {v:.1f}px  [{s}]')
    print('='*60)


if __name__ == '__main__':
    main()

print("train.py OK")
