r"""Optional Weights & Biases logging for the example scripts.

`add_wandb_args` adds `--wandb` (off by default) plus project/group/name
switches; `init_wandb` returns a logger whose `.log()` / `.summary()` /
`.finish()` are no-ops unless `--wandb` was given, so the scripts run
unchanged without wandb installed or configured."""
import os


def add_wandb_args(parser):
    parser.add_argument('--wandb', action='store_true',
                        help='log metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='rational-splinecnn')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='wandb group (default: <dataset>/<backbone>)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='wandb run name (default: wandb picks one)')
    return parser


class _NoLogger:
    enabled = False

    def log(self, metrics, step=None):
        pass

    def summary(self, metrics):
        pass

    def finish(self):
        pass


class _WandbLogger(_NoLogger):
    enabled = True

    def __init__(self, run):
        self.run = run

    def log(self, metrics, step=None):
        self.run.log(metrics, step=step)

    def summary(self, metrics):
        self.run.summary.update(metrics)

    def finish(self):
        self.run.finish()


def init_wandb(args, dataset, model=None, **extra_config):
    r"""Starts a wandb run (if `args.wandb`) tagged with the full argparse
    config, the dataset name and the backbone parameter count."""
    if not getattr(args, 'wandb', False):
        return _NoLogger()
    import wandb
    config = dict(vars(args))
    config.update(extra_config)
    config['dataset'] = dataset
    if model is not None:
        config['num_params'] = sum(p.numel() for p in model.parameters())
    tags = [dataset, args.backbone]
    if os.environ.get('SLURM_JOB_ID'):
        config['slurm_job_id'] = os.environ['SLURM_JOB_ID']
    run = wandb.init(project=args.wandb_project,
                     group=args.wandb_group or f'{dataset}/{args.backbone}',
                     name=args.wandb_name, config=config, tags=tags)
    return _WandbLogger(run)


class StepMeter:
    r"""Wraps :obj:`optimizer.step()` to record, per optimisation step, the
    global gradient norm :math:`\|\nabla_\theta L\|_2` and the parameter
    update norm :math:`\|\Delta\theta\|_2` (the effective step size), and
    reports per-epoch statistics of those together with the parameter norm
    and learning rate."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.reset()

    def reset(self):
        self.grad_norms, self.update_norms = [], []

    def step(self):
        import torch
        params = [p for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params]))
        before = [p.detach().clone() for p in params]
        self.optimizer.step()
        update_norm = torch.norm(torch.stack(
            [(p.detach() - b).norm() for p, b in zip(params, before)]))
        self.grad_norms.append(grad_norm.item())
        self.update_norms.append(update_norm.item())

    def metrics(self, prefix='opt/'):
        import torch
        g = torch.tensor(self.grad_norms)
        u = torch.tensor(self.update_norms)
        param_norm = torch.norm(torch.stack(
            [p.detach().norm() for p in self.model.parameters()])).item()
        out = {prefix + 'param_norm': param_norm,
               prefix + 'lr': self.optimizer.param_groups[0]['lr']}
        if len(g) > 0:
            out.update({
                prefix + 'grad_norm': g.mean().item(),
                prefix + 'grad_norm_max': g.max().item(),
                prefix + 'update_norm': u.mean().item(),
                prefix + 'update_norm_max': u.max().item(),
                prefix + 'update_ratio': (u / param_norm).mean().item(),
            })
        self.reset()
        return out
