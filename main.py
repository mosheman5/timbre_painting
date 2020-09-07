import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from pathlib import Path
from models.networks import init_models
from models.losses import TotalLoss
from utils import utils, sampling
from tqdm import tqdm
from data_utils.data import F0Dataset
from trainers import Trainer
from data_utils.crepe_pytorch import load_crepe
import logging
import hydra
import sys
import random
from shutil import copy
#
# Train
#

def train(
    output_dirpath,
    trainer,
    init_models,
    srs,
    device,
    dataset_len,
    input_path,
    duration,
    max_val,
    max_val_f0,
    batch_size,
    batch_size_min,
    num_workers,
    noise,
    use_prev_weights,
    checkpoint,
    distributed=False,
    world_size=1,
    rank=0
):
    # create audios on different scales
    samplers = sampling.create_samplers(srs, device=device)
    Gs = []
    Ds = []

    for scale_num in tqdm(range(len(srs))):
        print(f"Scale {scale_num}")
        scale_output_dirpath = f"{output_dirpath}/{scale_num}"
        if rank == 0:
            os.makedirs(scale_output_dirpath)
        scale_tb_prefix = f"scale{scale_num}"
        batch_size_scale = max(batch_size // 2**scale_num, batch_size_min)

        D_curr, G_curr = init_models()

        if distributed:
            # wrap model for distributed training
            try:
                from apex.parallel import DistributedDataParallel
                # from torch.nn.parallel import DistributedDataParallel
            except ImportError:
                raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
            G_curr = DistributedDataParallel(G_curr)
            D_curr = DistributedDataParallel(D_curr)
        else:
            G_curr = torch.nn.DataParallel(G_curr)
            D_curr = torch.nn.DataParallel(D_curr)

        if use_prev_weights and scale_num>0:
            G_curr.module.load_state_dict((Gs[scale_num - 1].module.state_dict()))
            D_curr.module.load_state_dict((Ds[scale_num - 1].module.state_dict()))


        # sampler for f0 loss, accepts only 16 kHz
        sampler_16k = sampling.Sampler(orig_freq=srs[scale_num], new_freq=16000, device=device)
        # set paths and create dataset
        train_dataset = F0Dataset(dataset_len=dataset_len, srs=srs[:scale_num+1] , input_path=input_path,
                            duration=duration, val_flag=False, noise=noise, rank=rank, num_ranks=world_size,
                                  max_val=max_val, max_val_f0=max_val_f0)
        val_dataset = F0Dataset(dataset_len=dataset_len, srs=srs[:scale_num+1] , input_path=input_path,
                            duration=duration, val_flag=True, noise=noise, rank=rank, num_ranks=world_size,
                                  max_val=max_val, max_val_f0=max_val_f0)
        train_dataloader = partial(DataLoader, batch_size=batch_size_scale, num_workers=num_workers, pin_memory=True)
        val_dataloader = partial(DataLoader, batch_size=batch_size_scale, num_workers=num_workers, pin_memory=True)

        if checkpoint:
            checkpoint_scale = Path(checkpoint).joinpath(f'{scale_num}/last.pth')
            if not checkpoint_scale.exists():
                checkpoint_scale = None
        else:
            checkpoint_scale = None

        draw_f0_par = partial(
            utils.draw_f0,
            Gs=Gs,
            samplers=samplers,
            max_val=max_val,
        )

        trainer_scale = trainer(
            netD=D_curr,
            netG=G_curr,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            draw_f0=draw_f0_par,
            scale_output_dirpath=scale_output_dirpath,
            scale_tb_prefix=scale_tb_prefix,
            sr=srs[scale_num],
            sr_f0=srs[0],
            sampler_16k=sampler_16k,
            checkpoint=checkpoint_scale,
            distributed=distributed,
            rank=rank
        )

        G_curr = trainer_scale.train()

        for p in G_curr.parameters():
            p.requires_grad = False
        for p in D_curr.parameters():
            p.requires_grad = False
        G_curr.eval()
        D_curr.eval()

        Gs.append(G_curr)
        Ds.append(D_curr)

        del D_curr, G_curr, train_dataloader, val_dataloader, train_dataset, val_dataset, trainer_scale
    return

#
# Main script
#


log = logging.getLogger(__name__)

@hydra.main(config_path="conf/run_config.yaml", strict=True)
def main(args):
    ## define distributed run
    dist_args= {'distributed': False}
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        try:
            rank = int(os.environ['LOCAL_RANK'])
        except KeyError:
            rank = 0
        torch.cuda.set_device(rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            dist_args['world_size'] = int(os.environ["WORLD_SIZE"])
            dist_args['distributed'] = dist_args['world_size'] > 1
        else:
            dist_args['world_size'] = 1
        if dist_args['distributed']:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if rank != 0:
        sys.stdout = open(os.devnull, "w")

    sr = args.sr

    # Convert filepaths
    original_dirpath = Path(hydra.utils.get_original_cwd())
    input_path = original_dirpath.joinpath(args.paths.input_data)
    output_dirpath = Path.cwd()

    ## save run args
    torch.save(args, 'args.pth')
    copy(input_path / 'loudness.json', output_dirpath / 'loudness.json')

    if args.paths.checkpoint:
        checkpoint = original_dirpath.joinpath(args.paths.checkpoint)
    else:
        checkpoint = None

    # logdir
    log_dirpath = original_dirpath.joinpath(
        'logs', output_dirpath.parent.name, output_dirpath.name)

    # Tensorboard, write logs into it
    if rank == 0:
        writer = SummaryWriter(log_dirpath)
    else:
        writer = None

    # Seed script
    if args.workspace.manualSeed is None:
        args.workspace.manualSeed = random.randint(1, 10000)
    random.seed(args.workspace.manualSeed)
    torch.manual_seed(args.workspace.manualSeed)

    # load sample
    srs = utils.create_srs(sr, args.num_scales, args.scale_factor)

    init_models_partial = partial(
        init_models,
        device=device,
        dparams=args.discriminator_params,
        gparams=args.generator_params
    )

    loss_gen = TotalLoss(hp_recon=args.optim.hp_recon, hp_adv=args.optim.hp_adv, hp_f0=args.optim.hp_f0)
    if args.optim.hp_f0:
        small_crepe = load_crepe(original_dirpath.joinpath('data_utils/crepe_models/small.pth'), device, 'small')
    else:
        small_crepe = None

    trainer_partial = partial(
        Trainer,
        device=device,
        epochs=args.optim.epochs,
        beta1=args.optim.beta1,
        gamma=args.optim.gamma,
        lr_d=args.optim.lr_d,
        lr_g=args.optim.lr_g,
        niter=args.optim.niter,
        loss_gen=loss_gen,
        log_writer=writer,
        hp_f0=args.optim.hp_f0,
        hp_adv=args.optim.hp_adv,
        f0_model=small_crepe,
        optim_type=args.optim.type,
        disc_start=args.optim.disc_start,
        log_audio=args.log_audio
    )

    train(
        output_dirpath,
        trainer=trainer_partial,
        srs=srs,
        init_models=init_models_partial,
        device=device,
        dataset_len=args.optim.dataset_len,
        input_path=input_path,
        duration=args.duration,
        max_val=args.max_val,
        max_val_f0=args.max_val_f0,
        batch_size=args.optim.batch_size,
        batch_size_min=args.optim.batch_size_min,
        num_workers=args.optim.num_workers,
        noise=args.noise,
        use_prev_weights = args.optim.use_prev_weights,
        checkpoint=checkpoint,
        distributed=dist_args['distributed'],
        rank=rank,
        world_size=dist_args['world_size'],
    )


if __name__ == "__main__":
    main()
