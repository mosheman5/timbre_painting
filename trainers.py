import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import soundfile as sf
from models.losses import MultiResolutionSTFTLoss
from utils import utils, writers, sampling
from tqdm import tqdm
from data_utils.spectral_feats import py_get_activation


class Trainer():
    def __init__(self,
                 netD,
                 netG,
                 device,
                 train_dataset,
                 val_dataset,
                 train_dataloader,
                 val_dataloader,
                 epochs,
                 beta1,
                 gamma,
                 lr_d,
                 lr_g,
                 niter,
                 loss_gen,
                 draw_f0,
                 scale_output_dirpath,
                 scale_tb_prefix,
                 sr,
                 sr_f0,
                 log_writer,
                 optim_type,
                 disc_start,
                 hp_f0=0,
                 hp_adv=0,
                 f0_model=None,
                 sampler_16k=None,
                 checkpoint='',
                 distributed=False,
                 rank=0,
                 log_audio=False):

        # set parameters for stft loss
        self.sr = sr
        self.sr_f0 = sr_f0
        self.stft_loss = self.init_stft_loss()
        self.netD = netD
        self.netG = netG
        self.niter = niter
        self.epochs = epochs
        self.hp_f0 = hp_f0
        self.hp_adv = hp_adv
        self.f0_model = f0_model
        self.sampler_16k = sampler_16k
        self.loss_gen = loss_gen
        self.scale_output_dirpath = scale_output_dirpath
        self.log_writer = log_writer
        self.scale_tb_prefix = scale_tb_prefix
        self.draw_f0 = draw_f0
        self.device = device
        self.optim_type = optim_type
        self.disc_start = int(disc_start * epochs) - 1
        self.epochs_trained = 0
        self.distributed=distributed
        self.rank = rank
        self.optimizerD, self.optimizerG, self.schedulerD, self.schedulerG = \
            self.init_optimizers(lr_d, lr_g, beta1, gamma)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.partial_train_dataloader = train_dataloader
        self.partial_val_dataloader = val_dataloader
        self.loss_meter_train, self.loss_meter_val = {}, {}
        self.loss_meter_keys = ['errD_real', 'errD_fake', 'errD', 'rec_loss', 'f0_loss', 'errG', 'loss']
        self.init_losses_meter()
        self.log_audio=log_audio

        # load checkpoint
        if checkpoint:
            # checkpoint_path = checkpoint.joinpath('last.pth')
            self._load_checkpoint(checkpoint_path=checkpoint)


    def train(self):
        best_loss = 100000000
        ## Load pre-trained and save net
        if self.epochs - self.epochs_trained < 2:
            if self.rank == 0:
                self._save_checkpoint(f"{self.scale_output_dirpath}/last.pth")
            return self.netG
        ## Run training script
        for epoch in tqdm(range(self.epochs_trained, self.epochs)):
            self.reset_losses()
            self.init_data(epoch)
            self.train_epoch(epoch)
            self.eval_epoch(epoch)
            ## save checkpoint
            if self.rank == 0:
                self._save_checkpoint(f"{self.scale_output_dirpath}/last.pth")
                rec_loss = self.loss_meter_train['rec_loss'].summarize_epoch()
                if rec_loss < best_loss:
                    best_loss = rec_loss
                    self._save_checkpoint(f"{self.scale_output_dirpath}/best.pth")
                self.epochs_trained += 1
        return self.netG

    def train_epoch(self, epoch):
        self.netD.train()
        self.netG.train()
        for it in tqdm(range(self.niter)):

            f0, audio, loudness_list = next(self.train_iter)
            f0, audio = f0.to(self.device), audio.to(self.device)
            loudness_list = [loudness.to(self.device) for loudness in loudness_list]

            # train D
            errD_real, errD_fake, errD,  prev = self.iterD(f0, audio, loudness_list, epoch)
            if self.hp_adv and epoch > self.disc_start:
                errD.backward()
                self.optimizerD.step()

            # train G
            rec_loss, f0_loss, errG, loss, audio_gen = self.iterG(prev, audio, loudness_list, epoch)
            loss.backward()
            self.optimizerG.step()

            # update losses
            if self.rank == 0:
                self.update_losses(errD_real.detach(), errD_fake.detach(), errD.detach(), rec_loss.detach(),
                                   f0_loss.detach(), errG.detach(), loss.detach(), flag='train')

            self.schedulerG.step()
            if self.hp_adv and epoch > self.disc_start:
                self.schedulerD.step()

        if self.rank == 0:
            self.log(epoch, 'train', audio_gen, f0, audio)

    def eval_epoch(self, epoch):
        with torch.no_grad():
            self.netD.eval()
            self.netG.eval()
            for it in tqdm(range(max(self.niter//10, 1))):

                f0, audio, loudness_list = next(self.val_iter)
                f0, audio = f0.to(self.device), audio.to(self.device)
                loudness_list = [loudness.to(self.device) for loudness in loudness_list]
                # train D
                errD_real, errD_fake, errD, prev = self.iterD(f0, audio, loudness_list, epoch)

                # train G
                rec_loss, f0_loss, errG, loss, audio_gen = self.iterG(prev, audio, loudness_list, epoch)

                # update losses
                if self.rank == 0:
                    self.update_losses(errD_real.detach(), errD_fake.detach(), errD.detach(),
                                       rec_loss.detach(), f0_loss.detach(), errG.detach(), loss.detach(), flag='val')

        if self.rank == 0:
            self.log(epoch, 'val', audio_gen, f0, audio)

    def log(self, epoch, flag, audio_gen, f0, audio):
        # log gradients
        if flag == 'train':
            self.log_writer.add_scalar(
                f"{self.scale_tb_prefix}_grads/G/grad/tail",
                self.netG.module.last_conv_layers[-1].weight_v.grad.abs().mean(),
                epoch,
            )
            self.log_writer.add_scalar(
                f"{self.scale_tb_prefix}_grads/G/grad/head",
                self.netG.module.first_conv.weight_v.grad.abs().mean(),
                epoch,
            )

            if self.hp_adv and epoch > self.disc_start:
                self.log_writer.add_scalar(
                    f"{self.scale_tb_prefix}_grads/D/grad/tail",
                    self.netD.module.conv_layers[-1].weight_v.grad.abs().mean(), epoch
                )
                self.log_writer.add_scalar(
                    f"{self.scale_tb_prefix}_grads/D/grad/head",
                    self.netD.module.conv_layers[0].weight_v.grad.abs().mean(),
                    epoch,
                )
        # log losses
        for key in self.loss_meter_keys:
            if flag == 'train':
                self.log_writer.add_scalar(f"{self.scale_tb_prefix}_{flag}/{key}",
                                       self.loss_meter_train[key].summarize_epoch(), epoch)
            elif flag == 'val':
                self.log_writer.add_scalar(f"{self.scale_tb_prefix}_{flag}/{key}",
                                           self.loss_meter_val[key].summarize_epoch(), epoch)
        # log audios
        if self.log_audio:
            audios = [audio_gen, f0, audio]
            names = ['audio_gen', 'f0', 'audio']
            srs = [self.sr, self.sr_f0, self.sr]

            for audio, name, sr in zip(audios, names, srs):
                audio_numpy = audio[0].clamp(-1,1).squeeze().detach().cpu().numpy()
                sf.write(
                    f"{self.scale_output_dirpath}/{name}_sample_{flag}.wav",
                    audio_numpy,
                    sr
                )
                audio = sampling.resample_torch(audio[0], sr, 16000).clamp(-1, 1).detach().squeeze()
                self.log_writer.add_audio(
                    f"{self.scale_tb_prefix}_{flag}/{name}", audio, epoch, sample_rate=16000)
                nfft = int(sr / (48000 / 4096))
                sample_stft = writers.show_spec(audio_numpy, nfft=nfft, sr=sr)
                self.log_writer.add_figure(f'{self.scale_tb_prefix}_{flag}/stft_{name}', sample_stft, epoch)

    def iterD(self, f0, audio, loudness_list, epoch):

        # train with fake
        # load prev
        prev = self.draw_f0(in_s=f0, loudness_list=loudness_list)
        if self.hp_adv and epoch > self.disc_start:
            self.netD.zero_grad()
            self.netG.zero_grad()

            output = self.netD(audio)
            errD_real = F.mse_loss(output, output.new_ones(output.size())).mean()

            fake = self.netG(prev.detach(), loudness_list[-1])
            fake = fake.detach()
            output = self.netD(fake)
            errD_fake = F.mse_loss(output, output.new_zeros(output.size())).mean()

            errD = errD_real + errD_fake
            # errD = errD_real + errD_fake
        else:
            errD_real = torch.tensor(0)
            errD_fake = torch.tensor(0)
            errD = torch.tensor(0)

        return errD_real, errD_fake, errD,  prev

    def iterG(self, prev, audio, loudness_list, epoch):
        self.netG.zero_grad()

        # generate_audio
        audio_gen = self.netG(prev.detach(), loudness_list[-1].detach())

        #generate_fake
        if self.hp_adv and epoch > self.disc_start:
            self.netD.zero_grad()
            output = self.netD(audio_gen)
            errG = F.mse_loss(output, output.new_ones(output.size())).mean()
        else:
            errG = torch.tensor(0)
        # f0 loss
        if self.hp_f0:
            in_features = py_get_activation(audio.squeeze(1), self.sr, self.f0_model,
                                            layer=18, grad=False, sampler=self.sampler_16k)
            in_features = in_features.detach()
            out_features = py_get_activation(audio_gen.squeeze(1), self.sr, self.f0_model,
                                             layer=18, grad=True, sampler=self.sampler_16k)
            f0_loss = F.l1_loss(out_features, in_features)
        else:
            f0_loss = torch.tensor(0)


        # Reconstruction loss
        sc_loss, mag_loss = self.stft_loss(audio_gen.squeeze(1), audio.squeeze(1))
        rec_loss = sc_loss + mag_loss

        loss = self.loss_gen(errG=errG, rec_loss=rec_loss, f0_loss=f0_loss)

        return rec_loss, f0_loss, errG, loss, audio_gen

    def init_data(self, epoch):
        self.train_dataset.set_epoch(epoch)
        self.train_dataset.split_keys_epoch()
        self.val_dataset.set_epoch(epoch)
        self.val_dataset.split_keys_epoch()
        train_dataloader = self.partial_train_dataloader(dataset=self.train_dataset)
        val_dataloader = self.partial_train_dataloader(dataset=self.val_dataset)
        self.train_iter = iter(train_dataloader)
        self.val_iter = iter(val_dataloader)

    def init_stft_loss(self):
        fft_size = 2048
        fft_sizes = [int(fft_size / 2 ** i) for i in range(0, 6)]
        hop_sizes = [fft_size//4 for fft_size in fft_sizes]
        win_lengths = fft_sizes
        return MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths)

    def init_optimizers(self, lr_d, lr_g, beta1, gamma):
        if self.optim_type == 'adam':
            optimizerD = optim.Adam(self.netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
            optimizerG = optim.Adam(self.netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
        elif self.optim_type == 'radam':
            optimizerD = RAdam(self.netD.parameters(), lr=lr_d)
            optimizerG = RAdam(self.netG.parameters(), lr=lr_g)
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizerD, milestones=[int(self.niter * self.epochs * 0.5)], gamma=gamma
        )
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizerG, milestones=[int(self.niter * self.epochs * 0.5)], gamma=gamma
        )

        return optimizerD, optimizerG, schedulerD, schedulerG

    def init_losses_meter(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key] = utils.LossMeter(key)
            self.loss_meter_val[key] = utils.LossMeter(key)

    def reset_losses(self):
        for key in self.loss_meter_keys:
            self.loss_meter_train[key].reset()
            self.loss_meter_val[key].reset()

    def update_losses(self, errD_real, errD_fake, errD, rec_loss, f0_loss, errG, loss, flag):
        losses = [errD_real, errD_fake, errD, rec_loss, f0_loss, errG, loss]
        for key, current_loss in zip(self.loss_meter_keys, losses):
            if flag == 'train':
                self.loss_meter_train[key].add(current_loss.data.cpu().numpy().mean())
            elif flag == 'val':
                self.loss_meter_val[key].add(current_loss.data.cpu().numpy().mean())
            else:
                raise ValueError('accept train or flag only!')

    def _save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {"optimizer": {
            "generator": self.optimizerG.state_dict(),
            "discriminator": self.optimizerD.state_dict(),
        }, "scheduler": {
            "generator": self.schedulerG.state_dict(),
            "discriminator": self.schedulerD.state_dict(),
        }, "epochs": self.epochs_trained,
        "model": {"generator": self.netG.module.state_dict(),
            "discriminator": self.netD.module.state_dict(),}
        }

        torch.save(state_dict, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
        """
        print('Loading checkpoint')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.epochs_trained = state_dict["epochs"]
        self.optimizerG.load_state_dict(state_dict["optimizer"]["generator"])
        self.optimizerD.load_state_dict(state_dict["optimizer"]["discriminator"])
        self.schedulerG.load_state_dict(state_dict["scheduler"]["generator"])
        self.schedulerD.load_state_dict(state_dict["scheduler"]["discriminator"])
        self.netG.module.load_state_dict(state_dict["model"]["generator"])
        self.netD.module.load_state_dict(state_dict["model"]["discriminator"])
