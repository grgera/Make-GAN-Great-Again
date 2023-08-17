
import wandb
import os, argparse, glob, librosa, itertools, time, torch, random
import numpy as np
import json
import soundfile as sf
import torch.optim as optim
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from generator import Generator
from discriminator import ResWiseMSD, ResWiseMPD
from dataset import *
from loss import *
from utils import *

def train(args, h):
    
    mel_list = sorted(glob.glob(os.path.join(h.out_dir[2:] + '/mel', '*.npy')))
    audio_list = sorted(glob.glob(os.path.join(h.out_dir[2:] + '/audio', '*.npy')))
    test_mel = sorted(glob.glob(os.path.join(h.valid_dir[2:] + '/mel', '*.npy')))
    testset = [torch.from_numpy(np.load(x)).float().unsqueeze(0) for x in test_mel]

    G = Generator(h).to(h.device)
    mpd = ResWiseMPD().to(h.device)
    msd = ResWiseMSD().to(h.device)

    optim_g = AdamW(G.parameters(), h.learning_rate, betas=[h.b1, h.b2])
    optim_d = AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.b1, h.b2])

    step, epochs = 0, -1
    if args.checkpoint is not None:
        fg = torch.load(args.checkpoint)
        G.load_state_dict(fg['G'])
        optim_g.load_state_dict(fg['optim_g'])
        mpd.load_state_dict(fg['mpd'])
        msd.load_state_dict(fg['msd'])
        optim_d.load_state_dict(fg['optim_d'])
        step = fg['step'],
        epochs = fg['epoch']
        step = step[0]
        print('Load Status: Step %d' % (step))
        
    scheduler_g = ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=epochs)
    scheduler_d = ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=epochs)

    torch.backends.cudnn.benchmark = True
    
    trainset = MelDataset(mel_list, audio_list, h)
    train_loader = DataLoader(trainset, batch_size=h.batch_size, num_workers=0, shuffle=True, drop_last=True)
    
    G.train()
    mpd.train()
    msd.train()
    
    for epoch in range(max(0, epochs), args.training_epochs):
        g_losses, d_losses = [], []
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for (mel, audio) in tqdm(train_loader):
            x = mel.to(h.device)
            y = audio.to(h.device)
            
            y_g_hat = G(x)
            mel_spec = get_mel(y_g_hat, h, 'train')
            
            # Discriminator
            optim_d.zero_grad()
                
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            d_loss = loss_disc_s + loss_disc_f
            
            d_loss.backward()
            optim_d.step()

            d_losses.append(d_loss.item())
            
            # Generator
            optim_g.zero_grad()

            loss_mel = F.l1_loss(x, mel_spec) * 45
                
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                
            g_loss = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            g_loss.backward()
            optim_g.step()

            g_losses.append(g_loss.item())
            
            #Logging and checkpointing
            step += 1
            if step % h.log_step == 0:
                print('step: {}, D_loss: {:.3f}, G_loss: {:.3f}, {:.3f} sec/step'.format(
                    step, d_loss, g_loss, (time.time() - start) / h.log_step))
                start = time.time()

            if step % h.wandb_step == 0:
                wandb.log({
                    "Step": step,
                    "Generator Loss": np.mean(g_losses),
                    "Discriminator Loss": np.mean(d_losses)})
                g_losses = []
                d_losses = []

            if step % h.checkpoint_step == 0:
                save_dir = './fg_model/' + args.name
                with torch.no_grad():
                    for i, mel_test in enumerate(testset):
                        g_audio = G(mel_test.to(h.device))
                        g_audio = g_audio.squeeze().cpu()
                        audio = (g_audio.numpy() * 32768)
                        sf.write(os.path.join(save_dir, 'generated-{}-{}.wav'.format(step, i)),
                                    audio.astype('int16'),
                                    h.sample_rate)

                print("Saving checkpoint")
                torch.save({
                    'G': G.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'mpd': mpd.state_dict(),
                    'msd': msd.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch
                }, os.path.join(save_dir, 'fg-{}.pt'.format(step)))
            
        scheduler_g.step()
        scheduler_d.step()
            
def main():
    print('Training process starts!')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', '-p', default=None)
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--name', '-n', required=True)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    save_dir = os.path.join('./fg_model', a.name)
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        h.device = 'cuda'
    else:
        h.device = 'cpu'

    wandb.init(name='Fre-GAN', 
           project='personal project', 
           resume= "allow",
           tags=['LJ dataset', 'Train Run'])

    train(a, h)

if __name__ == '__main__':
    main() 