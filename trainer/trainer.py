import torch
from .decompose import Decompose
import .train_step
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import sys
sys.path.append('..')

import model.model as model
import os
from tqdm import tqdm
import numpy
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args["model"] == "CVFModel":
            f = model.CVFModel()
        elif args["model"] == "SMCVFModel":
            f = model.SMCVFModel()
        else:
            raise Exception(f"Not implemented model {args['model']}")

        if args["trainer"] == "TrainStep":
            trainer = train_step.TrainStep
        elif args["trainer"] == "TrainStepSM":
            trainer = train_step.TrainStepSM
        else:
            raise Exception(f"Not implemented trainer {args['trainer']}")

        self.trainer = torch.nn.DataParallel(trainer(f).to(args["device"])) if args["usedataparallel"] else trainer(f).to(args["device"])
        self.trainer.requires_grad=False
        
        self.optimizer = torch.optim.Adam(self.trainer.module.f.parameters() if args["usedataparallel"]
                                        else self.trainer.f.parameters(), lr=args["lr"], weight_decay=1e-9, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args["scheduler_step"], gamma=0.99)
        self.epochs = args["epochs"]
        
        if args["dataset"] == 'realhaze':
            from dataset import RealHazyDataset
            dataset = RealHazyDataset
        elif args["dataset"] == 'reside':
            from dataset import RESIDEHazyDataset
            dataset = RESIDEHazyDataset
        else:
            raise Exception("Not implemented dataset")

        train_dataset = dataset(root=args["dataroot"], mode='train', )
        g = torch.Generator()
        g.manual_seed(args["seed"])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, args["batchsize"], shuffle=True, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g, drop_last=True)
        val_dataset = dataset(root=args["dataroot"], mode='validation', )
        self.val_loader = torch.utils.data.DataLoader(val_dataset, args["valbatchsize"], shuffle=False, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g)

        self.out_dir = args["outdir"]
        num = 0
        while os.path.isdir(self.out_dir+str(num)):
            num+=1
        self.out_dir = self.out_dir + str(num)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        
        self.decompose = torch.nn.DataParallel(Decompose().to(args["device"])) if args["usedataparallel"] else Decompose().to(args["device"])
        self.decompose.requires_grad=False

    def train(self):
        train_losses = []
        validation_losses = []

        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
            prog_bar.set_description(f'[Progress] epoch {epoch}/{self.epochs} - saving at {self.out_dir}')
            train_loss = self.train_epoch()
            self.scheduler.step()

            if epoch % self.args["validateevery"] == self.args["validateevery"] - 1:
                validation_loss = self.validation_epoch()

            train_losses.append(train_loss)
            while len(validation_losses) < len(train_losses):
                validation_losses.append(validation_loss)

            plt.clf()
            plt.plot(train_losses, label='train')
            plt.plot(validation_losses, label='validation')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, 'train_plot.png'))

            if epoch % 10 == 0:
                torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"]
                                    else self.trainer.f.state_dict(),
                            'optim': self.optimizer.state_dict()},
                            os.path.join(self.out_dir, 'checkpoints', 'checkpoint.tar'))

        validation_loss = self.validation_epoch()
        torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"] 
                            else self.trainer.f.state_dict()},
                    os.path.join(self.out_dir, 'checkpoints', 'final.tar'))

    def train_epoch(self):
        num_samples = 0
        accum_losses = 0
        prog_bar = tqdm(self.train_loader)
        for batchIdx, img in enumerate(prog_bar):
            img = img.to(self.args["device"])
            loss = self.trainer(img).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainer.module.f.parameters() if self.args["usedataparallel"]
                                            else self.trainer.f.parameters(), 1)
            self.optimizer.step()
            num_samples += img.size(0)
            accum_losses += loss.item() * img.size(0)

            prog_bar.set_description(f'[Train] batch {batchIdx}/{len(prog_bar)} loss {loss:.2f} acc {accum_losses/num_samples:.2f}')

        return accum_losses / num_samples

    @torch.no_grad()
    def validation_epoch(self):
        num_samples = 0
        accum_losses = 0
        prog_bar = tqdm(self.val_loader)
        f = self.trainer.module.f if self.args["usedataparallel"] else self.trainer.f
        if self.args["usedataparallel"]:
            f = torch.nn.DataParallel(f)
        for batchIdx, img in enumerate(prog_bar):
            img = img.to(self.args["device"])
            loss = self.trainer(img).mean()
            num_samples += img.size(0)
            accum_losses += loss.item() * img.size(0)
                
            noiseD, noiseI, clean = f(img)
            T, A, captureNoise = self.decompose(noiseD, noiseI, clean)
            rec = clean + clean * noiseD + noiseI

            for idx in range(img.size(0)):
                torchvision.utils.save_image(noiseD[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noiseD.png'))
                torchvision.utils.save_image(noiseI[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noiseI.png'))
                torchvision.utils.save_image(clean[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_clean.png'))
                torchvision.utils.save_image(img[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_img.png'))
                torchvision.utils.save_image(T[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_T.png'))
                torchvision.utils.save_image(A[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_A.png'))
                torchvision.utils.save_image(captureNoise[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_captureNoise.png'))
                torchvision.utils.save_image(rec[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_reconstuct.png'))
            
            prog_bar.set_description(f'[Val] batch {batchIdx}/{len(prog_bar)} loss {loss:.2f} acc {accum_losses/num_samples:.2f}')

        return accum_losses / num_samples


if __name__=='__main__':
    print('Module file')