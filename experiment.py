from src.encoders import SimpleRasterizeEncoder, GasfEncoder, NegGasfEncoder
from src.models import Generator, Discriminator
from src.dataset import get_dataset
from src.data_utils import get_gp, get_ks, get_model_data
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from random import randint
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser("Train a WGAN with GP on image-encoded data.")
parser.add_argument("--encoding", type=str, default="simple", choices=['simple', 'relative', 'gasf', 'relative_gasf'], help="Encoding method.")
parser.add_argument("--load", type=str, default="", help="Path to trained weights dir (to continue training).")
parser.add_argument("--logdir", type=str, default="experiments", help="Directory to save experiment results to.")
parser.add_argument("--outdir", type=str, default="models", help="Directory to save checkpoints to.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
parser.add_argument("--critic-iters", type=int, default=5, help="Number of critic iterations per generator iteration.")
parser.add_argument("--tag", type=str, default="", help="Tag to add to experiment name.")
parser.add_argument("--cuda", action="store_true", help="Use CUDA.")
parser.add_argument("--steps", type=float, default=10, help="Number of steps to train for (thousands).")
parser.add_argument("--gp-weight", type=float, default=10, help="Weight of the gradient penalty.")
parser.add_argument("--checkpoint-every", type=int, default=2000, help="Save a checkpoint every this many steps.")
parser.add_argument("--metrics-every", type=int, default=500, help="Save intensive metrics every this many steps.")
parser.add_argument("--dataset", type=str, default="data/eurusd_minute.csv", help="Path to eurusd dataset.")

args = parser.parse_args()

if args.seed is None:
    args.seed = randint(0, 9999)

device = torch.device('cuda:0' if args.cuda else 'cpu')
print("Using device:", device)

cudnn.benchmark = True

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run = args.tag + "_" + date if args.tag else date
logdir = os.path.join(args.logdir, run)
outdir = os.path.join(args.outdir, run)

if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

writer = SummaryWriter(logdir)
writer.add_text("args", str(args))

print("Loading dataset...")
dataset = get_dataset(args.encoding, path=args.dataset)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


print("Processing data for later KS metric calculation...")
standardize = lambda x: (x-x.mean())/x.std()

simple_encoder = SimpleRasterizeEncoder()
gasf_encoder = GasfEncoder()
neg_gasf_encoder = NegGasfEncoder()

decoder = {
    'simple': lambda dataset: [standardize(np.diff(simple_encoder.decode(x))) for x in dataset],
    'relative': lambda dataset: [standardize(gasf_encoder.decode(x)) for x in dataset],
    'relative_gasf': lambda dataset: [standardize(neg_gasf_encoder.decode(x)) for x in dataset],
    'gasf': lambda dataset: [standardize(np.diff(gasf_encoder.decode(x))) for x in dataset],
}[args.encoding]

dataset_data = np.concatenate(decoder(dataset))



print("Loading models...")
g = Generator(channels=1)
c = Discriminator(channels=1)

if args.load:
    g.load_state_dict(torch.load(os.path.join(args.load, "g.pt")))
    c.load_state_dict(torch.load(os.path.join(args.load, "c.pt")))
    print(f"Loaded weights from {args.load}.")
else:
    g._initialize_weights()

g.to(device)
c.to(device)

g_optim = torch.optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
c_optim = torch.optim.Adam(c.parameters(), lr=1e-4, betas=(0.5, 0.9))
# g_optim = torch.optim.RMSprop(g.parameters(), lr=5e-5)
# c_optim = torch.optim.RMSprop(c.parameters(), lr=5e-5)

losses = {"g": [], "c": [], "gp": []}


static_noise = torch.randn(args.batch_size, 100).to(device)
writer.add_images("Dataset sample", next(iter(data_loader)), 0)


print("Starting training...")

steps = int(args.steps * 1000)
checkpoint_every = args.checkpoint_every
metrics_every = args.metrics_every
stats_every = 50

checknum=0
stepnum = 0

epochs = steps//len(data_loader)
for epoch in range(1, epochs+1):
    loader = tqdm(data_loader)
    loader.set_description(f"Epoch [{epoch}/{epochs}]")
    for x_real in loader:
        x_real = x_real.to(device)
        
        # Train the critic
        noise = torch.randn(x_real.size(0), 100).to(device)
        x_fake = g(noise)
        x_real_pred = c(x_real)
        x_fake_pred = c(x_fake)

        gp = get_gp(x_real, x_fake, critic=c, device=device)
        c_loss = -(torch.mean(x_real_pred) - torch.mean(x_fake_pred))*0.5 + gp*args.gp_weight

        c_optim.zero_grad()
        c_loss.backward(retain_graph=(stepnum%args.critic_iters==0))
        c_optim.step()

        losses["c"].append(c_loss.item())
        losses["gp"].append(gp.item())

        # Train the Generator       
        if stepnum % args.critic_iters == 0:
            noise = torch.randn(x_real.size(0), 100).to(device)
            x_fake = g(noise)
            x_fake_pred = c(x_fake)
            g_loss = -torch.mean(x_fake_pred)
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            losses["g"].append(-g_loss.item())
            

        if stepnum % stats_every == 0:
            writer.add_scalars("Losses", {
                "generator": -np.mean(losses['g'][-10:]),
                "critic": -np.mean(losses['c'][-10:]),
                "gradient_penalty": np.mean(losses['gp'][-10:]),
                }, stepnum)
            
            static_samples = g(static_noise)
            dynamic_samples = x_fake
            writer.add_images("Static sample", static_samples, stepnum)
            writer.add_images("Dynamic sample", dynamic_samples, stepnum)


        if stepnum % metrics_every == 0:
            # Calculate and save KS metric
            model_data = get_model_data(g, decoder, 1000, device)
            ks = get_ks(g, decoder, dataset_data, device=device, n=1000)
            writer.add_scalar("KS", ks, stepnum)
            writer.add_histogram("Model data", model_data, stepnum)


        if stepnum % checkpoint_every == 0:
            # Save model
            checkpoint_dir = os.path.join(outdir, "checkpoints", f"{checknum}")
            os.makedirs(checkpoint_dir)

            torch.save(g.state_dict(), os.path.join(checkpoint_dir, "g.pt"))
            torch.save(g.state_dict(), os.path.join(outdir, "g.pt"))

            torch.save(c.state_dict(), os.path.join(checkpoint_dir, "c.pt"))
            torch.save(c.state_dict(), os.path.join(outdir, "c.pt"))
            checknum += 1

        stepnum+=1


print("Finished training.")
writer.close()
