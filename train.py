from models import Discriminator, Generator
from dataset import ForexData
from trainer import WGanGpTrainer, WGanTrainer
import torch.optim as optim
import argparse
import torch


def main():
    parser = argparse.ArgumentParser('Train a Wasserstein imaged-based-GAN on time-series data.')
    parser.add_argument('--data-csv', type=str, default='data/eurusd_minute.csv', help="Path to the data csv file.")
    parser.add_argument('--data-column', type=str, default='BidClose', help="Name of the column in the csv file that contains the time-series data.")
    parser.add_argument('--gan-type', default='wgan-gp', help='Type of GAN to use. One of "wgan" or "wgan-gp".')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0, help='Overwrite learning rate. If 0, use a default value.')
    parser.add_argument('--critic-iters', type=int, default=5, help='Number of critic iterations per generator iteration.')
    parser.add_argument('--weight-clip', type=float, default=1e-2, help='Clip weights to enforce Lipshitz constraint.')
    parser.add_argument('--penalty-weight', type=float, default=10, help='Weight of the gradient penalty used to enforce Lipschitz constraint.')
    parser.add_argument('--load-checkpoint', default=None, help='Directory to load checkpoint from.')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Number of epochs between checkpoints.')
    parser.add_argument('--model-dir', default=None, help='Directory to save models to.')
    parser.add_argument('--tboard-dir', default=None, help='Directory to save tensorboard logs to.')

    args = parser.parse_args()

    LATENT_DIM=100
    SEQ_LENGTH=32

    # Load data
    import os
    from torch.utils.data import DataLoader

    # if os.path.isfile('./data/train_data.pt'):
    #     dataset = torch.load('./data/train_data.pt')
    # else:
    dataset = ForexData(args.data_csv, args.data_column,SEQ_LENGTH)
    # torch.save(dataset, './data/train_data.pt')

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create models
    g = Generator()
    d = Discriminator()
    g._initialize_weights()


    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make dirs
    if args.model_dir is not None:
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)
    if args.tboard_dir is not None:
        if not os.path.isdir(args.tboard_dir):
            os.makedirs(args.tboard_dir)
    
    # Define training methods
    trainers = {
        'wgan': WGanTrainer
            (
                generator=g, 
                critic=d,
                gen_optimizer=optim.RMSprop(g.parameters(), lr=args.lr if args.lr > 0 else 5e-5),
                critic_optimizer=optim.RMSprop(d.parameters(), lr=args.lr if args.lr > 0 else 5e-5),
                latent_dimension=LATENT_DIM,
                device=device,
                model_dir=args.model_dir,
                write_dir=args.tboard_dir,
                checkpoint=args.load_checkpoint,
                checkpoint_interval=args.checkpoint_interval,
                critic_iterations=args.critic_iters,
                weigth_clip=args.weight_clip,
            ),
        'wgan-gp': WGanGpTrainer
            (
                generator=g,
                critic=d,
                gen_optimizer=optim.Adam(g.parameters(), lr=args.lr if args.lr > 0 else 1e-4, betas=(0.0,0.9)),
                critic_optimizer=optim.Adam(d.parameters(), lr=args.lr if args.lr > 0 else 1e-4, betas=(0.0,0.9)),
                latent_dimension=LATENT_DIM,
                device=device,
                model_dir=args.model_dir,
                write_dir=args.tboard_dir,
                checkpoint=args.load_checkpoint,
                checkpoint_interval=args.checkpoint_interval,
                critic_iterations=args.critic_iters,
                gp_weight=args.penalty_weight,
            ),
    }

    # Train
    trainer = trainers.get(args.gan_type)
    if not trainer:
        raise ValueError(f'Invalid GAN type: {args.gan_type}')

    trainer.train(train_loader, args.epochs)



if __name__ == "__main__":
    main()