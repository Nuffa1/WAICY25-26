from train_funcs import *
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import glob

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"Using {DEVICE}")

    # Data setup
    IMG_H = 64
    IMG_W = 256
    IMG_CHANNELS = 1
    MAX_SEQ_LEN = 50
    BATCH_SIZE = 32

    # Load dataset
    data_transforms = T.Compose([
        FixedHeightResize(target_height=64),
        PadToWidth(target_width=256),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    train_dataset = HW_Dataset(data_root='IIIT-HW-Hindi_v1', data_type='train', transform=data_transforms)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_dataset = HW_Dataset(data_root='IIIT-HW-Hindi_v1', data_type='val', transform=data_transforms)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Model params
    VOCAB_SIZE = len(train_dataset.char_to_int)
    EMBEDDING_DIM = 256
    CONDITION_DIM = 128  # The output size of TextEncoder
    Z_DIM = 100  # Noise dimension
    LR_D = 0.0001
    LR_G = 0.0001
    BETA1 = 0.0
    BETA2 = 0.9
    D_STEPS = 5  # Train D 5 times every time G is trained
    LAMBDA_GP = 10.0
    NUM_EPOCHS = 150

    # Initialize Models
    text_encoder = TextEncoder(VOCAB_SIZE, EMBEDDING_DIM, CONDITION_DIM, CONDITION_DIM).to(DEVICE)
    gen = Generator(Z_DIM, CONDITION_DIM, IMG_CHANNELS, IMG_H, IMG_W).to(DEVICE)
    disc = Discriminator(CONDITION_DIM, IMG_CHANNELS, IMG_H, IMG_W).to(DEVICE)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Loss and Optimizers
    opt_disc = optim.Adam(disc.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    opt_gen = optim.Adam(
        list(gen.parameters()) + list(text_encoder.parameters()),
        lr=LR_G,
        betas=(BETA1, 0.999)
    )

    start_epoch = 1
    start_idx = 0
    first_new_epoch = True
    try:
        # Find all checkpoint files (we'll save everything into a single file per epoch)
        checkpoints = glob.glob("checkpoints/checkpoint_epoch_*.pth")
        if not checkpoints:
            raise FileNotFoundError  # Skip to 'except' block if no checkpoints

        # Find the latest epoch number
        latest_epoch_num = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoints])

        # Create the file path for the latest epoch
        checkpoint_path = f"checkpoints/checkpoint_epoch_{latest_epoch_num}.pth"

        if os.path.exists(checkpoint_path):
            print(f"Loading models and optimizers from epoch {latest_epoch_num}...")

            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

            # Load Model Weights
            gen.load_state_dict(checkpoint['gen_state_dict'])
            disc.load_state_dict(checkpoint['disc_state_dict'])
            text_encoder.load_state_dict(checkpoint['encoder_state_dict'])

            # Load Optimizer States (CRUCIAL for proper resumption)
            opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])

            # Set the starting epoch for the training loop
            start_epoch = latest_epoch_num
            start_idx = checkpoint['batch_idx']

            print(f"All states loaded successfully. Resuming training from batch {start_idx}")

        else:
            print("Checkpoint file for latest epoch is missing. Initializing new weights.")
            gen.apply(weights_init)
            disc.apply(weights_init)
            text_encoder.apply(weights_init)

    except (FileNotFoundError, ValueError, IndexError):
        print("No valid checkpoints found. Initializing new weights...")
        gen.apply(weights_init)
        disc.apply(weights_init)
        text_encoder.apply(weights_init)

    # We'll use a fixed batch of noise and text to see G's progress
    # We take a batch from the val set, so this is new to the model
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(DEVICE)
    fixed_batch_data = next(iter(val_loader))
    fixed_real_images, fixed_text_ids = fixed_batch_data
    fixed_text_ids = fixed_text_ids.to(DEVICE)
    fixed_real_images = fixed_real_images.to(DEVICE)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Save the fixed real batch for comparison
    vutils.save_image(fixed_real_images, "outputs/real_samples.png", normalize=True)

    print("Starting Training...")

    progress_bar = tqdm(total = int(np.ceil(train_dataset.__len__() / BATCH_SIZE) * (NUM_EPOCHS - start_epoch + 1)) - start_idx)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        for batch_idx, (real_images, text_ids) in enumerate(train_loader):
            if not (first_new_epoch and batch_idx < start_idx):
                real_images = real_images.to(DEVICE)
                text_ids = text_ids.to(DEVICE)

                # Get the condition vector (c)
                condition = text_encoder(text_ids)
                noise = torch.randn(real_images.size(0), Z_DIM).to(DEVICE)

                # --- Train Discriminator ---
                for _ in range(D_STEPS):
                    opt_disc.zero_grad()

                    # Add noise and label smoothing
                    real_labels = torch.empty(real_images.size(0), 1).uniform_(0.7, 1.0).to(DEVICE)

                    # Train with real images
                    # Use condition.detach() so text encoder is not updated
                    d_real_output = disc(real_images, condition.detach()).reshape(-1)

                    # Train with fake images
                    with torch.no_grad():
                        fake_images = gen(noise, condition.detach())
                    d_fake_output = disc(fake_images.detach(), condition.detach()).reshape(-1)

                    # Calculate gradient penalty
                    gp = calculate_gradient_penalty(
                        disc,
                        real_images,
                        fake_images.detach(),
                        condition.detach(),
                        DEVICE,
                        LAMBDA_GP
                    )

                    d_loss = d_fake_output.mean() - d_real_output.mean() + gp
                    d_loss.backward()
                    opt_disc.step()

                # --- Train Generator ---
                opt_gen.zero_grad()

                # Generate new fake images
                fake_images_gen = gen(noise, condition)

                # See what the discriminator thinks (no detaching)
                # g_output = disc(fake_images_gen, condition_gen).reshape(-1)
                g_output = disc(fake_images_gen, condition).reshape(-1)

                # Calculate loss (Generator wants discriminator to think they are real)
                g_loss = -g_output.mean()

                # Backprop (updates both generator and text encoder)
                g_loss.backward()
                opt_gen.step()

                # Logging
                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {batch_idx + 1}/{len(train_loader)}] "
                        f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
                    )
                    # Save generated images every once in a while
                    with torch.no_grad():
                        fixed_condition = text_encoder(fixed_text_ids)
                        fake_samples = gen(fixed_noise, fixed_condition)
                        vutils.save_image(
                            fake_samples,
                            f"outputs/fake_samples_epoch_{epoch}.png",
                            normalize=True
                        )

                    # Save a model training checkpoint so we don't lose progress
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'gen_state_dict': gen.state_dict(),
                        'disc_state_dict': disc.state_dict(),
                        'encoder_state_dict': text_encoder.state_dict(),
                        'opt_gen_state_dict': opt_gen.state_dict(),
                        'opt_disc_state_dict': opt_disc.state_dict(),
                    }
                    torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch}.pth")
                    # Remove old checkpoints to save storage space
                    if f"checkpoint_epoch_{epoch - 1}.pth" in os.listdir("checkpoints"):
                        os.remove(f"checkpoints/checkpoint_epoch_{epoch - 1}.pth")

                progress_bar.update(1)

        first_new_epoch = False

    print("Training finished.")
