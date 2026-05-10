import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 784 # dimensionality of flattened input images
HIDDEN_DIM = 256 # specifies the number of hidden 
LATENT_DIM = 16 # dimensionality of the latent space z 
CONDITION_DIM = 256 # dimensionality of caption embeddings

BATCH_SIZE = 64
LEARNING_RATE = 1e-3 # optimizer learning rate
NUM_EPOCHS = 20

BETA = 1.0

class ConditionalVAE(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim, 
            latent_dim,
            condition_dim
    ):
        super().__init__()

        self.input_dim = input_dim # dimensionality of images
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condtion_dim = condition_dim # dimensionality of caption embeddings

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.log_var_layer = nn.Linear(latent_dim, latent_dim)

        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    

    def reparameterize(self, mu, log_var):
        """
        The function enables backpropagation through sampling process.
        Transforms it using: z = mu + sd*eps.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
    

    def condition_on_caption(self, z, caption_embedding):
        """
        Caption embedding is projected into the latent dimensionality and combined
        with sampled latent vector.
        """
        projected_caption = self.condition_projector(
            caption_embedding.float()
        )

        return z + projected_caption
    

    def forward(self, x, caption_embedding):
        """
        Forward pass:
        1. Concatenate image and caption embedding.
        2. Encode the combined input.
        3. Compute latent mean and log-variance. 
        4. Sample latent vector using reparametrization.
        5. Condition the latent vector on caption embedding.
        6. Decode the conditioned latent vector into a reconstructed image. 
        """
        # Concatenate image + caption embedding
        encoder_input = torch.cat(
            [x, caption_embedding],
            dim=1
        )

        encoded = self.encoder(encoder_input)

        mu = self.mu(encoded)
        log_var = self.log_var_layer(encoded)

        z = self.reparameterize(mu, log_var)

        conditioned_z = self.condition_on_caption(
            z,
            caption_embedding
        )

        decoded = self.decoder(conditioned_z)

        return encoded, decoded, mu, log_var
    

    def sample(self, num_samples, caption_embedding):
        """
        Generate new samples conditioned on caption embeddings.
        """
        with torch.no_grad():
            z = torch.randn(
                num_samples,
                self.latent_dim
            ).to(DEVICE)

            # Condition on caption embeddings
            conditioned_z = self.condition_on_caption(
                z,
                caption_embedding
            )

            samples = self.decoder(conditioned_z)

        return samples
    

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        recon_x,
        x.view(-1, recon_x.shape[-1]),
        reduction="sum"
    )

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_cvae(
    X_train,
    caption_embeddings,
    learning_rate=1e-3,
    num_epochs=10,
    batch_size=32
):
    
    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    caption_embeddings = torch.tensor(
        caption_embeddings,
        dtype=torch.float32
    )

    dataset = torch.utils.data.TensorDataset(
        X_train,
        caption_embeddings
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )


    model = ConditionalVAE(
        input_dim=X_train.shape[1],
        hidden_dim=256,
        latent_dim=16,
        condition_dim=caption_embeddings.shape[1]
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, captions) in enumerate(train_loader):
            data = data.to(DEVICE)
            captions = captions.to(DEVICE)

            encoded, decoded, mu, log_var = model(data, captions)

            loss = loss_function(decoded, data, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader.dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"loss={epoch_loss:.4f}"
        )

    return model

