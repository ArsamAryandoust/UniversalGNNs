from models import AutoEncoder, VAE
from datasets import ClimARTDataset, BuildingElectricityDataset
import torch
from torch.utils.data import DataLoader

climart_train = ClimARTDataset(normalize=True)
BE_train = BuildingElectricityDataset(normalize=True)

climart_loader = DataLoader(climart_train, batch_size=1024, num_workers=64, drop_last=True)
BE_loader = DataLoader(BE_train, batch_size=1024, num_workers=64, drop_last=True)

ae_climart = AutoEncoder(climart_train.input_dim, 512)
ae_climart.load_state_dict(torch.load("checkpoints/encoders/ClimARTDataset/AutoEncoder_512.pt"))
vae_climart = VAE(climart_train.input_dim, 512, False)
vae_climart.load_state_dict(torch.load("checkpoints/encoders/ClimARTDataset/VAE_512.pt"))

ae_BE = AutoEncoder(BE_train.input_dim, 512)
ae_BE.load_state_dict(torch.load("checkpoints/encoders/BuildingElectricityDataset/AutoEncoder_512.pt"))
vae_BE = VAE(BE_train.input_dim, 512, False)
vae_BE.load_state_dict(torch.load("checkpoints/encoders/BuildingElectricityDataset/VAE_512.pt"))

# how do I pick close points in the input space to check if they are close in the latent space?
#  - calculate clusters?
def compute_cos_similarity(x: torch.Tensor):
    x_norm = x / x.norm(dim=1)[:, None]
    res = torch.mm(x_norm, x_norm.transpose(0,1))
    return res

def compute_euclidean_dist(x: torch.Tensor):
    x_norm = x / x.norm(dim=1)[:, None]
    res = torch.cdist(x_norm, x_norm)
    return res


def compute_mean_distance(loader: DataLoader, ae: AutoEncoder, vae: VAE):
    ae_sim = 0.0
    vae_sim = 0.0
    ae_dist = 0.0
    vae_dist = 0.0
    for features, labels in loader:
        feature_similarity = compute_cos_similarity(features)
        feature_dist = compute_euclidean_dist(features)
        latent_ae = ae(features)
        latent_vae = vae(features)
        latent_ae_similarity = compute_cos_similarity(latent_ae)
        latent_vae_similarity = compute_cos_similarity(latent_vae)
        latent_ae_dist = compute_euclidean_dist(latent_ae)
        latent_vae_dist = compute_euclidean_dist(latent_vae)

        ae_dist += (latent_ae_dist - feature_dist).abs().mean()
        vae_dist += (latent_vae_dist - feature_dist).abs().mean()

        ae_sim += (latent_ae_similarity - feature_similarity).abs().mean()
        vae_sim += (latent_vae_similarity - feature_similarity).abs().mean()
    
    ae_dist = ae_dist / len(loader)
    vae_dist = vae_dist / len(loader)
    ae_sim = ae_sim / len(loader)
    vae_sim = vae_sim / len(loader)

    return ae_dist, vae_dist, ae_sim, vae_sim

climart_ae_dist, climart_vae_dist, climart_ae_sim, climart_vae_sim = compute_mean_distance(climart_loader, ae_climart, vae_climart)
BE_ae_dist, BE_vae_dist, BE_ae_sim, BE_vae_sim = compute_mean_distance(BE_loader, ae_BE, vae_BE)

print("Climart AE dist :", climart_ae_dist, "sim:", climart_ae_sim)
print("Climart VAE dist:", climart_vae_dist, "sim:", climart_vae_sim)
print("BE AE dist :", BE_ae_dist, "sim:", BE_ae_sim)
print("BE VAE dist:", BE_vae_dist, "sim:", BE_vae_sim)