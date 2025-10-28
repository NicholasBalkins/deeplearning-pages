# Deep Learning — Atividade VAE

## O que eu queria fazer
Treinar um VAE no Fashion-MNIST e mostrar que ele aprende a comprimir e reconstruir imagens. A ideia é ver o encoder gerando μ e logσ², aplicar o reparameterization trick e o decoder tentando refazer a imagem. No final, comparar originais vs. reconstruções, olhar o espaço latente em 2D e gerar imagens novas amostrando z ~ N(0, I).

## O que eu fiz
Peguei o Fashion-MNIST em CSV, normalizei as imagens pra [0,1] e separei treino/val (90/10). Montei um VAE simples (MLP): entrada 784 (28×28), duas camadas ocultas, latente de 2 dimensões e saída com sigmoide. Treinei com Adam (lr 1e-3), batch 256, por 10 épocas. Registrei três coisas por época: perda total, perda de reconstrução (BCE) e KL.

## O que aconteceu
O loss total foi caindo certinho. A parte de reconstrução desceu de ~306 pra ~253 por amostra, mostrando que o modelo aprendeu padrões básicos das peças. A KL ficou ali na faixa ~6–6.5, que é o que empurra o latente pra perto de uma normal padrão. Com z=2, as reconstruções ficam reconhecíveis, mas um pouco borradas. No latente 2D, as classes tendem a formar grupos.


## Logs do treino
Epoch 1/10  | train=312.6787 (rec 306.5060, kl 6.1727) | val=275.4814 (rec 268.6384, kl 6.8429)  
Epoch 2/10  | train=273.4665 (rec 267.0483, kl 6.4181) | val=269.5273 (rec 263.2228, kl 6.3044)  
Epoch 3/10  | train=269.4855 (rec 263.2802, kl 6.2053) | val=266.6567 (rec 260.7225, kl 5.9343)  
Epoch 4/10  | train=267.1733 (rec 261.0109, kl 6.1624) | val=265.0443 (rec 259.0338, kl 6.0105)  
Epoch 5/10  | train=265.5084 (rec 259.3227, kl 6.1856) | val=263.3511 (rec 257.0348, kl 6.3163)  
Epoch 6/10  | train=264.2982 (rec 258.1057, kl 6.1925) | val=262.6569 (rec 256.7305, kl 5.9264)  
Epoch 7/10  | train=263.4492 (rec 257.2423, kl 6.2069) | val=261.4873 (rec 255.2323, kl 6.2550)  
Epoch 8/10  | train=262.7956 (rec 256.5756, kl 6.2199) | val=261.1750 (rec 254.9435, kl 6.2315)  
Epoch 9/10  | train=262.0082 (rec 255.7559, kl 6.2523) | val=260.9220 (rec 254.5064, kl 6.4156)  
Epoch 10/10 | train=261.5091 (rec 255.2296, kl 6.2795) | val=259.8884 (rec 253.4013, kl 6.4871)

## O que dá pra ver nas figuras
- **Originais vs. Reconstruções:** dá pra reconhecer a peça (camiseta, bota etc.), mesmo com leve borrado — esperado pro setup.
- **Latente (μ, 2D):** os pontos tendem a se agrupar por classe.
- **Amostras novas:** quando amostro z ~ N(0, I), o decoder gera imagens plausíveis do conjunto.

## Conclusão
O VAE fez o que precisava: comprimiu as imagens num espaço 2D razoável e conseguiu reconstruir com qualidade “ok” pra um modelo básico. Se eu quiser reconstruções mais nítidas, posso treinar mais épocas, aumentar capacidade (hidden_dim) ou subir a dimensão do latente (depois dá pra visualizar com PCA/t-SNE/UMAP). Se eu quiser um latente mais “organizado”, posso testar um β-VAE (β>1), sabendo que isso normalmente sacrifica um pouco da fidelidade nas reconstruções.


## Codigo
```python
import os, zipfile, math, argparse, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_csv_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"Nenhum CSV em {zip_path}")
        with z.open(csv_names[0]) as f:
            return pd.read_csv(f)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=2):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),  # saída em [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar

def vae_loss(xr, x, mu, logvar, beta=1.0):
    bce = nn.functional.binary_cross_entropy(xr, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + beta * kl) / x.size(0), bce / x.size(0), kl / x.size(0)

def make_grid(imgs, nrow=8):
    if imgs.dim() == 2:
        imgs = imgs.view(-1, 1, 28, 28)
    n = imgs.size(0)
    ncol = nrow
    nrow_grid = math.ceil(n / ncol)
    canvas = torch.zeros(1, nrow_grid * 28, ncol * 28, device=imgs.device)
    k = 0
    for r in range(nrow_grid):
        for c in range(ncol):
            if k >= n: break
            canvas[:, r*28:(r+1)*28, c*28:(c+1)*28] = imgs[k]
            k += 1
    return canvas.squeeze(0).detach().cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="Pasta com fashion-mnist_train.csv.zip e fashion-mnist_test.csv.zip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--report_dir", type=str, default="vae_report")
    parser.add_argument("--subsample_train", type=int, default=60000, help="limite de linhas p/ treino (acelera testes)")
    parser.add_argument("--subsample_test", type=int, default=10000, help="limite de linhas p/ teste")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    train_zip = os.path.join(args.data_dir, "fashion-mnist_train.csv.zip")
    test_zip  = os.path.join(args.data_dir, "fashion-mnist_test.csv.zip")
    assert os.path.exists(train_zip), f"Faltando {train_zip}"
    assert os.path.exists(test_zip),  f"Faltando {test_zip}"

    train_df = load_csv_from_zip(train_zip)
    test_df  = load_csv_from_zip(test_zip)

    label_col = "label" if "label" in train_df.columns else train_df.columns[0]
    pixel_cols = [c for c in train_df.columns if c != label_col]
    assert len(pixel_cols) == 784, "Esperava 784 colunas de pixel (28x28)."

    if len(train_df) > args.subsample_train:
        train_df = train_df.sample(n=args.subsample_train, random_state=args.seed)
    if len(test_df) > args.subsample_test:
        test_df = test_df.sample(n=args.subsample_test, random_state=args.seed)

    def df_to_tensors(df):
        y = torch.tensor(df[label_col].to_numpy(), dtype=torch.long)
        X = torch.tensor(df[pixel_cols].to_numpy(), dtype=torch.float32) / 255.0
        return X, y

    X_train_all, y_train_all = df_to_tensors(train_df)
    X_test, y_test = df_to_tensors(test_df)

    val_ratio = 0.1
    n_total = X_train_all.size(0)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    full_ds = TensorDataset(X_train_all, y_train_all)
    val_ds, train_ds = random_split(full_ds, [n_val, n_train], generator=torch.Generator().manual_seed(args.seed))
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=784, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_total":[], "val_total":[], "train_recon":[], "val_recon":[], "train_kl":[], "val_kl":[]}

    def run_epoch(dl, train=True):
        if train: model.train()
        else: model.eval()
        total, recon_tot, kl_tot, nb = 0.0, 0.0, 0.0, 0
        with torch.set_grad_enabled(train):
            for xb, _ in dl:
                xb = xb.to(device)
                xr, mu, logvar = model(xb)
                loss, bce, kl = vae_loss(xr, xb, mu, logvar)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total += loss.item(); recon_tot += bce.item(); kl_tot += kl.item(); nb += 1
        return total/nb, recon_tot/nb, kl_tot/nb

    for e in range(1, args.epochs+1):
        tr_tot, tr_rec, tr_kl = run_epoch(train_loader, True)
        va_tot, va_rec, va_kl = run_epoch(val_loader,   False)
        history["train_total"].append(tr_tot)
        history["val_total"].append(va_tot)
        history["train_recon"].append(tr_rec)
        history["val_recon"].append(va_rec)
        history["train_kl"].append(tr_kl)
        history["val_kl"].append(va_kl)
        print(f"Epoch {e}/{args.epochs} | train={tr_tot:.4f} (rec {tr_rec:.4f}, kl {tr_kl:.4f}) | val={va_tot:.4f} (rec {va_rec:.4f}, kl {va_kl:.4f})")

    os.makedirs(args.report_dir, exist_ok=True)
    assets = os.path.join(args.report_dir, "assets")
    os.makedirs(assets, exist_ok=True)

    import numpy as np
    epochs = np.arange(1, len(history["train_total"])+1)

    def save_plot(x, ys, labels, title, out_path, ylabel="loss"):
        plt.figure()
        for y, lab in zip(ys, labels):
            plt.plot(x, y, label=lab)
        plt.xlabel("epoch"); plt.ylabel(ylabel); plt.title(title); plt.legend()
        plt.savefig(out_path, bbox_inches="tight"); plt.close()

    save_plot(epochs,
              [history["train_total"], history["val_total"]],
              ["train_total", "val_total"],
              "VAE Loss (total)",
              os.path.join(assets, "loss_total.png"))

    save_plot(epochs,
              [history["train_recon"], history["val_recon"]],
              ["train_recon", "val_recon"],
              "Reconstruction Loss",
              os.path.join(assets, "loss_recon.png"))

    save_plot(epochs,
              [history["train_kl"], history["val_kl"]],
              ["train_kl", "val_kl"],
              "KL Loss",
              os.path.join(assets, "loss_kl.png"))

    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(test_loader))
        xb = xb.to(device)
        xr, _, _ = model(xb)
        originals = xb[:32].view(-1, 1, 28, 28)
        recons    = xr[:32].view(-1, 1, 28, 28)

    orig_grid  = make_grid(originals, 8)
    recon_grid = make_grid(recons,    8)

    plt.figure(); plt.imshow(orig_grid, cmap="gray"); plt.axis("off")
    plt.title("Originais (teste)")
    plt.savefig(os.path.join(assets, "originals.png"), bbox_inches="tight", pad_inches=0); plt.close()

    plt.figure(); plt.imshow(recon_grid, cmap="gray"); plt.axis("off")
    plt.title("Reconstruções")
    plt.savefig(os.path.join(assets, "reconstructions.png"), bbox_inches="tight", pad_inches=0); plt.close()

    all_mu, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            mu, _ = model.encode(xb)
            all_mu.append(mu.cpu().numpy()); all_y.append(yb.numpy())
    all_mu = np.concatenate(all_mu, axis=0); all_y = np.concatenate(all_y, axis=0)

    plt.figure()
    plt.scatter(all_mu[:,0], all_mu[:,1], c=all_y, s=6)
    plt.xlabel("z1"); plt.ylabel("z2"); plt.title("Espaço latente (μ) — teste")
    plt.savefig(os.path.join(assets, "latent.png"), bbox_inches="tight"); plt.close()

    with torch.no_grad():
        z = torch.randn(64, args.latent_dim, device=device)
        samples = model.decode(z).view(-1, 1, 28, 28)
    sample_grid = make_grid(samples, 8)

    plt.figure(); plt.imshow(sample_grid, cmap="gray"); plt.axis("off")
    plt.title("Amostras (z ~ N(0, I))")
    plt.savefig(os.path.join(assets, "samples.png"), bbox_inches="tight", pad_inches=0); plt.close()

    torch.save(model.state_dict(), os.path.join(args.report_dir, "vae_state.pt"))

    index_html = f"""<!doctype html>
<html lang="pt-br">
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VAE — Fashion-MNIST</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:24px;line-height:1.5}}img{{max-width:100%;height:auto;display:block;margin:8px 0 24px}}</style>
<h1>VAE em Fashion-MNIST</h1>
<p>
Dataset: Fashion-MNIST (CSV). Normalização para [0,1]. Split treino/val=90/10.<br>
Arquitetura: MLP (784→{args.hidden_dim}→{args.hidden_dim}→μ,logσ²; z∈ℝ^{args.latent_dim}; decoder simétrico com sigmoide).<br>
Treino: Adam(1e-3), batch {args.batch_size}, épocas {args.epochs}. Loss = BCE (reconstrução) + KL.
</p>
<h2>Curvas de loss</h2>
<img src="assets/loss_total.png" alt="Loss total">
<img src="assets/loss_recon.png" alt="Reconstruction loss">
<img src="assets/loss_kl.png" alt="KL loss">
<h2>Reconstruções</h2>
<img src="assets/originals.png" alt="Originais"><img src="assets/reconstructions.png" alt="Reconstruções">
<h2>Espaço latente</h2>
<img src="assets/latent.png" alt="Latente">
<h2>Amostras</h2>
<img src="assets/samples.png" alt="Amostras">
<h2>Conclusões</h2>
<p>
O VAE com z=2 reconstrói padrões básicos e organiza classes em regiões do latente.
Dimensões latentes maiores tendem a melhorar a reconstrução e piorar a visualização direta;
para z&gt;3, recomenda-se redução de dimensionalidade (PCA/t-SNE/UMAP).
</p>
</html>"""
    with open(os.path.join(args.report_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

if __name__ == "__main__":
    main()

```