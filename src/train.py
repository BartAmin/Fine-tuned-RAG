import torch
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader

from src.model import EmbeddingDataset, NeuralNet, MNRLossWithMasking
from config    import DIM, EPOCHS, N_TRIALS, PATIENCE, BEST_MODEL


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for q, t, chunk_ids in loader:
        optimizer.zero_grad()
        loss = criterion(model(q), t, chunk_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_hit_rate(model, val_df, corpus_embeddings, corpus_store, top_k=5):
    model.eval()
    c_norm = F.normalize(corpus_embeddings, dim=1)
    hits   = 0
    with torch.no_grad():
        for _, row in val_df.iterrows():
            q_emb        = torch.tensor(row['question_embedding'], dtype=torch.float32).unsqueeze(0)
            q_emb        = model(q_emb)
            similarities = (c_norm @ F.normalize(q_emb, dim=1).T).squeeze(1)
            top_k_ids    = corpus_store.iloc[similarities.topk(top_k).indices.tolist()]['chunk_id'].tolist()
            hits        += int(row['chunk_id'] in top_k_ids)
    return hits / len(val_df)


def run_optuna(embeddings_train, val_df, corpus_embeddings, corpus_store):
    best_overall_rate = 0.0

    def objective(trial):
        nonlocal best_overall_rate

        params = {
            "hidden_dim":  trial.suggest_categorical("hidden_dim", [512, 1024, 2048]),
            "num_layers":  trial.suggest_int("num_layers", 1, 3),
            "dropout":     trial.suggest_float("dropout", 0.05, 0.3),
            "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "temperature": trial.suggest_float("temperature", 0.05, 0.2),
            "batch_size":  trial.suggest_categorical("batch_size", [32, 64]),
        }
        print(f"\nTrial {trial.number} | Params: {params}")

        train_loader = DataLoader(EmbeddingDataset(embeddings_train), batch_size=params['batch_size'], shuffle=True)

        model     = NeuralNet(DIM, params['hidden_dim'], params['num_layers'], params['dropout'])
        criterion = MNRLossWithMasking(temperature=params['temperature'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

        best_trial_rate  = 0.0
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer, criterion)
            hit_rate   = val_hit_rate(model, val_df, corpus_embeddings, corpus_store)
            print(f"  Trial {trial.number} | Epoch {epoch:2d}/{EPOCHS} | Train: {train_loss:.4f} | Val Hit Rate: {hit_rate:.2%}")

            if hit_rate > best_trial_rate:
                best_trial_rate  = hit_rate
                patience_counter = 0
                if hit_rate > best_overall_rate:
                    best_overall_rate = hit_rate
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "params":           params,
                        "best_hit_rate":    best_overall_rate,
                        "dim":              DIM,
                    }, BEST_MODEL)
                    print(f"  → New best model saved (hit rate: {best_overall_rate:.2%})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  → Early stopping at epoch {epoch}")
                    break

        return best_trial_rate

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction = "maximize",  
        sampler   = optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest trial:     {study.best_trial.number}")
    print(f"Best hit rate:  {study.best_value:.2%}")
    print(f"Best params:    {study.best_params}")

    return study


def load_best_model():
    checkpoint = torch.load(BEST_MODEL, weights_only=False)
    params     = checkpoint['params']
    dim        = checkpoint.get('dim', DIM)
    model      = NeuralNet(dim, params['hidden_dim'], params['num_layers'], params['dropout'])

    state_dict = checkpoint['model_state_dict']
    if 'residual_scale' not in state_dict:
        state_dict['residual_scale'] = torch.ones(1)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded | Hit Rate: {checkpoint.get('best_hit_rate', 'N/A')} | Params: {params}")
    return model