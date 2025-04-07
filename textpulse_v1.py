import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from time import time
import logging
from collections import Counter
import os
import pickle
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Mapping and Chunking ---
def text_to_chunks(file_path, min_N=32, max_N=128, overlap_ratio=0.5):
    """Convert text file to adaptive overlapping chunks."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    byte_vals = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8)  # Full 0-255 range
    
    # Adaptive chunk size based on file length
    total_length = len(byte_vals)
    N = min(max_N, max(min_N, int(np.sqrt(total_length / 4))))  # Aim for ~4 chunks, clamped
    chunk_size = N * N
    overlap = int(N * overlap_ratio)
    step = chunk_size - overlap * N
    
    chunks = []
    for i in range(0, max(1, len(byte_vals) - chunk_size + 1), step):
        chunk = byte_vals[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant', constant_values=32)
        chunks.append(chunk.reshape(N, N) / 255.0)  # Normalize to [0, 1]
    return chunks, text, N

def infer_corruption_mask(chunk, window_size=5):
    """Infer corruption using local entropy."""
    N = chunk.shape[0]
    entropy_map = np.zeros_like(chunk)
    padded = np.pad(chunk, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode='edge')
    
    for i in range(N):
        for j in range(N):
            window = padded[i:i+window_size, j:j+window_size]
            entropy_map[i, j] = stats.entropy(window.flatten(), base=2)
    
    # Low entropy (uniformity) or extreme values indicate potential corruption
    mean_entropy = entropy_map.mean()
    std_entropy = entropy_map.std()
    mask = ((entropy_map < mean_entropy - std_entropy) | (chunk < 0.05) | (chunk > 0.95)).astype(float)
    return mask

def chunks_to_text(chunks, N, overlap, original_length):
    """Reassemble chunks into text with overlap blending."""
    chunk_size = N * N
    step = chunk_size - overlap * N
    total_length = min(original_length, step * (len(chunks) - 1) + chunk_size if chunks else chunk_size)
    flat_data = np.zeros(total_length, dtype=np.uint8)
    weights = np.zeros(total_length, dtype=float)
    
    for i, chunk in enumerate(chunks):
        start = i * step
        end = min(start + chunk_size, total_length)
        chunk_flat = (chunk.flatten() * 255).astype(np.uint8)
        chunk_flat = np.clip(chunk_flat, 0, 255)  # Full byte range
        
        weight = np.ones(chunk_size)[:end - start]
        if i > 0:
            weight[:overlap * N] = np.linspace(0, 1, overlap * N)[:end - start]
        if i < len(chunks) - 1 and end - start > overlap * N:
            weight[-overlap * N:] = np.linspace(1, 0, overlap * N)[-overlap * N:]
        
        flat_data[start:end] += chunk_flat * weight
        weights[start:end] += weight
    
    flat_data = flat_data / np.maximum(weights, 1e-6)
    return bytes(flat_data.astype(np.uint8)).decode('utf-8', errors='replace')

# --- N-gram Context Model ---
def build_ngram_model(text, n=2):
    """Build bigram model from text."""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return Counter(ngrams)

def ngram_loss(u_pred, ngram_model, N):
    """Penalize unlikely bigrams."""
    text_pred = chunks_to_text([u_pred], N, 0, N * N)  # Single chunk
    ngrams_pred = [text_pred[i:i+2] for i in range(len(text_pred) - 1)]
    loss = 0
    for ngram in ngrams_pred:
        loss += 1.0 / (ngram_model.get(ngram, 1e-3) + 1e-6)
    return loss / len(ngrams_pred)

# --- EnhancedMIONet ---
class EnhancedMIONet(nn.Module):
    def __init__(self, theta_dim=3, bc_dim=4, rho_dim=64, hidden_dim=512, num_quantum_weights=18):
        super().__init__()
        self.device = device
        
        self.branch_theta = nn.Sequential(nn.Linear(theta_dim, hidden_dim), nn.Tanh(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.branch_bc = nn.Sequential(nn.Linear(bc_dim, hidden_dim), nn.Tanh(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.branch_rho = nn.Sequential(nn.Linear(rho_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.quantum_weights = nn.Parameter(torch.randn(num_quantum_weights, device=device) * 0.1)
        self.qdev = qml.device("default.qubit.torch", wires=6, torch_device="cuda" if torch.cuda.is_available() else "cpu")
        
        @qml.qnode(self.qdev, interface='torch')
        def quantum_circuit(inputs, weights):
            inputs = torch.pi * (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
            for i in range(6):
                qml.RY(inputs[..., i], wires=i)
                qml.RX(weights[i], wires=i)
            qml.CNOT(wires=[0, 1]); qml.CNOT(wires=[2, 3]); qml.CNOT(wires=[4, 5])
            for i in range(6):
                qml.RZ(inputs[..., i + 3], wires=i)
                qml.RX(weights[i + 6], wires=i)
            qml.CZ(wires=[1, 2]); qml.CZ(wires=[3, 4])
            for i in range(6):
                qml.RY(weights[i + 12], wires=i)
            qml.CNOT(wires=[0, 5]); qml.CZ(wires=[2, 4])
            return [qml.expval(qml.PauliZ(w)) for w in range(6)]
        
        self.quantum_circuit = quantum_circuit
    
    def quantum_layer(self, theta, bc, rho):
        rho_stats = torch.stack([rho.mean(dim=1), rho.std(dim=1), rho.max(dim=1)[0]], dim=-1)
        inputs = torch.cat([theta, bc, rho_stats], dim=-1)[:, :9]
        q_out = torch.stack(self.quantum_circuit(inputs, self.quantum_weights), dim=-1)
        return q_out.mean(dim=-1)
    
    def forward(self, theta, bc, rho, X_ref):
        batch_size = theta.shape[0]
        n_points = X_ref.shape[-2]
        
        theta_out = self.branch_theta(theta).unsqueeze(1).expand(-1, n_points, -1)
        bc_out = self.branch_bc(bc).unsqueeze(1).expand(-1, n_points, -1)
        rho_out = self.branch_rho(rho).unsqueeze(1).expand(-1, n_points, -1)
        trunk_out = self.trunk(X_ref)
        if trunk_out.dim() == 2:
            trunk_out = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)
        
        combined = theta_out * bc_out * rho_out * trunk_out
        q_factor = self.quantum_layer(theta, bc, rho).unsqueeze(-1).unsqueeze(-1)
        u = self.final_layer(combined) * (1 + q_factor)
        return torch.clamp(u, 0, 1)

# --- eQ-DIMON with Enhanced Features ---
class eQ_DIMON:
    def __init__(self, batch_size=32, initial_lr=0.001, ngram_weight=0.3, memory_dir="memory"):
        self.device = device
        self.model = None  # Initialized with dynamic rho_dim later
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None
        self.ngram_weight = ngram_weight
        self.memory_dir = memory_dir
        self.playbook = {}
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
        self.load_memory()

    def load_memory(self):
        """Load past model and playbook."""
        model_path = os.path.join(self.memory_dir, "model.pth")
        playbook_path = os.path.join(self.memory_dir, "playbook.pkl")
        if os.path.exists(model_path):
            # Model loaded dynamically later with correct rho_dim
            logging.info("Model state found, will load with correct chunk size.")
        if os.path.exists(playbook_path):
            with open(playbook_path, 'rb') as f:
                self.playbook = pickle.load(f)
            logging.info("Loaded playbook.")

    def save_memory(self, task_id, N):
        """Save model and playbook."""
        model_path = os.path.join(self.memory_dir, f"model_task_{task_id}_N{N}.pth")
        playbook_path = os.path.join(self.memory_dir, "playbook.pkl")
        torch.save(self.model.state_dict(), model_path)
        with open(playbook_path, 'wb') as f:
            pickle.dump(self.playbook, f)
        logging.info(f"Saved model and playbook for task {task_id} with N={N}.")

    def get_user_feedback(self, chunk, mask, chunk_idx):
        """Interactive user feedback to refine corruption mask."""
        plt.imshow(chunk * 255, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Chunk {chunk_idx} - Suspected Corruption")
        plt.show()
        plt.imshow(mask, cmap='hot')
        plt.title(f"Chunk {chunk_idx} - Inferred Corruption Mask")
        plt.show()
        
        response = input(f"Chunk {chunk_idx}: Adjust mask? (y/n): ").lower()
        if response == 'y':
            good_regions = input("Enter good region indices (e.g., '0-10,20-30' or 'none'): ")
            bad_regions = input("Enter bad region indices (e.g., '15-25' or 'none'): ")
            
            if good_regions != 'none':
                for region in good_regions.split(','):
                    start, end = map(int, region.split('-'))
                    mask[start:end, :] = 0
            if bad_regions != 'none':
                for region in bad_regions.split(','):
                    start, end = map(int, region.split('-'))
                    mask[start:end, :] = 1
        return mask

    def _train_batch(self, batch, X_ref, ngram_model):
        theta_batch, bc_batch, rho_batch, u_true_batch = batch
        theta_batch, bc_batch, rho_batch, u_true_batch = [x.to(self.device) for x in batch]
        X_ref_tensor = torch.tensor(X_ref.reshape(-1, 2), dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        u_pred = self.model(theta_batch, bc_batch, rho_batch, X_ref_tensor).squeeze(-1).view(-1, self.N, self.N)
        data_loss = torch.mean((u_pred - u_true_batch)**2)
        context_loss = sum(ngram_loss(u_pred[i].cpu().numpy(), ngram_model, self.N) for i in range(u_pred.shape[0])) / u_pred.shape[0]
        loss = data_loss + self.ngram_weight * context_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), data_loss.item()

    def train(self, chunks, original_text, N, epochs=20, task_id="user_input"):
        """Train on real user-input chunks."""
        self.N = N
        self.model = EnhancedMIONet(rho_dim=N).to(device)  # Dynamic rho_dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Load existing model if available
        model_path = os.path.join(self.memory_dir, f"model_task_{task_id}_N{N}.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logging.info(f"Loaded model for N={N} from previous task.")
        
        X_ref = np.stack(np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N)), axis=-1)
        ngram_model = build_ngram_model(original_text[:N*N])
        
        data = []
        for idx, chunk in enumerate(chunks):
            mask = infer_corruption_mask(chunk)
            mask = self.get_user_feedback(chunk, mask, idx)  # Refine with user input
            theta = np.array([mask.mean(), mask.std(), np.random.uniform(-0.5, 0.5)])
            bc = np.array([chunk[0, 0], chunk[-1, 0], chunk[0, -1], chunk[-1, -1]])
            data.append((theta, bc, mask.flatten(), chunk))
        
        theta_data, bc_data, rho_data, u_data = zip(*data)
        theta_data = torch.tensor(np.stack(theta_data), dtype=torch.float32)
        bc_data = torch.tensor(np.stack(bc_data), dtype=torch.float32)
        rho_data = torch.tensor(np.stack(rho_data), dtype=torch.float32)
        u_data = torch.tensor(np.stack(u_data), dtype=torch.float32)
        
        n_samples = len(data)
        n_train = n_samples
        n_train_batches = n_train // self.batch_size
        n_train = n_train_batches * self.batch_size
        
        train_data = [theta_data[:n_train], bc_data[:n_train], rho_data[:n_train], u_data[:n_train]]
        
        self.playbook[task_id] = {
            'ngram_model': ngram_model,
            'corruption_info': {'entropy_based': True, 'mean_entropy': mask.mean()},
            'data_structure': 'user_text',
            'chunk_size': N
        }
        
        train_losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for i in range(0, n_train, self.batch_size):
                batch = [x[i:i + self.batch_size] for x in train_data]
                loss, _ = self._train_batch(batch, X_ref, ngram_model)
                epoch_loss += loss
            train_losses.append(epoch_loss / n_train_batches)
            logging.info(f"Epoch {epoch}: Train Loss={train_losses[-1]:.6f}")
        
        self.save_memory(task_id, N)
        return train_losses

    def predict(self, chunks, N, original_length, task_id="user_input"):
        """Predict and reassemble real chunks."""
        X_ref = torch.tensor(np.stack(np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N)), axis=-1).reshape(-1, 2), 
                            dtype=torch.float32, device=self.device)
        recovered_chunks = []
        
        for chunk in chunks:
            mask = infer_corruption_mask(chunk)
            theta = np.array([mask.mean(), mask.std(), np.random.uniform(-0.5, 0.5)])
            bc = np.array([chunk[0, 0], chunk[-1, 0], chunk[0, -1], chunk[-1, -1]])
            inputs = [torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0) for x in [theta, bc, mask.flatten()]]
            with torch.no_grad():
                u_pred = self.model(*inputs, X_ref).squeeze().cpu().numpy().reshape(N, N)
            recovered_chunks.append(u_pred)
        
        return chunks_to_text(recovered_chunks, N, int(N * 0.5), original_length)

    def visualize_anomalies(self, chunks, task_id="user_input"):
        """Visualize corruption heatmap for first chunk."""
        mask = infer_corruption_mask(chunks[0])
        plt.imshow(mask, cmap='hot')
        plt.title(f"Task {task_id} - Inferred Corruption Heatmap (Chunk 0)")
        plt.colorbar(label='Corruption Likelihood')
        plt.savefig(os.path.join(self.memory_dir, f"anomaly_{task_id}.png"))
        plt.close()
        logging.info(f"Anomaly heatmap saved to memory/anomaly_{task_id}.png")

# --- Main Execution ---
if __name__ == '__main__':
    start_time = time()
    
    # User input
    file_path = input("Enter the path to your text file: ")
    if not os.path.exists(file_path):
        logging.error("File not found!")
        exit(1)
    
    logging.info("Loading and chunking user file...")
    chunks, original_text, N = text_to_chunks(file_path)
    original_length = len(original_text.encode('utf-8', errors='ignore'))
    logging.info(f"File split into {len(chunks)} chunks with N={N}.")
    
    # Initialize and train
    eq_dimon = eQ_DIMON(batch_size=32, ngram_weight=0.3)
    logging.info("Training on user-input chunks...")
    train_losses = eq_dimon.train(chunks, original_text, N, epochs=20, task_id="user_input")
    
    # Visualize anomalies
    eq_dimon.visualize_anomalies(chunks)
    
    # Predict and reassemble
    logging.info("Recovering text from chunks...")
    recovered_text = eq_dimon.predict(chunks, N, original_length)
    
    # Save output
    output_path = "recovered_text.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(recovered_text)
    logging.info(f"Recovered text saved to {output_path}")
    
    # Display snippets
    logging.info(f"Original (first 50 chars): {original_text[:50]}...")
    logging.info(f"Recovered (first 50 chars): {recovered_text[:50]}...")
    logging.info(f"Total Runtime: {time() - start_time:.2f} seconds")