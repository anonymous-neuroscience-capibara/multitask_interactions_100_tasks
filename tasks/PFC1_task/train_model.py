import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model import FullModel
from load_data import load_and_create_dataset
import os
from scipy.stats import pointbiserialr


def train_epoch(model, dataloader, optimizer, alpha, n_interleave, beta_pred, beta_enc, beta_cons, beta_ent, beta_class=0.1, use_regularization=False, tau=0.01, M_reg=10):
    """
    Train for one epoch.
    
    Args:
        model: FullModel instance
        dataloader: DataLoader providing batches of data
        optimizer: Optimizer for updating model parameters
        alpha: Teacher forcing parameter
        n_interleave: Teacher forcing frequency
        beta_pred: Weight for prediction loss
        beta_enc: Weight for encoder loss
        beta_cons: Weight for consistency loss
        beta_class: Weight for classification loss
    """
    model.train()
    epoch_metrics = {
        'total_loss': 0,
        'prediction_loss': 0,
        'encoder_loss': 0,
        'consistency_loss': 0,
        'classification_loss': 0,
        'entropy_loss': 0,
        'accuracy': 0
    }
    
    n_correct = 0
    n_total = 0
    
    # Debug: Track encoded states and trajectories
    all_encoded_states = []
    all_trajectories = []
    all_initial_states = []  # Track initial states
    all_inputs = []  # Track inputs
    
    # Track lengths and labels for correlation analysis - convert to numpy arrays first
    all_lengths = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        trial_ids = batch['trial_id']
        lengths = batch['length']
        labels = batch['label']
        
        # Debug: Check initial states
        initial_states = model.rnn.get_initial_state(trial_ids)
        all_initial_states.append(initial_states.detach().numpy())
        
        # Debug: Check inputs if they exist
        if model.use_inputs:
            inputs = batch['inputs']
            all_inputs.append(inputs.detach().numpy())
        
        # Store lengths and labels for correlation analysis - convert to numpy arrays first
        all_lengths.append(lengths.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        # Forward pass with trial IDs
        outputs = model(
            batch, trial_ids, alpha, n_interleave,
            beta_pred, beta_enc, beta_cons, beta_class, beta_ent, use_regularization, tau, M_reg
        )
        
        # Debug: Store encoded states and trajectories
        with torch.no_grad():
            encoded_x, _ = model.encoder(batch['spikes'])
            if model.use_hierarchical:
                z_lat, _ = model.predict_sequence(
                    model.rnn, initial_states, batch['inputs'] if model.use_inputs else None,
                    encoded_x, alpha, n_interleave, trial_ids, batch['length']
                )
            else:
                z_lat, _ = model.predict_sequence(
                    model.rnn, initial_states, batch['inputs'] if model.use_inputs else None,
                    encoded_x, alpha, n_interleave, batch['length']
                )
            all_encoded_states.append(encoded_x.mean(dim=1).detach().numpy())
            all_trajectories.append(z_lat.mean(dim=1).detach().numpy())
        
        # Compute training-style accuracy (as before)
        predictions = (outputs['predictions'] > 0.5).float()
        n_correct += (predictions == batch['label']).sum().item()
        
        n_total += len(batch['label'])
        
        # Backward pass
        optimizer.zero_grad()
        outputs['total_loss'].backward()
        optimizer.step()
        
        # Update metrics
        for k, v in outputs.items():
            if k != 'total_loss' and k != 'predictions':
                epoch_metrics[k] += v
        epoch_metrics['total_loss'] += outputs['total_loss'].item()
    
    # Compute overall length-label correlation
    all_lengths = np.concatenate(all_lengths)
    all_labels = np.concatenate(all_labels)
    correlation, p_value = pointbiserialr(all_labels, all_lengths)
    
    # Average metrics
    n_batches = len(dataloader)
    for k in epoch_metrics:
        if k != 'accuracy':
            epoch_metrics[k] /= n_batches
    epoch_metrics['accuracy'] = n_correct / n_total
    
    return epoch_metrics

def train_model(dataloader, model, training_params):
    """
    Main training function.
    
    Args:
        dataloader: DataLoader instance providing batches of data
        model: FullModel instance (either newly created or loaded)
        training_params: Dictionary of training parameters
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize optimizer with appropriate parameter groups based on model type
    if model.use_hierarchical:
        # Hierarchical model: separate learning rates for different components
        param_groups = [
            {'params': [model.rnn.feature_vectors], 'lr': training_params['feature_lr']},
            {'params': [
                model.rnn.proj_A, 
                model.rnn.proj_W, 
                model.rnn.proj_h, 
                model.rnn.proj_z0,
                model.rnn.D,
                *([model.rnn.proj_C] if model.rnn.use_inputs else []),
                model.R_z_param
            ], 'lr': training_params['projection_lr']},
            {'params': list(model.decoder.parameters()) + list(model.encoder.parameters()), 'lr': training_params['encoder_lr']}
        ]
    else:
        # Non-hierarchical model: single learning rate for RNN parameters
        param_groups = [
            {'params': list(model.rnn.parameters()) + [model.R_z_param], 'lr': training_params['projection_lr']},
            {'params': list(model.decoder.parameters()) + list(model.encoder.parameters()), 'lr': training_params['encoder_lr']}
        ]
    
    optimizer = torch.optim.Adam(param_groups)
    
    # Create checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a unique identifier for this model configuration
    model_id = f"{model.session_name}_M{model.M}_P{model.P}_bin{model.bin_width:.3f}"
    if model.use_hierarchical:
        model_id += f"_hier_Nfeat{model.N_feat}"
    if not model.use_inputs:
        model_id += "_noinputs"
    model_id = model_id.replace('.', '_')
    
    checkpoint_path = f'{checkpoint_dir}/{model_id}_best_model.pth'
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(training_params['n_epochs']):
        # Compute current alpha value based on scheduling
        if training_params['use_alpha_scheduling']:
            progress = epoch / (training_params['n_epochs'] - 1)
            current_alpha = training_params['alpha_start'] + progress * (training_params['alpha_end'] - training_params['alpha_start'])
        else:
            current_alpha = training_params['alpha_start']
        
        metrics = train_epoch(
            model, dataloader, optimizer,
            current_alpha,
            training_params['n_interleave'],
            training_params['beta_pred'],
            training_params['beta_enc'],
            training_params['beta_cons'],
            training_params['beta_ent'],
            training_params.get('beta_class', 0.1),
            training_params['use_regularization'],
            training_params['tau'],
            training_params['M_reg']
        )
        
        # Print metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{training_params['n_epochs']}")
            print(f"Total Loss: {metrics['total_loss']:.4f}")
            print(f"Prediction Loss: {metrics['prediction_loss']:.4f}")
            print(f"Encoder Loss: {metrics['encoder_loss']:.4f}")
            print(f"Entropy Loss: {metrics['entropy_loss']:.4f}")
            print(f"Consistency Loss: {metrics['consistency_loss']:.4f}")
            print(f"Classification Loss: {metrics['classification_loss']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Save best model every 20 epochs
        if (epoch + 1) % 20 == 0:
            if metrics['total_loss'] < best_loss:
                best_loss = metrics['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'model_config': {
                        'M': model.M,
                        'P': model.P,
                        'decoder_type': model.decoder_type,
                        'use_inputs': model.use_inputs,
                        'session_name': model.session_name,
                        'bin_width': model.bin_width,
                        'n_trials': model.n_trials,
                        'N': model.N,
                        'input_dim': model.input_dim,
                        'use_hierarchical': model.use_hierarchical,
                        'N_feat': model.N_feat
                    },
                    'training_config': training_params
                }, checkpoint_path)
                print(f"\nSaved best model to {checkpoint_path}")

def create_new_model(model_params, dataset, session_name, bin_width):
    """Create a new model with the given parameters"""
    model = FullModel(
        M=model_params['M'],
        P=model_params['P'],
        N=dataset.n_neurons,  # Use actual number of neurons from dataset
        n_trials=dataset.n_trials,
        input_dim=dataset.n_inputs if dataset.use_inputs else 0,  # Use actual input dimension from dataset
        decoder_type=model_params['decoder_type'],
        use_hierarchical=model_params.get('use_hierarchical', False),
        N_feat=model_params.get('N_feat', 20),
        fix_R_z=True,
        use_inputs=dataset.use_inputs,
        learn_initial_states=model_params['learn_initial_states']  # Remove the .get() with default True
    )
    
    # Add session info to model for checkpoint naming
    model.session_name = session_name
    model.bin_width = bin_width
    model.decoder_type = model_params['decoder_type']
    
    return model

def load_model(checkpoint_path, dataset=None, use_hierarchical=True, N_feat=20, learn_initial_states=None):
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        dataset: Optional dataset instance to get n_trials if not in checkpoint
        use_hierarchical: Whether to load as hierarchical model (overrides checkpoint)
        N_feat: Number of features for hierarchical model (if use_hierarchical=True)
        learn_initial_states: Whether to learn initial states (if None, uses checkpoint value)
    """
    checkpoint = torch.load(checkpoint_path)
    model_config = checkpoint['model_config']
    
    # Get n_trials either from config or dataset
    n_trials = model_config.get('n_trials')
    if n_trials is None and dataset is not None:
        n_trials = dataset.n_trials
    elif n_trials is None:
        raise ValueError("n_trials not found in checkpoint and no dataset provided")
    
    # Use checkpoint value for learn_initial_states if not specified
    if learn_initial_states is None:
        learn_initial_states = model_config.get('learn_initial_states', False)
    
    # Create model with current configuration
    model = FullModel(
        M=model_config['M'],
        P=model_config['P'],
        N=model_config.get('N', dataset.n_neurons if dataset else None),
        n_trials=n_trials,
        input_dim=model_config.get('input_dim', dataset.n_inputs if dataset else 0),
        decoder_type=model_config.get('decoder_type', 'poisson'),
        use_hierarchical=use_hierarchical,  # Use provided setting instead of checkpoint
        N_feat=N_feat if use_hierarchical else None,  # Use provided N_feat if hierarchical
        fix_R_z=True,
        use_inputs=model_config.get('use_inputs', False),
        learn_initial_states=learn_initial_states  # Use the determined value
    )
    
    # Load state dict with strict=False to handle parameter name differences
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Add session info to model
    model.session_name = model_config.get('session_name', 'unknown_session')
    model.bin_width = model_config.get('bin_width', 0.05)
    model.decoder_type = model_config.get('decoder_type', 'poisson')
    
    print(f"\nLoaded model from checkpoint:")
    print(f"- Model type: {'Hierarchical' if use_hierarchical else 'Standard'} RNN")
    if use_hierarchical:
        print(f"- Number of features: {N_feat}")
    print(f"- Latent dimension (M): {model.M}")
    print(f"- Number of positive units (P): {model.P}")
    print(f"- Using external inputs: {model.use_inputs}")
    print(f"- Initial states: {'Learnable' if learn_initial_states else 'Fixed to zeros'}")
    
    return model

if __name__ == '__main__':
    # Model parameters
    model_params = {
        'M': 6,  # latent dimension
        'P': 1,  # number of positive units
        'decoder_type': 'poisson',  # decoder type
        'use_hierarchical': False,  # use hierarchical RNN
        'N_feat': 10,  # number of features for hierarchical RNN
        'learn_initial_states': False,  # whether to learn initial states
    }
    
    # Training parameters
    training_params = {
        'n_epochs': 1000,
        'batch_size': 64,  # Reduced batch size to handle variable lengths more efficiently
        'feature_lr': 8e-4,  # learning rate for feature vectors
        'projection_lr': 1e-4,  # learning rate for projection matrices
        'encoder_lr': 1e-3,  # learning rate for encoder parameters
        'alpha_start': 0.0,  # initial teacher forcing parameter
        'alpha_end': 0.0,   # final teacher forcing parameter
        'use_alpha_scheduling': False,  # toggle for alpha scheduling
        'n_interleave': 1,  # teacher forcing frequency
        'beta_pred': 0.15,  # prediction loss weight
        'beta_enc': 0.15,  # encoder loss weight
        'beta_cons': 0.00001,  # consistency loss weight
        'beta_class': 6.,  # classification loss weight
        'beta_ent': 0.0,  # entropy loss weight
        'use_regularization': True,
        'tau': 0.01,
        'M_reg': 6
    }
    
    # Data parameters
    session_dir = "Data"  # Directory containing the pickled data
    session_name = "CR12B_day4"
    bin_width = 0.05  # Time bin width in seconds
    use_inputs = True  # Whether to use external inputs
    load_existing_model = False # Toggle this to load existing model
    # First load the data
    print(f"\nLoading data from {session_dir}")
    dataset, dataloader = load_and_create_dataset(
        session_dir, 
        session_name,
        bin_width=bin_width,
        use_inputs=use_inputs,
        batch_size=training_params['batch_size'],
        shuffle=True
    )
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of trials: {len(dataset)}")
    print(f"Number of neurons: {dataset.n_neurons}")
    if dataset.use_inputs:
        print(f"Number of input dimensions: {dataset.n_inputs}")
    print(f"Sequence lengths - min: {min(dataset.sequence_lengths)}, max: {max(dataset.sequence_lengths)}, mean: {float(dataset.sequence_lengths.float().mean()):.1f}")
    
    # Create a unique identifier for this model configuration
    model_id = f"{session_name}_M{model_params['M']}_P{model_params['P']}_bin{bin_width:.3f}"
    if model_params['use_hierarchical']:
        model_id += f"_hier_Nfeat{model_params['N_feat']}"
    if not use_inputs:
        model_id += "_noinputs"
    model_id = model_id.replace('.', '_')
    
    if load_existing_model:
        checkpoint_path = f'checkpoints/{model_id}_best_model.pth'
        print(f"\nLoading model from {checkpoint_path}")
        model = load_model(
            checkpoint_path, 
            dataset,
            use_hierarchical=model_params['use_hierarchical'],
            N_feat=model_params['N_feat'],
            learn_initial_states=model_params['learn_initial_states']
        )
    else:
        print("\nCreating new model")
        model = create_new_model(model_params, dataset, session_name, bin_width)
    
    # Then train the model
    train_model(dataloader, model, training_params)









