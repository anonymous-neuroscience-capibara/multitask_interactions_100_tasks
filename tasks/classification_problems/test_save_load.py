#!/usr/bin/env python3
"""
Test script to verify that the saving and loading functions work correctly.
"""

import torch
import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_utils_mnist import (
    PLRNN, LSTMClassifier, GRUClassifier, 
    save_model_with_config, load_model_from_checkpoint
)


def test_plrnn_save_load():
    """Test saving and loading a PLRNN model."""
    print("Testing PLRNN save/load...")
    
    # Create a PLRNN model
    model = PLRNN(
        M=100,
        L=25,
        N=10,
        input_dim=100,
        readout_dim=100,
        nonlinearity_type='tanh'
    )
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    try:
        # Save the model
        save_model_with_config(
            model=model,
            save_path=model_path,
            test_metric=0.95
        )
        
        # Load the model
        loaded_model, checkpoint_info = load_model_from_checkpoint(model_path)
        
        # Verify the models are equivalent
        for name, param in model.named_parameters():
            loaded_param = loaded_model.state_dict()[name]
            if not torch.allclose(param, loaded_param):
                print(f"❌ Parameter {name} differs!")
                return False
        
        print("✅ PLRNN save/load test passed!")
        return True
        
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_lstm_save_load():
    """Test saving and loading an LSTM model."""
    print("Testing LSTM save/load...")
    
    # Create an LSTM model
    model = LSTMClassifier(
        M=100,
        L=0,  # Not used for LSTM
        N=10,
        input_dim=100,
        readout_dim=100
    )
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    try:
        # Save the model
        save_model_with_config(
            model=model,
            save_path=model_path,
            test_metric=0.92
        )
        
        # Load the model
        loaded_model, checkpoint_info = load_model_from_checkpoint(model_path)
        
        # Verify the models are equivalent
        for name, param in model.named_parameters():
            loaded_param = loaded_model.state_dict()[name]
            if not torch.allclose(param, loaded_param):
                print(f"❌ Parameter {name} differs!")
                return False
        
        print("✅ LSTM save/load test passed!")
        return True
        
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_gru_save_load():
    """Test saving and loading a GRU model."""
    print("Testing GRU save/load...")
    
    # Create a GRU model
    model = GRUClassifier(
        M=100,
        L=0,  # Not used for GRU
        N=10,
        input_dim=100,
        readout_dim=100
    )
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    try:
        # Save the model
        save_model_with_config(
            model=model,
            save_path=model_path,
            test_metric=0.90
        )
        
        # Load the model
        loaded_model, checkpoint_info = load_model_from_checkpoint(model_path)
        
        # Verify the models are equivalent
        for name, param in model.named_parameters():
            loaded_param = loaded_model.state_dict()[name]
            if not torch.allclose(param, loaded_param):
                print(f"❌ Parameter {name} differs!")
                return False
        
        print("✅ GRU save/load test passed!")
        return True
        
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)


def main():
    """Run all tests."""
    print("Testing save/load functionality...")
    print("=" * 50)
    
    tests = [
        test_plrnn_save_load,
        test_lstm_save_load,
        test_gru_save_load
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")


if __name__ == "__main__":
    main() 