import torch
from argparse import Namespace
import sys
import math

# Ensure we can import from your local modules
sys.path.append(".") 

from models.hyp_clip import hae_clip

def run_test():
    print("1. Setting up options for HYPERBOLIC geometry...")
    opts = Namespace(
        encoder_version="",  
        encoder_type='none', 
        feature_size=512,
        c=5.0,  
        dataset_type='caltech256_encode',
        transformer_layers=30,
        geometry='hyperbolic', # Changed to hyperbolic
        hyperbolic=True,       # Switched to True
        checkpoint_path=None,
        psp_checkpoint_path=None
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("2. Initializing the model...")
    # Monkeypatch to bypass heavy IO
    original_set_encoder = hae_clip.set_encoder
    original_load_weights = hae_clip.load_weights
    
    hae_clip.set_encoder = lambda self, version: torch.nn.Identity()
    hae_clip.load_weights = lambda self: print("-> Weights initialization bypassed for testing")
    
    try:
        model = hae_clip(opts).to(device)
        model.eval()
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    finally:
        hae_clip.set_encoder = original_set_encoder
        hae_clip.set_weights = original_load_weights

    print("\n3. Creating dummy input...")
    batch_size = 4
    dummy_code = torch.randn(batch_size, 1, 1024).to(device)

    print("\n4. Running Forward Pass...")
    try:
        with torch.no_grad():
            logits, feature_dist, ocodes, feature_euc = model(
                dummy_code, 
                batch_size=batch_size, 
                input_code=True
            )
        
        print("\n✅ FORWARD PASS SUCCESSFUL!")
        print("-" * 40)
        print(f"Logits shape                 : {logits.shape}")
        print(f"Hyperbolic Manifold shape    : {feature_dist.shape}")
        print(f"Decoded Euclidean shape      : {feature_euc.shape}")
        
        print("\n5. Checking Hyperbolic Geometry Constraints...")
        k = opts.c
        max_radius = 1.0 / math.sqrt(k)
        actual_norms = feature_dist.norm(dim=-1)
        
        print(f"Poincaré Ball Boundary Radius: {max_radius:.4f}")
        print(f"Actual feature norms         : {actual_norms.cpu().numpy()}")
        
        # Hyperbolic constraint: norms MUST be strictly less than the boundary radius
        is_inside_ball = torch.all(actual_norms < max_radius)
        if is_inside_ball:
            print("🚀 SUCCESS: The features are safely contained inside the Poincaré ball!")
        else:
            print("⚠️ WARNING: Features escaped the Poincaré ball! (Norms exceeded max boundary)")

    except Exception as e:
        print(f"\n❌ Forward pass crashed!")
        raise e

if __name__ == '__main__':
    run_test()