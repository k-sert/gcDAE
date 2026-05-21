import torch
from argparse import Namespace
import sys
import math

# Ensure we can import from your local modules
sys.path.append(".") 

from models.hyp_clip import hae_clip

def run_test():
    print("1. Setting up dummy options...")
    opts = Namespace(
        encoder_version="",  
        encoder_type='none', 
        feature_size=512,
        c=5.0,  
        dataset_type='caltech256_encode',
        transformer_layers=30,
        geometry='spherical',
        hyperbolic=False,
        checkpoint_path=None,
        psp_checkpoint_path=None
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("2. Initializing the model (bypassing heavy IO & weights)...")
    
    # Monkeypatch BOTH encoder setup and weight loading to bypass disk files
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
        # Restore original functionality
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
        print(f"Logits shape (Classes)         : {logits.shape} -> Expected: ({batch_size}, 257)")
        print(f"Spherical Manifold shape       : {feature_dist.shape} -> Expected: ({batch_size}, 512)")
        print(f"Decoded Euclidean shape        : {feature_euc.shape} -> Expected: ({batch_size}, 1, 1024)")
        
        print("\n5. Checking Spherical Geometry Constraints...")
        k = opts.c
        expected_radius = 1.0 / math.sqrt(k)
        actual_norms = feature_dist.norm(dim=-1)
        
        print(f"Expected Sphere Radius         : {expected_radius:.4f}")
        print(f"Actual feature norms           : {actual_norms.cpu().numpy()}")
        
        is_on_manifold = torch.allclose(actual_norms, torch.full_like(actual_norms, expected_radius), atol=1e-4)
        if is_on_manifold:
            print("🚀 SUCCESS: The features are correctly projected onto the sphere's surface!")
        else:
            print("⚠️ WARNING: The features do not match the expected spherical radius.")

        if torch.isnan(feature_dist).any():
            print("❌ ERROR: NaN values detected!")
        
        logit_scale = model.geometry_head.classifier.logit_scale.exp().item()
        print(f"Current Classifier Logit Scale : {logit_scale:.4f}")

    except Exception as e:
        print(f"\n❌ Forward pass crashed!")
        raise e

if __name__ == '__main__':
    run_test()