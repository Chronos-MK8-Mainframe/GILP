
import sys
import os
import torch

# Add root to path
sys.path.insert(0, os.getcwd())

from epsilon.system import Epsilon
from epsilon.config import PsychologyConfig, ExpressionConfig

def test_v2_instantiation():
    print("Initializing Epsilon v2...")
    
    # Instantiate
    system = Epsilon()
    
    # Check components
    print("Checking Logic Engine...")
    assert system.logic is not None
    assert system.logic.manifold is not None
    print("✓ Logic Engine Ready")
    
    print("Checking Psychology Engine...")
    assert system.psychology is not None
    assert isinstance(system.psych_config, PsychologyConfig)
    assert system.psychology.config.shell_radius == 0.3
    print("✓ Psychology Engine Ready")
    
    print("Checking Expression Engine...")
    assert system.expression is not None
    assert isinstance(system.expr_config, ExpressionConfig)
    assert system.expression.config.embedding_dim == 128
    print("✓ Expression Engine Ready")
    
    print("Checking Tiny Decoder...")
    assert system.decoder is not None
    print("✓ Tiny Decoder Ready")
    
    print("\nEpsilon v2 Architecture Verified Successfully!")

if __name__ == "__main__":
    test_v2_instantiation()
