import sys
sys.path.append('surgical_ai_system')
print('Testing instrument detector with trained model...')     
try:
    from models.instrument_detection.instrument_tracker import AdvancedInstrumentDetector
    import torch
    
    # Create model
    model = AdvancedInstrumentDetector()
    print('✅ Model created successfully')
    
    # Load trained weights
    state_dict = torch.load('surgical_ai_system/trained_models/instrument_tracker.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    print('✅ Model weights loaded successfully!')
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f'✅ Forward pass works! Output shape: {output.shape}')
    
    print('✅ Instrument detector fixed successfully!')
    
except Exception as e:
    print(f'❌ Failed: {e}')
    import traceback
    traceback.print_exc()