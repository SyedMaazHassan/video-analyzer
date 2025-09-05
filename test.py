import sys
sys.path.append('surgical_ai_system')
from models.phase_detection.phase_detector import AdvancedPhaseDetector
model = AdvancedPhaseDetector()
print('Current model expects these keys:')
for i, key in enumerate(list(model.state_dict().keys())[:10]):
    print(f'{i+1}. {key}: {model.state_dict()[key].shape}')