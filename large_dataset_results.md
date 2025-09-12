# P/NP Oscillator Results with Large Dataset

## Training Progress

- Epoch 0 | Avg Loss: 3.2938 | Phase: 0.50 | Events: 0
- Epoch 5 | Avg Loss: 1.5730 | Phase: 0.50 | Events: 25
- Epoch 10 | Avg Loss: 0.4747 | Phase: 0.50 | Events: 50
- Epoch 15 | Avg Loss: 0.1538 | Phase: 0.50 | Events: 75
- Epoch 20 | Avg Loss: 0.0743 | Phase: 0.50 | Events: 100
- Epoch 25 | Avg Loss: 0.0444 | Phase: 0.50 | Events: 125
- Epoch 30 | Avg Loss: 0.0329 | Phase: 0.50 | Events: 151
- Epoch 35 | Avg Loss: 0.0243 | Phase: 0.50 | Events: 176
- Epoch 40 | Avg Loss: 0.0179 | Phase: 0.50 | Events: 201
- Epoch 45 | Avg Loss: 0.0151 | Phase: 0.50 | Events: 226

## Test Results

Test: A B C D E → 'F' (Expected: 'F')
Test: F G H I J → 'K' (Expected: 'K')
Test: K L M N O → 'P' (Expected: 'P')

## Model Details

- Device: cpu
- Dataset shape: torch.Size([100, 5]), torch.Size([100])
- Total compression events detected: 251
- Total parameters: 103580

## Sample Compression Events

- Step 26: Rank 5 → 0 (Phase 0.44)
- Step 27: Rank 5 → 0 (Phase 0.38)
- Step 28: Rank 5 → 0 (Phase 0.32)
- Step 29: Rank 5 → 0 (Phase 0.26)
- Step 30: Rank 5 → 0 (Phase 0.21)
