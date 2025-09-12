# P/NP Oscillating Transformer Results

## Training Progress

- Step 0 | Loss: 1.8697 | Phase: 0.50 | Events: 0
- Step 20 | Loss: 0.0305 | Phase: 0.50 | Events: 10
- Step 40 | Loss: 0.0153 | Phase: 0.50 | Events: 20
- Step 60 | Loss: 0.0080 | Phase: 0.50 | Events: 30
- Step 80 | Loss: 0.0026 | Phase: 0.50 | Events: 40
- Step 100 | Loss: 0.0018 | Phase: 0.50 | Events: 50
- Step 120 | Loss: 0.0022 | Phase: 0.50 | Events: 61
- Step 140 | Loss: 0.0001 | Phase: 0.50 | Events: 71
- Step 160 | Loss: 0.1537 | Phase: 0.50 | Events: 81
- Step 180 | Loss: 0.0028 | Phase: 0.50 | Events: 92

🧪 Test: 'A B C' → 'D' (Expected: 'D')

🔍 COMPRESSION EVENTS DETECTED:
  Step 11: Rank 3 → 0 (Phase 0.35)
  Step 12: Rank 3 → 0 (Phase 0.21)
  Step 13: Rank 3 → 0 (Phase 0.10)
  Step 14: Rank 3 → 0 (Phase 0.02)
  Step 15: Rank 3 → 0 (Phase 0.00)

## Model Details

- Device: cpu
- Dataset shape: torch.Size([5, 3]), torch.Size([5])
- Total compression events detected: 102
- Total parameters: 34375
