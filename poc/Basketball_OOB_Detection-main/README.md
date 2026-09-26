# Basketball Out-of-Bounds Detection
Deep learning system to determine which team gets possession after an out-of-bounds call.

## Problem
Out-of-bounds calls often require referees to review video footage at courtside monitors, causing game delays that disrupt momentum and allow players to cool down. This system provides instant automated decisions to eliminate review delays.

## Approach
- **Architecture**: MobileNetV2 + LSTM
- **Transfer Learning**: Leveraged ImageNet pretraining
- **Dataset**: 20 real game clips from Golden State Warriors games challenges (courtesy to the official NBA website)
  **Note:** Video clips are not included in this repository due to copyright.
  
## Results
- **~80% accuracy** across 4 folds (accuracy variating from 70%-90% depending on random initialization)
- Trained on only 15 videos per fold
- Handles different jersey colors and game scenarios

## Technical Highlights
- Smart data augmentation
- Jersey color metadata integration
- Efficient architecture for limited compute (Colab free GPU)

## Dataset
This project uses 20 video clips from Golden State Warriors games sourced from 
the official NBA website. Due to copyright restrictions, the videos are not 
included in this repository.

To reproduce results:
1. Obtain similar basketball game clips
2. Place them in `/content/` directory
3. Name format: `clip X - [Home/Away] team ball.mp4`
4. Run the training pipeline

The model architecture and methodology are fully documented and can be applied 
to any basketball video dataset.

## Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Future Work
- Expand to all NBA teams
- Expand dataset
- Real-time inference optimization
