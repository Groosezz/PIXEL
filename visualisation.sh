export WANDB_API_KEY=51b11767e418e6e1b836ebd2559f3a7c074b70ed


python ./scripts/visualization/visualize_pixel.py \
  --input_str="Cats conserve energy by sleeping more than most animals, especially as they grow older." \
  --model_name_or_path="../cache/models/pixel-base" \
  --span_mask \
  --mask_ratio=0.25 \
  --max_seq_length=256

