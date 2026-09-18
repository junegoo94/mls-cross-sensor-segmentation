#!/bin/bash
# Publish the weights-only checkpoints as GitHub release v1.0-weights (needs an authenticated gh CLI).
set -e
cd /home/jmgoo/projects/os_seg/exp/release_weights
gh release create v1.0-weights *.pth WEIGHTS.md -R junegoo94/mls-cross-sensor-segmentation \
  --title "Trained weights (paper v1)" --notes-file RELEASE_NOTES.md
