#!/bin/bash
# Train grpo — see README "Training" section and train.sh for all knobs.
export METHOD=grpo
exec bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.sh" "$@"
