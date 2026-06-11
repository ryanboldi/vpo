#!/bin/bash
# Train goal_cond — see README "Training" section and train.sh for all knobs.
export METHOD=goal_cond
exec bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.sh" "$@"
