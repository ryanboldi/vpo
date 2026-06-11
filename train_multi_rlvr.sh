#!/bin/bash
# Train multi_rlvr — see README "Training" section and train.sh for all knobs.
export METHOD=multi_rlvr
exec bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.sh" "$@"
