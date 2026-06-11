#!/bin/bash
# Train vpo — see README "Training" section and train.sh for all knobs.
export METHOD=vpo
exec bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.sh" "$@"
