#!/bin/bash
# Train maxrl — see README "Training" section and train.sh for all knobs.
export METHOD=maxrl
exec bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.sh" "$@"
