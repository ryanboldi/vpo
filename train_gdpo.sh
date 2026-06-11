#!/bin/bash
# Train gdpo — see README "Training" section and train.sh for all knobs.
export METHOD=gdpo
exec bash "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.sh" "$@"
