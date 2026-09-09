#!/bin/bash

# Shared formatting for CLI overrides passed through submit.sh.
print_bed_cli_overrides() {
    local empty_description="${1:-none - using YAML defaults}"

    if [ -n "${BED_CLI_OVERRIDES:-}" ]; then
        echo "CLI Overrides (non-default values):"
        echo "------------------------------------"
        set -- $BED_CLI_OVERRIDES
        while [ $# -gt 0 ]; do
            if [[ "$1" == --* ]]; then
                if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                    echo "  $1 $2"
                    shift 2
                else
                    echo "  $1"
                    shift 1
                fi
            else
                shift 1
            fi
        done
    else
        echo "CLI Overrides: ($empty_description)"
    fi
}
