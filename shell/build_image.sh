#!/bin/bash

set -eu

function usage() {
cat <<EOF
Usage: ./shell/build_image.sh {image_name}
----args----
    image_name  Docker Image Name (Default: private_lora_image)

EOF
}

function validation() {
    # 引数の検証
    if [[ $# -ge 2 ]]; then
        echo 'Too many arguments!'
        usage
        exit 1
    fi

    # Docker環境の検証
    if ! which docker > /dev/null 2>&1; then
        echo 'It is possible that Docker is not installed on the machine.'
        exit 1
    fi
}

validation "$@"

image_name=${1:-'private_lora_image'}

if docker image inspect ${image_name} > /dev/null 2>&1; then
    read -p "The image ${image_name} already exits. Do you want to continue? (y/n): " choice
    case ${choice} in
        y|Y)
            echo "Continuing with the operation..."
            ;;
        n|N)
            echo "Operation aborted."
            exit 1
            ;;
        *)
            echo "Invalid input. Please enter 'y' or 'n'."
            exit 1
            ;;
    esac
fi

docker image build -t ${image_name} .