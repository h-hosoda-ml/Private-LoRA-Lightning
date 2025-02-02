#!/bin/bash

set -eu

function usage() {
cat <<EOF
Usage: ./shell/build_image.sh {image_name} {container_name}
----args----
    image_name  Docker Image Name (Default: private_lora_image)
    container_name Docker ContainerName (Default: private_lora_container)
EOF
}

function validation() {
    # 引数の検証
    if [[ $# -ge 3 ]]; then
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

# 引数の取得
image_name=${1:-'private_lora_image'}
container_name=${2:-'private_lora_container'}

# コンテナの立ち上げ
docker container run -dit --rm \
    --name ${container_name} \
    --gpus '"device=0"' \
    ${image_name}

# コンテナ内部へ
docker exec -it ${container_name} /bin/bash
