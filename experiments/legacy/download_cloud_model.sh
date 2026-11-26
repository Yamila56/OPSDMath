#!/usr/bin/env bash
set -euo pipefail

model="${1:?Pass the model destination}"
source_url=https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/master
mkdir -p "$model"

download_shard() (
    name="$1"
    size="$2"
    sha="$3"
    cd "$model"
    if [[ -s "$name" ]] && echo "$sha  $name" | sha256sum -c -; then
        return
    fi
    chunk=268435456
    pids=()
    parts=()
    for ((first=0; first<size; first+=chunk)); do
        last=$((first + chunk - 1))
        ((last < size)) || last=$((size - 1))
        part="$name.part.$first"
        parts+=("$part")
        (
            expected=$((last - first + 1))
            if [[ -f "$part" && "$(stat -c%s "$part")" == "$expected" ]]; then
                exit
            fi
            curl -fsSL --retry 2 --connect-timeout 20 --max-time 900 \
                -r "$first-$last" "$source_url/$name?opsd_range=$first" -o "$part"
            [[ "$(stat -c%s "$part")" == "$expected" ]]
            printf '%s range %s-%s complete\n' "$name" "$first" "$last"
        ) &
        pids+=("$!")
    done
    passed=1
    for pid in "${pids[@]}"; do
        wait "$pid" || passed=0
    done
    [[ "$passed" == 1 ]]
    cat "${parts[@]}" > "$name.verified"
    echo "$sha  $name.verified" | sha256sum -c -
    mv "$name.verified" "$name"
    rm -- "${parts[@]}"
)

download_shard model-00001-of-00002.safetensors 3441185608 169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5 &
first_shard=$!
download_shard model-00002-of-00002.safetensors 622329984 912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9 &
second_shard=$!
wait "$first_shard"
wait "$second_shard"
