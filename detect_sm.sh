#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1
then
  echo "nvidia-smi not found. Is the NVIDIA driver installed?"
  exit 1
fi

echo "Detecting NVIDIA GPUs..."
echo

sms=()
majors=()

while IFS=',' read -r idx name cap
do
  name=$(echo "$name" | xargs)
  cap=$(echo "$cap" | xargs)

  echo "GPU $idx: $name"

  if [[ "$cap" == "N/A" ]]
  then
    echo "  Compute Capability: unavailable"
    echo "  Suggested action: use CUDA-based detector"
    echo
    continue
  fi

  sm="${cap/./}"
  major=$(( sm / 10 ))

  sms+=("$sm")
  majors+=("$major")

  echo "  Compute Capability: $cap"
  echo "  Individual build: make SM=$sm"
  echo
done < <(
  nvidia-smi \
    --query-gpu=index,name,compute_cap \
    --format=csv,noheader
)

if (( ${#sms[@]} > 1 ))
then
  IFS=$'\n' sorted_sms=($(sort -n <<<"${sms[*]}"))
  IFS=$'\n' sorted_majors=($(sort -u <<<"${majors[*]}"))
  unset IFS

  common_sm="${sorted_sms[0]}"

  echo "Multiple GPUs detected."
  echo "Highest common supported SM: $common_sm"
  echo "Recommended portable build:"
  echo "  make SM=$common_sm"
  echo

  if (( ${#sorted_majors[@]} > 1 ))
  then
    echo "⚠️  Architecture diversity warning:"
    echo "   Detected GPUs span multiple major SM versions: ${sorted_majors[*]}"
    echo "   This may limit performance optimizations and instruction selection."
    echo "   Consider per-architecture builds if maximum performance is required."
    echo
  fi
elif (( ${#sms[@]} == 1 ))
then
  echo "Single GPU detected."
  echo "Recommended build:"
  echo "  make SM=${sms[0]}"
fi

# end of script
