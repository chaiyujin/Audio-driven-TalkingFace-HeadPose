#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

function NOHUP_RUN() {
  local _end=
  local device=
  local includes=()
  local log=
  local cmd=

  local tag="[NOHUP_RUN (${device})]"
  for i in "$@"; do
    if [ -z "${_end}" ]; then
      if [ "$i" = "--" ]; then
        _end=true
        continue
      fi
      case $i in
             --device=* ) device=${i#*=}; tag="[NOHUP_RUN (${device})]" ;;
        -i=*|--include=*) includes+=("'${i#*=}'") ;;
        -l=*|--log=*    ) log=${i#*=} ;;
        *) echo "[NOHUP_RUN]: Wrong argument ${i}"; exit 1;;
      esac
    else
      if [[ $i =~ ^[\+]*debug=.* ]]; then
        echo "${tag}: Ignore debug option, always set 'false'!"
        continue
      elif [[ $i =~ ^[\+]*utils\.matplotlib_using=.* ]]; then
        echo "${tag}: Ignore matplotlib_using option, always set 'agg'!"
        continue
      else
        cmd="$cmd '$i'"
      fi
    fi
  done

  # (Optional) log
  if [ -z "$log" ]; then
    # * Default log file
    log=".snaps/nohup-${device}.log"
  fi
  # create log file
  mkdir -p "$(dirname $log)"
  touch "$log"
  # to abspath
  log="$(cd $(dirname $log) && pwd)/$(basename $log)"

  # convert includes into source *
  local pre_cmd=
  local fpath=
  if [ ${#includes[@]} -gt 0 ]; then
    for inc in "${includes[@]}"; do
      fpath=${inc:1:-1}  # remove '' which added to surround before
      pre_cmd+="source $fpath; "
    done
  fi

  # * Process cmd and check some args of command
  # append matplotlib_using='agg', debug=false
  local real_cmd="$pre_cmd $cmd 'matplotlib_using=agg' 'debug=false'"

  # * Tell the information log position
  echo "${tag}: Redirect stdout to ${log}"

  # * Log command
  echo "================================================================================" >> $log
  echo "$real_cmd" >> $log
  echo "================================================================================" >> $log

  # * nohup
  COLUMNS=160 CUDA_VISIBLE_DEVICES=$device \
    nohup bash -c "$real_cmd" > $log 2>&1 & \
  disown;
}
