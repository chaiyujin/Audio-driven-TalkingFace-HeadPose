#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

function GetFilePath() {
  IFS='|' read -ra ADDR <<< "$1"
  echo "${ADDR[0]}" | xargs;
}

function GetSeqID() {
  IFS='|' read -ra ADDR <<< "$1"
  if [ "${#ADDR[@]}" -gt 1 ]; then
    echo "${ADDR[1]}" | xargs;
  else
    local fname=$(basename $ADDR[0])
    local dname=$(basename $(dirname $ADDR[0]))
    fname="${fname%.*}"
    if [ "$fname" == "audio" ]; then
      local ddname=$(basename $(dirname $(dirname $ADDR[0])))
      echo "${ddname}/${dname}" | xargs
    else
      echo "${dname}/${fname}" | xargs
    fi
  fi
}

function LoadMediaList() {
  local filepath=$(realpath $1);
  shift 1;

  input=$filepath
  while IFS= read -r line
  do
    # trim
    line=$(echo $line | xargs)
    # valid
    if [ -n "$line" ] && [ "${line:0:1}" != "#" ]; then
      if [ "${line:0:1}" != "/" ]; then
        line=$(dirname $filepath)/${line}
      fi
      echo $(GetFilePath "$line")"|"$(GetSeqID "$line")
    fi
  done < "$input"
}

function DRAW_DIVIDER() {
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" ' ' | tr ' ' '-'
}

function RUN_WITH_LOCK_GUARD() {
  local _end=
  local lock_file=
  local tag=
  local debug=
  local cmd=
  for i in "$@"; do
    if [ -z "${_end}" ]; then
      if [ "$i" = "--" ]; then
        _end=true
        continue
      fi
      case $i in
        -l=* | --lock_file=* ) lock_file=${i#*=}  ;;
        -t=* | --tag=*       ) tag=${i#*=}        ;;
        -d   | --debug       ) debug=true         ;;
        *) echo "[LockFileGuard]: Wrong argument ${i}"; exit 1;;
      esac
    else
      cmd="$cmd '$i'"
    fi
  done

  # lock_file must be given
  if [ -z "$lock_file" ]; then
    printf "lock_file is not set!\n"
    exit 1
  fi

  if [ -f "$lock_file" ]; then
    printf "Command '${tag}' is skipped due to existance of lock_file ${lock_file}\n"
    return
  fi

  # to abspath
  mkdir -p "$(dirname $lock_file)"
  lock_file="$(cd $(dirname $lock_file) && pwd)/$(basename $lock_file)"

  # debug message
  if [ -n "$debug" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "LOCK_FILE: $lock_file"
    echo "COMMAND:   $cmd"
    echo "--------------------------------------------------------------------------------"
  fi

  # Run command in subshell and creat lock file if success
  if (eval "$cmd") ; then
    [ -f "$lock_file" ] || touch "$lock_file"
  else
    echo "Failed to run command '${tag}'"
    exit 1
  fi
}