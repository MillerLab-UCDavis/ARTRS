#!/bin/bash
if [$1 != ""]; then
	export MEMCHUNK=$1
fi
source venv/bin/activate