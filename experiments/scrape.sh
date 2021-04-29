#!/usr/bin/env zsh

rm ./experiments/${1}/experiment1-${2}-iters.log
rm ./experiments/${1}/experiment1-${2}-dist.log
rm ./experiments/${1}/experiment1-${2}-obj.log

for filename in ./experiments/${1}/experiment1*${2}*.log; do
    echo $(grep 'iters' ${filename} | cut -d = -f 2) >> ./experiments/${1}/experiment1-${2}-iters.log
    echo $(grep 'dist' ${filename} | cut -d = -f 2)  >> ./experiments/${1}/experiment1-${2}-dist.log
    echo $(grep 'obj' ${filename} | cut -d = -f 2)   >> ./experiments/${1}/experiment1-${2}-obj.log
done