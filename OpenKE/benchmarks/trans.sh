#! /bin/bash

pattern="_1"

dir=`ls ./`
for f in $dir
do
    if [[  $f =~ $pattern ]]
    then
        scp -r ./${f}/model/baseline/0 penghao@192.168.5.201:~/dulinfeng/OpenKE/benchmarks/${f}/model/baseline/
    fi
done
