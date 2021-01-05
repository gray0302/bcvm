#!/bin/sh
source ~/.bashrc

stime=$1
etime=$2
HDFS_IN_PATH=/stage/outface/SNG/g_sng_qqmusic_develop/g_sng_qqmusic_develop/
for((i=$stime;i<=$etime;i++));
do
{
  dir="`pwd`/tfrecords/rt_mt"
  rm -r $dir/${i}
  mkdir -p $dir/${i}

  hadoop fs -Dfs.defaultFS=hdfs://ss-sng-dc-v2/ -Dhadoop.job.ugi=tdw_graywang:gray32_gogo -get $HDFS_IN_PATH/timmili/gray_temp/_anchor_mt/${i} $dir
} &
done
wait
echo "finish"
