#!/bin/bash
D=$(date +"%Y%m%d%H%M")
D=${D}
FNH1="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h1_iperf.txt"
FNH2="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h2_iperf.txt"
FNH3="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h3_iperf.txt"
FNH4="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h4_iperf.txt"
FNH5="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h5_iperf.txt"
FNH6="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h6_iperf.txt"
FNH7="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h7_iperf.txt"
FNH8="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_h8_iperf.txt"
OUTPUT="/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/"$D"/"$D"_output.log"

mkdir /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/output/$D
# 设置错误输出和标准输出同时重定向到文件
exec > >(tee $OUTPUT) 2>&1

/home/sinet/P4/mininet/util/m h1 iperf -s -p 5001 -t 110 -e -i 1 -f m > $FNH1 &
/home/sinet/P4/mininet/util/m h3 iperf -s -p 5001 -t 110 -e -i 1 -f m > $FNH3 &
/home/sinet/P4/mininet/util/m h4 iperf -s -p 5001 -t 110 -e -i 1 -f m > $FNH4 &
/home/sinet/P4/mininet/util/m h5 iperf -s -p 5001 -t 110 -e -i 1 -f m > $FNH5 &
sleep 2

/home/sinet/P4/mininet/util/m h2 iperf -c 10.0.1.2 -p 5001 -t 100 -i 1 -e -f m > $FNH2 &
/home/sinet/P4/mininet/util/m h7 iperf -c 10.0.3.2 -p 5001 -t 100 -i 1 -e -f m > $FNH7 &
/home/sinet/P4/mininet/util/m h6 iperf -c 10.0.4.2 -p 5001 -t 100 -i 1 -e -f m > $FNH6 &
/home/sinet/P4/mininet/util/m h8 iperf -c 10.0.5.2 -p 5001 -t 100 -i 1 -e -f m > $FNH8 
