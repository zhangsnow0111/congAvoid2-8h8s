#!/bin/bash
# 需要先sudo su 切换到root用户才能运行此sh

# 如果正在重放，就先杀掉tcpreplay进程
num_tcpreplay=$(ps -au | grep tcpreplay|awk 'END{print NR}')
if [ $num_tcpreplay -gt 1 ]; then
  echo "kill replay1!!!"
  ps -ef | grep tcpreplay | grep -v grep | cut -c 9-15 | xargs kill -9
  echo "kill replay2!!!"
	sleep 1s
fi
# 再杀掉screen
num_screen=$(screen -ls|awk 'NR>=2&&NR<=40{print $1}'|awk 'END{print NR}')
# echo $num_screen
if [ $num_screen -gt 1 ]; then
  echo "kill screens1!!!"
  screen -ls|awk 'NR>=2&&NR<=40{print $1}'|awk '{print "screen -S "$1" -X quit"}'|sh  # 传入命令进screen去关
  echo "kill screens2!!!"
	sleep 1s
fi

# screen -dmS h2 bash -c "tcpreplay -i s2-eth1 $2 /home/sinet/lt/dataset/20220413$1/h2.pcap\n"
# 这种方法的问题是上面关闭screen的代码不起作用。
# $1 : 要重放的文件， $2： -t x 以x倍速重放。
#screen -dmS h1 bash -c "tcpreplay -i s1-eth3 ./dataset/202204131400/h1.pcap";
#screen -dmS h2 bash -c "tcpreplay -i s2-eth1 ./dataset/202204131400/h2.pcap"
#screen -dmS h3 bash -c "tcpreplay -i s3-eth2 ./dataset/202204131400/h3.pcap"
#screen -dmS h4 bash -c "tcpreplay -i s4-eth2 ./dataset/202204131400/h4.pcap"
#screen -dmS h5 bash -c "tcpreplay -i s5-eth4 ./dataset/202204131400/h5.pcap"
#screen -dmS h6 bash -c "tcpreplay -i s6-eth5 ./dataset/202204131400/h6.pcap"
#screen -dmS h7 bash -c "tcpreplay -i s7-eth6 ./dataset/202204131400/h7.pcap"
#screen -dmS h8 bash -c "tcpreplay -i s8-eth6 ./dataset/202204131400/h8.pcap"

# 下面这种方法需要先 `screen -dmS h8` 创建screen，然后向其中传入命令。
# screen -dmS h1
# screen -dmS h2
# screen -dmS h3
# screen -dmS h4
# screen -dmS h5
# screen -dmS h6
# screen -dmS h7
# screen -dmS h8
sleep 1s
/home/sinet/P4/mininet/util/m s1 tcpreplay -i s1-eth3 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h1.pcap
/home/sinet/P4/mininet/util/m h2 tcpreplay -i s2-eth1 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h2.pcap
/home/sinet/P4/mininet/util/m h3 tcpreplay -i s3-eth2 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h3.pcap
/home/sinet/P4/mininet/util/m h4 tcpreplay -i s4-eth2 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h4.pcap
/home/sinet/P4/mininet/util/m h5 tcpreplay -i s5-eth4 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h5.pcap
/home/sinet/P4/mininet/util/m h6 tcpreplay -i s6-eth5 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h6.pcap
/home/sinet/P4/mininet/util/m h7 tcpreplay -i s7-eth6 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h7.pcap
/home/sinet/P4/mininet/util/m h8 tcpreplay -i s8-eth6 /home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h8.pcap
#/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h1.pcap
#/home/sinet/P4/tutorials/exercises/congAvoid2-8h8s/dataset/202204131400/h1.pcap
