#!/usr/bin/env python
# coding=utf-8
import os
import sys
import time
import yiqun as yq
import grpc
import argparse
import copy
import datetime
import subprocess 


# sys.path.append("/usr/local/lib/python3.8/dist-packages")
'''
我将162系统自带的python的google.rpc直接拷到pytorch虚拟环境下，
再将虚拟机里面的p4模块拷到pytorch虚拟环境下，
再安装一个google.protobuf模块，就可以直接运行而不需要再添加上面的额外环境了。 在`~/anaconda3/envs/pytorch/lib/python3.7/site-packages`里
'''
from p4.v1 import p4runtime_pb2
from p4.v1 import p4runtime_pb2_grpc
from predict.load_adj import load_adj 

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../../utils/'))
import p4runtime_lib.bmv2
import p4runtime_lib.helper

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn 
from torch import optim

TIME_STEPS = 10
NUMBER_OF_SWITCH = 8
host_IPs = ['10.0.1.0', '10.0.2.0', '10.0.3.0', '10.0.4.0',
            '10.0.5.0', '10.0.6.0', '10.0.7.0', '10.0.8.0', ]
# 只记录了终端的mac地址，因为P4交换机流表中与终端互联时mac地址也是必需点，交换机之间点互联mac地址无关紧要
host_macs = ['08:00:00:00:01:01', '08:00:00:00:02:02',
             '08:00:00:00:03:03', '08:00:00:00:04:04',
             '08:00:00:00:05:05', '08:00:00:00:06:06',
             '08:00:00:00:07:07', '08:00:00:00:08:08', ]
# log min max 归一化的参数
normalize_min_value = 0
normalize_max_value = 15  # e的15次方大概是3,270,000

# ports[i][j] 表示 交换机si到sj的出端口，对角线是到对应终端的
ports = [
    [3, 0, 1, 0, 0, 0, 0, 4],
    [0, 1, 0, 3, 4, 0, 5, 0],
    [4, 0, 2, 1, 0, 6, 0, 5],
    [0, 5, 3, 2, 4, 0, 6, 0],
    [0, 1, 0, 2, 4, 3, 6, 0],
    [0, 0, 2, 0, 1, 5, 0, 4],
    [0, 2, 0, 1, 3, 0, 6, 4],
    [4, 0, 3, 0, 0, 2, 1, 6]
]
# next_switch_table[i][j] 表示 交换机si到hj的下一跳交换机
next_switch_table = [   # 交换机编号0~7 而不用1~8，简化后面的操作
    [0, 7, 2, 2, 2, 7, 7, 7],  # next_switch_table [0][1] 从2改成7做测试
    [3, 1, 3, 3, 4, 4, 6, 6],   # 同时 next_switch_table[i][i] = i以供ports使用
    [0, 3, 2, 3, 5, 5, 3, 7],
    [2, 1, 2, 3, 4, 4, 6, 6],
    [5, 1, 5, 3, 4, 5, 6, 6],
    [2, 4, 2, 2, 4, 5, 7, 7],
    [7, 1, 3, 3, 4, 7, 6, 7],
    [0, 6, 2, 6, 6, 5, 6, 7]]



# 原始流表，保存以防止改忘了
# next_switch_table = [   #
#     [-1, 2, 2, 2, 2, 7, 7, 7],
#     [3, -1, 3, 3, 4, 4, 6, 6],
#     [0, 3, -1, 3, 5, 5, 3, 7],
#     [2, 1, 2, -1, 4, 4, 6, 6],
#     [5, 1, 5, 3, -1, 5, 6, 6],
#     [2, 4, 2, 2, 4, -1, 7, 7],
#     [7, 1, 3, 3, 4, 7, -1, 7],
#     [0, 6, 2, 6, 6, 5, 6, -1]]

# 必须在此主控控制器上安装流表，不能通过json文件安装。不然后续修改流表时会报grpc的错
# 可能的原因：安装流表的控制器必须和修改流表点控制器是同一个
def installRT(p4info_helper, switches, route_table):
    for sw_idx in range(8):
        # 首先写入 p4中的drop的流表
        table_entry_drop = p4info_helper.buildTableEntry(
            table_name="MyIngress.ipv4_lpm",
            default_action=True,
            action_name="MyIngress.drop",
            action_params={})
        switches[sw_idx].WriteTableEntry(table_entry_drop)
        # 再写入其他到终端的流表
        for host_idx in range(8):
            table_entry = p4info_helper.buildTableEntry(
                table_name="MyIngress.ipv4_lpm",
                match_fields={
                    "hdr.ipv4.dstAddr": [host_IPs[host_idx], 24]
                },
                action_name="MyIngress.ipv4_forward",
                action_params={
                    "dstAddr": host_macs[host_idx],     # 只有交换机直连的机器需要把mac地址写正确，交换机与交换机之间互联时mac地址无关。所以MAC地址全写主机点地址就行
                    "port": ports[sw_idx][route_table[sw_idx][host_idx]]
                })  # RECONCILE_AND_COMMIT
            # 此时安装所有的初始流表
            switches[sw_idx].WriteTableEntry(table_entry)

# 根据传入之前的流表和改之后的流表对应修改
def modifyRT(p4info_helper, switches, route_table_before, route_table_after):
    for sw_idx in range(len(route_table_before)):
        for host_idx in range(len(route_table_before[0])):
            if route_table_before[sw_idx][host_idx] != route_table_after[sw_idx][host_idx] :
                table_entry = p4info_helper.buildTableEntry(
                    table_name="MyIngress.ipv4_lpm",
                    match_fields={
                        "hdr.ipv4.dstAddr": [host_IPs[host_idx], 24]
                    },
                    action_name="MyIngress.ipv4_forward",
                    action_params={
                        "dstAddr": host_macs[host_idx],
                        "port": ports[sw_idx][route_table_after[sw_idx][host_idx]]
                    })  # RECONCILE_AND_COMMIT
                # 修改流表时必须用 MODIFY
                switches[sw_idx].ModifyTableEntry(table_entry)


# client_stub 读取这一个存根对象里点寄存器值
def read_counter(client_stub, device_id, counter_id, index = None):
    request = p4runtime_pb2.ReadRequest()
    request.device_id = device_id
    entity = request.entities.add()
    counter_entry = entity.counter_entry
    if counter_id is not None:
        counter_entry.counter_id = counter_id
    else:
        counter_entry.counter_id = 0
    if index is not None:
        counter_entry.index.index = index
    for response in client_stub.Read(request):
        yield response

# 读取8个交换机里点counter 值
def get_counter_value(switchs, client_stubs, p4info_helper, counter_name):
    counter_value = [[0 for _ in range(NUMBER_OF_SWITCH)] for _ in range(NUMBER_OF_SWITCH)]
    for i in range(NUMBER_OF_SWITCH):
        j = 0
        for response in read_counter(client_stubs[i], switchs[i].device_id,
                                     p4info_helper.get_counters_id(counter_name)):
            for entity in response.entities:  # 它只有一个entity
                counter = entity.counter_entry
                counter_value[i][j] = counter.data.byte_count
                j += 1
                # print("%s %s: %d packets (%d bytes)" % (switchs[i].name, counter_name, counter.data.packet_count, counter.data.byte_count))
    return counter_value

def normalize(mx):
    # 该归一化方法是 每个数 除以 所在行的行和 [[1,1], [1,3]] -> [[0.5,0.5],[0.25,0.75]]
    rowsum = mx.sum(1).astype(np.float64)  
    r_inv = np.power(rowsum, -1).flatten()  # 尽量不要让数据有0，不然报运行时警告
    r_inv[np.isinf(r_inv)] = 0. # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# 给蚁群算法实验用的归一化方法
# 用此方法就必须要保证data的最小值为1 
def log_min_max_normalize3(data):
    data = np.log(data) # np.log以e为底数
    # min-max 归一化 normalize_min_value = 0， normalize_max_value = 15
    data = (data - normalize_min_value) / (normalize_max_value - normalize_min_value)
    return data

def denormalize(data_tensor):
    data = data_tensor.numpy()  # 直接 .numpy创建其实是共享内存的。
    data = data * (normalize_max_value - normalize_min_value) + normalize_min_value
    data = np.exp(data)
    return data.tolist()

# 打印矩阵的帮助函数
def print_matrix(matrix_name, matrix):
    print("打印矩阵：", matrix_name)
    for item in matrix:
        print(item)

def queue_rate_set(rate):
    for i in range(8):
        p = subprocess.Popen('simple_switch_CLI --thrift-port 909%d'%i,shell=True,stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                universal_newlines=True) 
        
        p.stdin.write(f'set_queue_rate {rate}\n')
        
        # print(f'修改交换机{i}的queue_rate')
        # 读取命令行界面的输出
        # output = p.communicate()[0]
        # # 打印输出
        # print(output)
        p.communicate()
    # p.stdin.close()


def main():
    parser = argparse.ArgumentParser(description="use congestion avoidance or not")
    parser.add_argument("-c", "--ca",action="store_true", help="open avoidance or not") # 默认是 not
    parser.add_argument("-l", "--learn",action="store_true", help="open learning or not") # 默认是not
    parser_args = parser.parse_args()

    count = 1
    cur_route_table = next_switch_table  # 当前流表
    # s2h_10steps 保存前10次的s2h流量矩阵，单位是Byte/s
    s2h_10steps = [] # s2h_10steps[-1] = counter_value - last_counter_value
    # 交换机中点计数器的内容都是以Byte为单位的
    last_counter_value = [[0 for _ in range(NUMBER_OF_SWITCH)] for _ in range(NUMBER_OF_SWITCH)]
    p4info_helper = p4runtime_lib.helper.P4InfoHelper("./build/basic.p4.p4info.txt")
    counter_name = "MyIngress.pkt_counter"
    switches = []
    # 得到8个交换机对象
    for i in range(NUMBER_OF_SWITCH):
        switch = p4runtime_lib.bmv2.Bmv2SwitchConnection(
            name='s%d' % (i+1),
            address='127.0.0.1:5005%d' % (i+1), # 192.168.199.162
            device_id=i)
        # proto_dump_file='logs/s1-p4runtime-requests.txt' 这个参数就不写了，不然每次修改流表就多一条记录，后期就太长了
        switch.MasterArbitrationUpdate()  # 向交换机发送主控握手请求,设置当前控制平面为主控平面。
        # 设置接管P4流水线以及所有转发条目
        switch.SetForwardingPipelineConfig(p4info=p4info_helper.p4info, bmv2_json_file_path="./build/basic.json")
        switches.append(switch)
    # 得到8个连接交换机的存根
    client_stubs = []
    for i in range(NUMBER_OF_SWITCH):
        channel = grpc.insecure_channel(switches[i].address)
        client_stubs.append(p4runtime_pb2_grpc.P4RuntimeStub(channel))  # P4 交换机对象

    # 安装初始流表
    installRT(p4info_helper, switches, next_switch_table)
    time.sleep(3)

    # 设置队列处理速度，要不然队列跑不上来，通过实验觉得队列处理速度设置为600比较合适（本来是500，但是不想太深）
    # queue_rate_set(500)

    if parser_args.ca:
    # 载入深度学习模块
        print("拥塞避免开启")
        net = torch.load("predict/TCN10509.pt")   # 使用临时的模型训练试试 
        if parser_args.learn:
            print("开启在线学习...")
            net.train() # 开启训练模式
        else:
            net.eval()
        # 把170上训练好的模块发送到162上，下面的语句在170上执行
        # scp /home/sinet/lt/predict/TCN1.py sinet@192.168.199.162:/home/sinet/p4/tutorials/exercises/congAvoid2-8h8s/predict/
        loss_func = nn.MSELoss()    # 均方误差 是预测值与真实值之差的平方和的平均值
        lr = 0.00001
        weight_decay = 5e-4
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.Adam(params=[param for model in net.tcn_models for param in model.parameters()], lr=lr, weight_decay=weight_decay)
        
        best_rt_list, best_fit_list, best_pro_list = [], [], []
        model_file = "predict/TCN10509_learn2.pt"
        last_predict_output = None
    
    # 新建进程去放流量
    print("开始重放流量...（如果之前重放未结束就会自动关闭重开）")
    flow_inject = "bash traffic_inject.sh"    # 在0045后加上-t 参数可以加快重放（尽可能快）
    os.system(flow_inject)
    time.sleep(1)
    cnt = 20  #只运行了100s，获取了100s的数据
    while cnt: 
        start_time = time.perf_counter()  # 计算程序用时的变量，以微秒为单位
        # print("switches: %s, client_stubs: %s, p4info_helper: %s, counter_name: %s", switches, client_stubs, p4info_helper, counter_name)

        # 查询8个交换机的计数器
        counter_value = get_counter_value(switches, client_stubs, p4info_helper, counter_name)
        # print(counter_value)

        # 与上一次计数器的值相减得到s2h的流量矩阵
        s2h_10steps.append([[i - j for i, j in zip(row_cur, row_last)] for row_cur, row_last in zip(counter_value, last_counter_value)])
        # print(s2h_10steps)
        last_counter_value = counter_value  # 当前counter值赋给last_counter

        s2h = s2h_10steps[-1]
        print("***************** 第%d次 ****************" % count)
        print_matrix("流量矩阵s2h", s2h)  # 打印s2h矩阵, 最后一位才是最新的s2h流量矩阵
        print("改流表之前MLU = ", yq.get_MLU(s2h, cur_route_table))
        if parser_args.ca and len(s2h_10steps) >= TIME_STEPS:
            data_arr = np.array(s2h_10steps).reshape(-1 ,len(s2h_10steps[0])) # 改变形状以使能用normalize 进行归一化
            data_arr = log_min_max_normalize3(data_arr).reshape(10 ,len(s2h_10steps[0]), -1)
            data_tensor = torch.tensor(data_arr).unsqueeze(0).float()    # 在最前面增加一维  data_tensor 的寸尺 1*10*8*8
            # print(data_tensor.shape)
            
            if parser_args.learn:   # 开启了在线学习
                optimizer.zero_grad()
                if last_predict_output is not None:     
                    label = data_tensor[0][-1]
                    loss = loss_func(label, last_predict_output)
                    loss.backward()
                    optimizer.step()
                
            # predict_output = net(data_tensor, 0)[0]  # 得到预测结果  1*8*8 
            # LSTM_GCN需要下面这个
            predict_output = net(data_tensor, load_adj())[0]  # 得到预测结果  1*8*8
            # print("predict_output:\n")
            # print(predict_output)
            # ("predict_output size: {} \n".format(predict_output.shape))

            last_predict_output = predict_output.clone()    # 保存此轮的预测结果供下一轮训练更新模型
            # print(predict_output)
            # 关于clone()和 detach()，请看 https://blog.csdn.net/winycg/article/details/100813519
            s2h_predicted = denormalize(predict_output.clone().detach())# 去归一化，传进去克隆出的（新开辟了内存，因为需要detach）
            # print("s2h_predicted\n")
            # print(s2h_predicted)
            # print("s2h_predicted size: {} \n".format(s2h_predicted.shape))

            # print("s2h_predicted: %s" % type(s2h_predicted))

            best_rt, best_fit, best_pro = yq.ant_colony_optimization(s2h_predicted, cur_route_table, copy.deepcopy(yq.pheromone), copy.deepcopy(yq.probabilities))
            
            
            # 必须要传入克隆的pheromone和probabilities，不然就进去把这两个变量给改了，而每一个时隙都需要使用原始的他俩
            best_rt_list.append(copy.deepcopy(best_rt)) # 对于多维list，必须用深拷贝
            best_fit_list.append(best_fit)
            best_pro_list.append(copy.deepcopy(best_pro))
            print_matrix("计算出的路由表", best_rt)
            modifyRT(p4info_helper, switches, route_table_before=cur_route_table, route_table_after=best_rt)
            cur_route_table = best_rt
            print("改流表之后MLU = ", yq.get_MLU(s2h, cur_route_table))
            s2h_10steps.pop(0)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print('第%d次，耗时： %f ms \n' % (count, run_time * 1000))
        count += 1
        time.sleep(1-run_time)
        cnt -= 1
    stop_inject = "bash stop_inject.sh"    
    os.system(stop_inject)
    cur_date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name ="TCN10509"
    directory = 'output/%s_%s_wo' % (cur_date_time, model_name)
    
    if parser_args.ca:
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open('output/%s_%s_wo/best_rt_list_%s.txt'% (cur_date_time, model_name, cur_date_time), 'w') as file:
            for best_rt in best_rt_list:
                #for row in best_rt:
                file.write(str(best_rt))
                file.write('\n')
        with open('output/%s_%s_wo/best_fit_list_%s.txt'% (cur_date_time, model_name, cur_date_time), 'w') as file:
            file.write(str(best_fit_list))    
        with open('output/%s_%s_wo/best_pro_list_%s.txt'% (cur_date_time, model_name, cur_date_time), 'w') as file:
            for best_pro in best_pro_list:  # best_pro 是三维的， 由 sihjsk 组成
                best_pro = best_pro.tolist()
                for si in best_pro:   
                    for hj in si:
                        file.write(str(hj))
                        file.write('\n')
                    file.write('\n')
                file.write('\n')
                file.write('\n')
        if parser_args.learn:   # 开启了在线学习，最后再把学习到的参数保存到文件
            print("保存本次运行学习到的模型参数...")
            torch.save(net, model_file) # 保存模型

if __name__ == "__main__":
    main()

