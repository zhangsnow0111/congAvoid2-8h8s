import subprocess 

def queue_rate_set(rate):
    for i in range(8):
        p = subprocess.Popen('simple_switch_CLI --thrift-port 909%d'%i,shell=True,stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                universal_newlines=True) 
        
        p.stdin.write(f'set_queue_rate {rate}')
        
        # print(f'修改交换机{i}的queue_rate')
        # 读取命令行界面的输出
        output = p.communicate()[0]
        # # 打印输出
        print(output)
        
        # p.communicate()
    p.stdin.close()

def reset_idx():
    for i in range(8):
        p = subprocess.Popen('simple_switch_CLI --thrift-port 909%d'%i,shell=True,stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                universal_newlines=True) 
        
        p.stdin.write(f'register_reset reg_idx_and_cnt')
        
        #print(f'修改交换机{i}的queue_rate')
        # 读取命令行界面的输出
        output = p.communicate()[0]
        # # 打印输出
        print(output)
        
        # p.communicate()
    p.stdin.close()


# def queue_rate_set(name, index, value):
#     p = subprocess.Popen('simple_switch_CLI', shell=True, stdin=subprocess.PIPE,
#                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                             universal_newlines=True)
#     p.stdin.write(f'register_write reg_idx_and_cnt') 
#     p.communicate()

def main():
    # for i in range(8):
    queue_rate_set(600)
    #reset_idx()

if __name__ == "__main__":
    main()