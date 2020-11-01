"""
Fixed Size Sampling
"""
import sys
import time
import random
from blackbox import BlackBox

hash_params = None

def get_input():
    file_name_ = sys.argv[1]
    stream_size_ = sys.argv[2]
    num_of_asks_ = sys.argv[3]
    output_file_ = sys.argv[4]
    return file_name_, int(stream_size_), int(num_of_asks_), output_file_


def process(ip_file, s_size, asks, op_file):

    bx = BlackBox()
    result = []

    # Fixed size sample of size s_size
    users = None
    # serial number of the next item
    n = 1

    for i in range(asks):
        # Current batch of users received from blackbox
        stream_users = bx.ask(ip_file, s_size)

        if not users:
            users = stream_users[::]
            n += 100
            result.append(
                "{},{},{},{},{},{}\n".format(100, users[0], users[20], users[40], users[60], users[80]))
            continue

        for j in range(s_size):
            r = random.randint(0, 100000)
            if r%n < s_size:
                idx = random.randint(0, 100000) % s_size
                users[idx] = stream_users[j]
            n += 1

        result.append("{},{},{},{},{},{}\n".format((i+1)*100,users[0],users[20],users[40],users[60],users[80]))

    with open(output_file, 'w') as fp:
        fp.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        fp.writelines(result)


if __name__ == '__main__':
    random.seed(553)
    start_time = time.time()
    file_name, stream_size, num_of_asks, output_file = get_input()
    process(file_name, stream_size, num_of_asks, output_file)
    print("Duration: {}".format(time.time()-start_time))