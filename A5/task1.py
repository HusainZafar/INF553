import sys
import random
import time
import binascii
from blackbox import BlackBox

hash_params = None

def get_input():
    file_name_ = sys.argv[1]
    stream_size_ = sys.argv[2]
    num_of_asks_ = sys.argv[3]
    output_file_ = sys.argv[4]
    return file_name_, int(stream_size_), int(num_of_asks_), output_file_


def get_hash_params(count):
    random.seed(553)
    return [(random.randint(1,1000), random.randint(1,1000)) for i in range(count)]


def myhashs(s):
    """
    Hashes for bloomfilter
    :param s: user_id of type string
    :param hash_params: a, b values for each hash
    :return:
    """
    global hash_params
    if not hash_params:
        hash_params = get_hash_params(40)

    #  ((ax + b) % p) % m
    hashes = []

    # int id
    key = int(binascii.hexlify(s.encode('utf8')), 16)

    for a, b in hash_params:
        hashes.append((a*key+b)%69997)
    return hashes


def process(ip_file, s_size, asks, op_file):
    """
    Bloomfilter implementation.
    """
    bloomfilter_array = [0 for i in range(69997)]
    users_seen = set()
    bx = BlackBox()
    result = []

    for i in range(asks):
        stream_users = bx.ask(file_name, stream_size)
        fp = 0
        for user in stream_users:
            flag = 1
            for bit in myhashs(user):
                if not bloomfilter_array[bit]:
                    flag = 0
                bloomfilter_array[bit] = 1
            if flag and user not in users_seen:
                fp += 1
        users_seen.update(stream_users)
        result.append('{},{}\n'.format(i, fp/100))

    with open(output_file, 'w') as fp:
        fp.write("Time,FPR\n")
        fp.writelines(result)


if __name__ == '__main__':
    start_time = time.time()
    file_name, stream_size, num_of_asks, output_file = get_input()
    process(file_name, stream_size, num_of_asks, output_file)
    print("Duration: {}".format(time.time()-start_time))