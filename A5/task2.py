"""
Flajolet-Martin
"""
import sys
import time
import random
import binascii
import statistics
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
    span = [i for i in range(1000) if i%2 == 1]
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
        hash_params = get_hash_params(1000)

    #  ((ax + b) % p) % m
    hashes = []

    # int id
    key = int(binascii.hexlify(s.encode('utf8')), 16)

    for a, b in hash_params:
        hashes.append((a*key+b)%2500)
    return hashes


def trailing_zeros(number):
    # Count number of trailing zeros in the binary representation of int number
    if not number:
        return 0
    binary = bin(number)
    return len(binary) - len(binary.rstrip('0'))


def process(ip_file, s_size, asks, op_file):

    bx = BlackBox()
    result = []
    score = 0
    total_distinct_users = 0

    for i in range(asks):
        # Current batch of users received from blackbox
        stream_users = bx.ask(ip_file, s_size)

        # user -> 100 hashes for each user
        user_hashes = [myhashs(user) for user in stream_users]

        # list of all user hashes for a certain hash function
        per_hash_users = list(zip(*user_hashes))

        # For each hash function, get the length of the longest trailing zeros among all users
        hash_max_trailing_bits = [max([trailing_zeros(hash_user) for hash_user in hash_users])
                                  for hash_users in per_hash_users]

        # get estimate: 2**r
        estimate_distinct_users = [2**r for r in hash_max_trailing_bits]
        # print(estimate_distinct_users)

        # Divide hashes into groups of 10 and take their averages
        avg_estimate = []
        for j in range(0,1000,20):
            avg_estimate.append(statistics.mean(estimate_distinct_users[j:j+20]))
        # median of avg
        estimate = statistics.median(avg_estimate)
        score += estimate
        total_distinct_users += len(set(stream_users))

        result.append('{},{},{}\n'.format(i, s_size, int(estimate)))

    print("Score: {}".format(score/total_distinct_users))

    with open(output_file, 'w') as fp:
        fp.write("Time,Ground Truth,Estimation\n")
        fp.writelines(result)


if __name__ == '__main__':
    start_time = time.time()
    file_name, stream_size, num_of_asks, output_file = get_input()
    process(file_name, stream_size, num_of_asks, output_file)
    print("Duration: {}".format(time.time()-start_time))