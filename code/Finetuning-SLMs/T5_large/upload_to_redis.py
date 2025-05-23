import redis
import tqdm
import argparse
import os
import random
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import numpy as np
import secrets
import string
import re


def is_empty(input_text):
    input_text = input_text.strip()
    if input_text=='' or len(input_text)==0:
        return True
    return False

def load_text_stream(load_path, max_limit=-1):
    index = 0
    with open(load_path, 'r', encoding='utf-8') as f:
        for row in f:
            index+=1
            input_text = row.strip()
            if (max_limit>0 and index>max_limit):
                yield False, ""
            yield True, input_text
    yield False, ""

def upload_to_redis(args):
    # started redis connection
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db)
    
    random.seed(42)
    
    streams = load_text_stream(args.file_path, args.data_limit)
    stats = 0
    query_stats ={"total_instances": 0, "except_count": 0}
    dtype_query = "t" if args.dtype=="train" else "v"

    total_instances = 0
    except_count = 0
    start_time = datetime.utcnow()    
    while True:
        valid, field = next(streams)
        if not valid:
            break
        if is_empty(field):
            continue
        row = field.strip().split('\t')
        query=row[0]
        enhanced_query=row[1]
        stats+=1
        if is_empty(query):
            continue
        if is_empty(enhanced_query):
            continue
        try:
            redis_server.hset("%s:%s"%(dtype_query, str(total_instances)), mapping={"head": query, "tail": enhanced_query})
        except Exception as ex:
            except_count+=1
            query_stats["except_count"]+=1
            continue
        total_instances+=1
        query_stats["total_instances"]+=1
        if total_instances % args.log_freq == 0:
            time_diff = (datetime.utcnow() - start_time).total_seconds()
            print("complete %d in %0.2f secs" % (total_instances, time_diff))
            start_time = datetime.utcnow()
    print("**"*30)
    print("total instances : %d, exception count : %d" % (total_instances, except_count))
    redis_server.hset("dataset_len", args.dtype, str(total_instances))
    print("%d [%0.2f]" % (total_instances, total_instances/stats))

def main():
    parser = argparse.ArgumentParser(description='Generate and import to Redis DB')
    parser.add_argument('--file_path',
                        type=str,
                        required=True,
                        help='file containing tab separated fields: query, language')
    parser.add_argument('--redis-host',
                        type=str,
                        default='0.0.0.0',
                        help='Redis Host or IP Address')
    parser.add_argument('--redis-port',
                        type=int,
                        default=6379,
                        help='Redis Host or IP Address')
    parser.add_argument('--redis-db',
                        type=int,
                        default=0,
                        help='Redis DB')
    parser.add_argument('--dtype',
                        type=str,
                        required=True,
                        help='data type : train or validation', choices=['train', 'val'])
    parser.add_argument('--log-freq',
                        type=int,
                        default=1e6,
                        help='logging frequency...')
    parser.add_argument('--data-limit',
                        type=int,
                        default=-1,
                        help='maximum number of instances to load. -1 to include all.')
    
    args = parser.parse_args()
    start_time = datetime.utcnow()
    upload_to_redis(args)
    time_diff = (datetime.utcnow() - start_time).total_seconds()
    print("completed populating redis in %0.3f secs" % time_diff)
    
if __name__ == "__main__":
    main()
