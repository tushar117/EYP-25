#!/bin/bash

mkdir -p /usr/local/lib/
chmod a+w /usr/local/lib/
tar -C /usr/local/lib/ -xzf ./redis-stable.tar.gz
cd /usr/local/lib/redis-stable/
make && make install