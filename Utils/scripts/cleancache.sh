#!/bin/bash
/usr/bin/sync
echo 3 |sudo tee /proc/sys/vm/drop_caches
