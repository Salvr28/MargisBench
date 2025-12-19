#!/bin/bash

echo "[BASE DEPENDENCIES INSTALLATION...]"

source venv/bin/activate
python3.10 -m pip install -r requirements.txt
chmod +x ./Utils/scripts/cleancache.sh
#De-comment this line in order to not prompt sudo password request during the execution...

echo " " && echo "[DROP CACHES]"
echo "Searching if tee has root grand in sudoers file to modify /proc/sys/vm/drop_caches. May ask password for checking..."


if sudo grep -q "^${USER} ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" /etc/sudoers; then 
  echo "The sudo grant to /usr/bin/tee is alreay allowed on /proc/vm/sys/drop_caches in /etc/sudoers file!"
else
  echo "${USER} ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"| sudo EDITOR='tee -a' visudo
fi

echo "Done!"
