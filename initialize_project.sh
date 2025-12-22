#!/bin/bash


echo -e "[FILE INTIALIZATION...]\n"

tr -d '\n' < ./ConfigurationModule/ConfigFiles/supported_devices_library.json | \
sed 's/.*\[\(.*\)\].*/\1/' | \
tr ',' '\n' | \
sed 's/[" ]//g' | \
awk 'BEGIN {print "{"} {printf "%s  \"%s\": false", (NR>1?",\n":""), $1} END {print "\n}"}' \
> ./PackageDownloadModule/requirementsFileDirectory/.installed.json

echo -e "File .installed.json setted...\n"


echo -e "[BASE DEPENDENCIES INSTALLATION...]\n"


rm -rf venv || true; python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install -r requirements.txt
chmod +x ./Utils/scripts/cleancache.sh
#De-comment this line in order to not prompt sudo password request during the execution...

echo -e "\n[DROP CACHES] \n"
echo "Searching if tee has root grand in sudoers file to modify /proc/sys/vm/drop_caches. May ask password for checking..."


if sudo grep -q "^${USER} ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" /etc/sudoers; then 
  echo "The sudo grant to /usr/bin/tee is alreay allowed on /proc/vm/sys/drop_caches in /etc/sudoers file!"
else
  echo "${USER} ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"| sudo EDITOR='tee -a' visudo
fi

echo "Done!"
