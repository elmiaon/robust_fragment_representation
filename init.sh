echo "##################################################"
echo "### load data                                  ###"
echo "##################################################"
bash load_raw.sh

echo "##################################################"
echo "### create docker                              ###"
echo "##################################################"
docker pull fnndsc/ubuntu-python3
cd docker/
docker build . -t ubuntu-python3
docker-compose up -d
cd ..
docker exec -ti ubuntu-python3 /bin/bash
cd /mount
