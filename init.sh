echo "##################################################"
echo "### create docker                              ###"
echo "##################################################"
docker pull fnndsc/ubuntu-python3
cd docker/
docker build . -t ubuntu-python3
docker-compose up -d
cd ..
docker exec -ti fragment_encoder /bin/bash
cd /mount

echo "##################################################"
echo "### load data                                  ###"
echo "##################################################"
bash load_raw.sh