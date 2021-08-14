echo "##################################################"
echo "### create docker                              ###"
echo "##################################################"
docker pull fnndsc/ubuntu-python3
cd docker/
docker build . -t ubuntu-python3
docker-compose up -d
cd ..