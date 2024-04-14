#use ./build.sh 1.0
cd ../..
docker build -t nurcsp-arandu:$1 -f Papers/NURCSP/Dockerfile .
