#use ./build.sh 1.0
cd ../..
docker build -t nurcsp:$1 -f Papers/NURCSP/Dockerfile .