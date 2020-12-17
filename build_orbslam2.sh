# setup
apt-get -y update
apt-get -y upgrade
apt-get install -y cmake git unzip wget pkg-config build-essential libgl1-mesa-dev libglew-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev

# build opencv
wget https://github.com/opencv/opencv/archive/3.4.12.zip
unzip 3.4.12.zip
cd opencv-3.4.12
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_GENERATE_PKGCONFIG=ON ..
make 
make install
pkg-config opencv --modversion #(the output should be "3.4.12")
cd ../..

# build pangolin (visualizer)
git clone https://github.com/stevenlovegrove/Pangolin
cd Pangolin
mkdir build && cd build
cmake .. 
cmake --build .
cd ../..

# build eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.zip
unzip eigen-3.3.8.zip
cd eigen-3.3.8
mkdir build && cd build
cmake ..
make
make install
cd ../..

# build ORB_SLAM2
cd ORB_SLAM2

# build ORB-SLAM2
chmod +x ./build.sh
./build.sh


