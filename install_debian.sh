mkdir app
cd app

sed -i 's#http://deb.debian.org#https://mirrors.163.com#g' /etc/apt/sources.list
apt-get update && apt-get install -y \
    build-essential automake \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

git clone https://github.com/bforecast/strategy-app.git .
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --upgrade pip

# TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz  
tar zxvf ta-lib-0.4.0-src.tar.gz 
cd ta-lib

get_arch=`arch`
if [[ $get_arch =~ "x86_64" ]];then
    echo "this is x86_64"
    ./configure --prefix=/usr 

elif [[ $get_arch =~ "aarch64" ]];then
    echo "this is arm64"
    cp /usr/share/automake-1.16/config.guess .
    ./configure --prefix=/usr

elif [[ $get_arch =~ "mips64" ]];then
    echo "this is mips64"
else
    echo "unknown!!"
fi
make && make install

pip3 install -r requirements.txt