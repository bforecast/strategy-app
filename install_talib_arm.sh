apt install -y build-essential automake
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz  
tar zxvf ta-lib-0.4.0-src.tar.gz 
cd ta-lib
cp /usr/share/automake-1.16/config.guess .
./configure --prefix=/data/data/com.termux/files/usr
make && make install
pip install TA-Lib
