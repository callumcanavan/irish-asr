git clone https://github.com/libsndfile/libsndfile.git
cd libsndfile
sudo apt install autoconf autogen automake build-essential libasound2-dev \
  libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev \
  libmpg123-dev pkg-config python
autoreconf -vif
./configure --enable-werror
sudo make
sudo make check
sudo make install
sudo mkdir /usr/local/lib/python3.8/dist-packages/_soundfile_data/ -p
sudo cp /usr/local/lib/libsndfile.* /usr/local/lib/python3.8/dist-packages/_soundfile_data/