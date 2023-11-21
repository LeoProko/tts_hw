echo "download LJ speech"
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
rm LJSpeech-1.1.tar.bz2

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

echo "download Waveglow"
gdown 'https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx'
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)
rm mel.tar.gz

echo "download alignments"
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
rm alignments.zip

# # we will use waveglow code, data and audio preprocessing from this repo
# git clone https://github.com/xcmyz/FastSpeech.git
# mv FastSpeech/text ./
# mv FastSpeech/audio ./
# mv FastSpeech/waveglow/* waveglow/
# mv FastSpeech/utils.py ./
# mv FastSpeech/glow.py ./
# rm -rf FastSpeech
