#!/bin/bash
sudo apt install python3-pip
echo "pip3 version"
echo pip3 --version


if [ $? -eq 0 ]; then
	echo "OK : virtualenv already installed"
else
	echo "FAIL : virtualenv load failure"
	echo "pip3 installing virtualenv"
	sudo pip3 install virtualenv
	echo "Installed virtualenv"
fi
	virtualenv -p python3 ./sensorEngineDevelopmentSetup

	source ./sensorEngineDevelopmentSetup/bin/activate

python3 -c "import imgaug"
if [ $? -eq 0 ]; then
	echo "OK : import imgaug"
else
	echo "FAIL : import imgaug"
	echo "pip3 installing imgaug"
	pip3 install imgaug
	echo "Installed imgaug"
fi

python3 -c "import keras"
if [ $? -eq 0 ]; then
	echo "OK : import keras"
else
	echo "FAIL : import keras"
	echo "pip3 installing keras"
	pip3 install keras==2.2.4
	echo "Installed keras"
fi

python3 -c "import sklearn"
if [ $? -eq 0 ]; then
	echo "OK : import scikit-learn"
else
	echo "FAIL : import scikit-learn"
	echo "pip3 installing scikit-learn"
	pip3 install scikit-learn
	echo "Installed scikit-learn"
fi


python3 -c "import tensorflow"
if [ $? -eq 0 ]; then
	echo "OK : import tensorflow-gpu"
else
	echo "FAIL : import tensorflow-gpu"
	echo "pip3 installing tensorflow-gpu"
	pip3 install tensorflow-gpu==1.14.0
	#pip3 install --upgrade tensorflow
	echo "Installed tensorflow-gpu"
	echo Verifying TF installation
	python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
	if [ $? -eq 0 ]; then
		echo "OK : tensorflow-gpu Installation success.."
	else
		echo "FAIL : tensorflow installation FAILURE !!!"
	fi
fi

python3 -c "import tqdm"
if [ $? -eq 0 ]; then
	echo "OK : import tqdm"
else
	echo "FAIL : import tqdm"
	echo "pip3 installing tqdm"
	pip3 install tqdm
	echo "Installed tqdm"
fi
python3 -c "import lxml"
if [ $? -eq 0 ]; then
	echo "OK : import lxml"
else
	echo "FAIL : import lxml"
	echo "pip3 installing lxml"
	pip3 install lxml
	echo "Installed lxml"
fi

python3 -c "import BeautifulSoup"
if [ $? -eq 0 ]; then
	echo "OK : import BeautifulSoup"
else
	echo "FAIL : import BeautifulSoup"
	echo "pip3 installing BeautifulSoup"
	pip3 install beautifulsoup4
	echo "Installed BeautifulSoup"
fi

python3 -c "import pydot"
if [ $? -eq 0 ]; then
	echo "OK : import pydot"
else
	echo "FAIL : import pydot"
	echo "pip3 installing pydot"
	pip3 install pydot
	echo "Installed pydot"
fi

python3 -c "import pyedflib"
if [ $? -eq 0 ]; then
	echo "OK : import pyedflib"
else
	echo "FAIL : import pyedflib"
	echo "pip3 installing pyedflib"
	pip3 install pyedflib
	echo "Installed pyedflib"
fi

python3 -c "import mne"
if [ $? -eq 0 ]; then
	echo "OK : import mne"
else
	echo "FAIL : import mne"
	echo "pip3 installing mne"
	pip3 install mne
	echo "Installed mne"
fi

