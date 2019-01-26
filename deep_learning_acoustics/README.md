# Workshop for PyLadies: February 12th, 2019

**In the works and subject to change!!** For background, please reference my <a href="https://a-n-rose.github.io/2019/01/13/workshop-pyladies-deeplearning-speech.html">blog post</a>.

## The Application of Deep Learning to Acoustics via Python

Sound is incredibly complex. A lot of information is stored within sound waves, ranging from speech to envioronmental information, with a lot of additional information squeezed in, i.e. noise. In recent times, with the development of various deep learning algorithms, key information can be more easily extracted despite the complexities of sound and the noise that is so often present. 

In this workshop, I will lightly cover the background of both sound and a few deep neural networks. With this knowledge, I will accompany the participants through the collection of acoustic data, in this case speech and noise, its preprocessing and preparation for training of deep neural networks with the end goal in mind: to deploy the model we train on **new real-world** speech.
 
By the end of this workshop, participants should know how to:
 
* collect speech from an online database

* store the speech into a local feature database via SQlite3

* prepare speech data to train deep neural networks: convolutional neural networks( ConvNet), long short-term memory networks (LSTM), and ConvNet+LSTM neural networks

* deploy the trained model in an application to process real-word data
 

The task for the workshop sounds simple: to train a neural network to identify male or female speech, but it is quite a task to take on, in one evening, the preparation of sound data for deep learning, to train the models, and then to deploy them. We will get as far as we can! All while learning little tricks in Python to help in any project you develop. 
 
## Virtual Environment

I suggest using a virtual environemt. This allows you to use any package versions without them interfering with other programs on your system. 

You can set up a virtual environment different ways. One way is with Python3.6-venv.

### Python3.6-venv

To install, enter into the command line:

(Linux)
```
sudo apt install python3.6-venv
```

In folder of cloned directory (i.e. "deep_learning_acoustics"), write in the command-line:

```
python3 -m venv env
```

This will create a folder 'env'.

Then type into the command-line:

```
source env/bin/activate
```

and your virtual envioronment will be activated. Here you can install packages that will not interfere with any other programs.

To deactivate this environment, simply type in the command line:

```
deactivate
```


### PyCharm

Another option to explore (and would be good to know) is PyCharm. PyCharm creates a virtual environment *for* you. You can download it from here:

https://www.jetbrains.com/pycharm/

I suggest the community version as it is free and open source. It is super easy on linux via snap (you might have to install snap first)

```
sudo apt install snap

sudo snap install pycharm-community --classic
```

Once installed, you can run PyCharm by entering the following into the command line:

```
pycharm-community
```
 
## Prerequisites

1) Computer with CPU or GPU and sound recording capabilities, ideally via headset

2) A way to examine SQL tables (I use <a href="https://sqlitebrowser.org/">DB Browser for SQLite</a>)

3) Python 3.6

To check your version type the following into the command-line (Linux):

```
python3 --version
```

4) For PyAudio, need to install portaudio19-dev. Otherwise can't pip install pyaudio :(

```
sudo apt-get install portaudio19-dev
```

To install all the python packages we will use, first start up your virtual environment (in the folder "deep_learning_acoustics"):

```
source env/bin/activate
```

4) In your virtual environment, run 'requirements.txt' to install all necessary packages via pip. 

```
pip install -r requirements.txt
```

## Download the Data

Please download the female and male speech data *before* coming to the workshop. The database may limit the number of requests for download, so there is a risk that you won't be able to download it the evening of (although I doubt it).

Request the data from here: 

<a href="http://www.stimmdatenbank.coli.uni-saarland.de/index.php4#target">Saarbrücker Voice Database</a>

### Male Speech Data:

I've posted some screenshots to help you navigate the webpage. We will first download healthy male speech, then healthy female speech. The webpage is in German, so real quick: 

healthy = gesund
male = männlich
female = weiblich


1) Check the boxes for "männlich" and "gesund"

2) Click "Exportieren" at the bottom of the page.

![Imgur](https://i.imgur.com/KcKSA4x.png)

3) Check the boxes "Satzdatei" and "WAV" for the 'Sprach-Signal' (not the 'EEG-Signal').

4) Once the data has loaded, click on "Herunterladen". Save this zip file in the workshop's "data" folder as "male_speech.zip". Extract the zipfile. 

![Imgur](https://i.imgur.com/26Fz5EE.png)

5) Rename the extracted file "export" as "male_speech"

![Imgur](https://i.imgur.com/SHhfjxM.png)
##### This is just to show how the export file should be renamed to "male_speech" and "female_speech"

6) On the database website: click on "Zurück" and "Neue Anfrage"

This should reset the values and you can do the same to download the female speech.

### Female Speech Data

Now, repeat the process for the female speech: 

1) Check the boxes for "weiblich" and "gesund"

2) Click "Exportieren"

3) Check the boxes "Satzdatei" and "WAV" for the 'Sprach-Signal' (not the 'EEG-Signal').

4) Once the data has loaded, click on "Herunterladen". Save this zip file in the "data" folder as "female_speech.zip". Extract the zipfile. 

5) Rename the extracted file "export" as "female_speech"

### Check the speech

Ensure the downloads were successful. You should have 252 wave files of men and 632 wave files of women saying the phrase "Guten Morgen, wie geht es Ihnen". 


## Set up the Database

Run the following script to set up the database and table architecture:

```
python3 set_up_sql.py
```

## Record background noise

This should be done "on site" as I want to deploy the model where it will be used. 
The aim here is to train the model with noise that will be prevalent where the model will be implemented

To do this, run the following:
```
python3 collect_background_noise.py
```

## Extract MFCC features and save to SQL database

Finally, to extract the MFCC features with varying amounts of noise combined, run the following:

```
python3 speech_prep.py
```

## Train the ConvNet+LSTM

To get the data split into the train, validation, and test datasets, to zero-pad them, and get them in the right dimensions, oh and of course to train and test a male-female speech classifer, run the following:

```
python3 train_models.py
```
To run through 100 epochs, it should only take around 5 minutes on a CPU. Once the model is finished traing, it saves the architecture and the weights in a h5 file.

## Classify new data with the trained model

To see how well the classifier handles new data, run the following script:

```
python3 implement_model.py
```

This will ask the user/s to record themselves saying "Hallo, wie geht es Ihnen". The users can try saying anything, but since the classifier was trained with speakers only saying this phrase, it will likely have more success if users say the same thing as well.

