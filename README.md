# YouTube cantopop title parser

We can download a YouTube video with `youtube_dl`. I usually do this to collect
cantopop in MP3 format but the issue will be the id3 tags.

This is a script to train a MLP to figure out the artist and song title from
the YouTube video title. I hand crafted the features (should try word2vec but
I did not) and feed into a simple 3-layer MLP to identify tokens.

The training data is in `titles.txt` and I used `crawler-dbm.py` to preprocess
the data into `feat.pickle`. Then running `train.py` will train a MLP (using
scikit-learn) for the purpose, which is then saved as `mlp-trained.pickle`.

When we have a trained model, we can tag all MP3 files based on their filename
(as if `youtube_dl` give you by default):

    python deploy.py [files...]

The ID3v2 access is using mutagen library.

An alternative version is built using Keras/tensorflow as well:
`train_keras.py` and `deploy_keras.py`. The model of MLP is same as
scikit-learn but the code is slower to initialize.
