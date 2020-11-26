# Our model

## Setup

Install Python packages

```
pip install -r requirements.txt
```

Download necessary data

```
sh download.sh
```

Download ConceptNet

```
mkdir ConceptNet_data
```
Then download the file from https://drive.google.com/file/d/14nb2lM_KrWReSHlEaXVg9KE1WrcAV2Lj/view and put it in the folder.

## Train

```
sh train_quac_[bert/roberta/cs].sh
```
## Test

Remove the --do_train flag in the corresponding .sh file.
