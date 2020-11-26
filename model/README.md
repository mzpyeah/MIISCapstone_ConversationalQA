# BERT-FlowDelta

## Setup

Install all depedent python packages

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
sh train_quac.sh
```
## Test

```
sh test_quac.sh
```

please refer to our CodaLab submission pages for inference details and pretrained models

[QuAC](https://worksheets.codalab.org/worksheets/0xacb00235ee6b42b3aa682c5d62204a81)
