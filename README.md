# fragment_encoder
README composed of

- Build and run docker
- Load and prepare data
- Run the program
- How to use your own data

## Build and run docker

after clone this git, run the snippet below to run docker

    sh init.sh

Attach to running docker

    docker exec -ti fragment_encoder /bin/bash
    
change direction to /mount

    cd /mount
    
## Load and prepare data
    
Run a snippet below to load dataset. This step may take several minutes to hours.

    bash load_opus.sh
    
At this point, you should get the data directory as shown below.

- data/
    - raw/
        - opus/
            - JW300/
            - QED/
            - TED2020/

In JW300, QED, and TED2020 should have following files.

- ar-en.ar
- ar-en.en
- de-en.de
- de-en.en
- fr-en.en
- fr-en.fr
- th-en.en
- th-en.th

Run this snippet to reformat the data.

    python3 create.opus.py

## Running the program
run this file to test the program.

    python3 RFRv2.JW300s0main.test.py
    
If the program runs correctly, you will get this file.

    data/tested/RFRt.RFRr0.cosine50.RFRa0/test_opus-JW300-CLSRs0_te-c50.tune_opus-JW300-CLSRs0_tu-c50.fr-en.csv

which is the csv file that have same result with the table below.

| k | beta | fil              | p_thres           |n  |acc   | fil_p             | fil_r| fil_f1            |align_p            | align_r| align_f1          |
|---|------|------------------|-------------------|---|------|-------------------|------|-------------------|-------------------|--------|-------------------|
|45 |90    |0.6000000000000001| 0.7999999999999999| 1 | 0.817| 0.9968652037617555| 0.636| 0.7765567765567766| 0.9968652037617555| 0.636  | 0.7765567765567766|
|45 |90    |0.6000000000000001| 0.7999999999999999| 5 | 0.817| 0.9968652037617555| 0.636| 0.7765567765567766| 0.9968652037617555| 0.636  | 0.7765567765567766|
|45 |90    |0.6000000000000001| 0.7999999999999999| 10| 0.817| 0.9968652037617555| 0.636| 0.7765567765567766| 0.9968652037617555| 0.636  | 0.7765567765567766|

In addition, you will have the pairing result as csv file in the following directory.

    data/ans/RFRt.RFRr0.cosine50.RFRa0/test_opus-JW300-CLSRs0_te-c50.tune_opus-JW300-CLSRs0_tu-c50.fr-en.csv
    
the format is shown below.

- id: sentence id of each query sentence in source language
- candidates: tuples of candidate sentences sort by probability in descending order
- prob: byte string of probability of each candidate sentence
- ans: boolean indicate if top 1 candidate probability > p_thres?
    - True means top1 is a translation of query sentence
    - False means this is a non-pairing query sentence
