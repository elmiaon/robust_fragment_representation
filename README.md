# Robust Fragment-Based Representation Framework for Cross-lingual Sentence Retrieval (EMNLP'21 finding)

@inproceedings{trijakwanich-etal-2021-robust-fragment,
    title = "Robust Fragment-Based Framework for Cross-lingual Sentence Retrieval",
    author = "Trijakwanich, Nattapol  and
      Limkonchotiwat, Peerat  and
      Sarwar, Raheem  and
      Phatthiyaphaibun, Wannaphong  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.80",
}


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

## How to use your own data
You can use your own dataset by follow these steps

1. Prepare your data: Your data must have tuning set and test set. Each set composed of 3 files.

    1. SOURCE_LANGUAGE-TARGET_LANGUAGE.SOURCE_LANGUAGE.csv: columns = [id, SOURCE_LANGUAGE]
    2. SOURCE_LANGUAGE-TARGET_LANGUAGE.TARGET_LANGUAGE.csv: columns = [id, TARGET_LANGUAGE]
    3. SOURCE_LANGUAGE-TARGET_LANGUAGE.gold.csv: columns = [SOURCE_LANGUAGE, TARGET_LANGUAGE]

    You can see the test data for the format of each file.
    
2. Put your files in the following directory

    data/reformatted/DATASET_NAME/tuning/
    
    data/reformatted/DATASET_NAME/test/

3. Your dataset in the config/corpus.json, for example,

    <pre><code>
    "YOUR_DATASET": [
        ["YOURE_DATASET"  , "tuning", "YOUR_DATASET"  , "test"],
    ],
    </code></pre>

4. Add new language config in config/language.json, for example,
    
    <pre><code>
    "YOUR_DATASET":[
        ["SOURCE_LANGUAGE", "TARGET_LANGUAGE"]
    ]
    </code></pre>

5. Duplicate and rename your running file from TEMPLATE, then, replace the setting at the head of the file.

6. Running you file.
