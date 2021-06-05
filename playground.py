import src.load_corpus as load_corpus
import numpy as np
import src.log as log

def main():

    # read input - get sdf, tdf, gdf
    # sub_corpus = ['JW300', 'QED', 'TED2020']
    # src = ['fr', 'de', 'ar', 'th']
    # tar = ['en']
    # for sub in sub_corpus:
    #     for s in src:
    #         for t in tar:
    #             sdf, tdf, gdf = load_corpus.load_opus('JW300', s, t)
    #             sdf[s].replace('', np.nan, inplace=True)
    #             sdf = sdf.dropna()
    #             tdf[t].replace('', np.nan, inplace=True)
    #             tdf = tdf.dropna()
    #             gdf = gdf.loc[gdf[s].isin(sdf['id'].values) & gdf[t].isin(tdf['id'].values)]
    #             print(f"sdf:{sdf.shape}, \n{sdf[sdf.isna().any(axis=1)]}")
    #             print(f"tdf:{tdf.shape}, \n{tdf[tdf.isna().any(axis=1)]}")
    #             print(f"gdf:{gdf.shape}, \n{gdf[gdf.isna().any(axis=1)]}")

    corpus = 'opus'
    sub_corpus = 'QED'
    src = 'th'
    tar = 'en'
    load_corpus.load_CLSR((corpus, sub_corpus, src, tar))
    # tokenize - save

    # embed

    # retrieve

    # aggregate

    # answer

    # calculate score

    # analyse

    pass

if __name__ == '__main__':
    logger = log.init_logger(__file__, __name__, "DEBUG")
    main()