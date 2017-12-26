The submission files contains the following files:
Proposal.pdf (review link: https://review.udacity.com/#!/reviews/878560)
Report.pdf
DataPreprocessing.py
Analysis.py
Visulization.py

The code for this project has been contained in following files:
DataPreprocessing.py
Analysis.py
Visulization.py

1.
DataPreprocessing.py is getting the raw data and build indicators to do the training:
(1). To get raw data from yahoo finance run:
get_data_from_yahoo()
All file in 'stock_dfs'folder has been generated
(2). To create features/indicators run:
generateIndicatorDf()
All file in 'stockIndicator_dfs'folder has been generated

2.
Analysis.py is to combine the dataframe for individual stock and index, split the data for training and testing, training and testing the data, get the performance for the model and benchmark and generate prediction figure for the model.
1. To train and get the performance for the model run:
run()
2. To get the performance for the benchmark run:
benchmark()
3. To generate prediction figure run:
plotPriceLabel()

3.
Visulization.py contains the code to plot preditions, raw data of features, it is called by Analysis.py