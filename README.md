# DSBA_6211_Airbnb_Nashville

I have not uploaded the data folder to avoid storage problem.

1. First file is analyze.ipynb & it's corresponding tableau file Analysis.twb
2. Second file is reviews_keyphrases.ipynb
3. Third file is sentiment_analysis_reviews.ipynb
4. Fourth File is - prepare_clean_comments.ipynb
5. Fifth File is - extraction_of_entities_from_reviews.py . This uses the parquet file generated in 4th step.

Note:- For reviews file the comments column is procesed for each listing by extracting keyphrases, sentiments & entities for each comment. This was the process which took longer durations  due to the file size  and models used. The checkpoints for each process is created by saving the output in parquet type file. The analyis is still pending to be done on the data.
