from news_tls import data, datewise, summarizers, summarizer_new, DateBART
from tilse.data.timelines import Timeline as TilseTimeline
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

dataset = data.Dataset('/mnt/d/furan/Documents/Projects/Capstone/Entities')

#datewise.SupervisedDateRanker(method='regression')

"""
for i, collection in enumerate(dataset.collections):
    testdata = collection
    ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]
    ref_dates = sorted(ref_timelines[0].dates_to_summaries)
    start, end = data.get_input_time_span(ref_dates, extension=0)
    testdata.start = start
    testdata.end = end

    break
"""


# Data
for i, collection in enumerate(dataset.collections): # iterate over "entity"
	# Name
    
    if collection.name=="Tiger_Woods":
        '''
        for x in collection.articles():
            if x.time.year == 2017:
                print(x.text)

        '''
    # This is for reference timelines for the entity. Normally should only be one reference timeline.
    # For entities dataset it is all 1 ref
        for tl in collection.timelines: 
            print(tl)
            print(tl.date_to_summaries)
