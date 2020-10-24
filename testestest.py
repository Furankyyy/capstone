from news_tls import data, datewise, summarizers, summarizer_new, DateBART
from tilse.data.timelines import Timeline as TilseTimeline
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from DateBART import DateBART_TimelineGenerator


dataset = data.Dataset('/mnt/d/furan/Documents/Projects/Capstone/Entities')

#datewise.SupervisedDateRanker(method='regression')


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
ranker = datewise.MentionCountDateRanker()
collector = datewise.PM_Mean_SentenceCollector(clip_sents=5,pub_end=2)
summarizer = summarizers.CentroidOpt()
summarizer0 = summarizer_new.BART()

ranked_dates = ranker.rank_dates(testdata)

start = testdata.start.date()
end = testdata.end.date()
ranked_dates = [d for d in ranked_dates if start <= d <= end]

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
vectorizer.fit([s.raw for a in testdata.articles() for s in a.sentences])

dates_with_sents = collector.collect_sents(
            ranked_dates,
            testdata,
            vectorizer,
            include_titles=False,
        )

for i, (d, d_sents) in enumerate(dates_with_sents):

    summary = summarizer.summarize(
                d_sents,
                k=1,
                vectorizer=vectorizer,
                filter=None
            )
    
    summary_0 = summarizer0.summarize(d_sents)

    break

print(summary)
print(summary_0)
print(collection.timelines[0])
"""
###完整验证 - evaluation
l = len(ref_dates)

model = DateBART.DateBART_TimelineGenerator()

print(model.predict(testdata,max_dates=l))
