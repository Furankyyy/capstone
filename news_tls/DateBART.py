from news_tls import data, datewise, summarizers, summarizer_new
from tilse.data.timelines import Timeline as TilseTimeline
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime


class DateBART_TimelineGenerator():

    def __init__(self,
                 date_ranker=None,
                 summarizer=None,
                 sent_collector=None,
                 clip_sents=5,
                 pub_end=2,
                 key_to_model=None):

        self.date_ranker = date_ranker or datewise.MentionCountDateRanker()
        self.sent_collector = sent_collector or datewise.PM_Mean_SentenceCollector(
            clip_sents, pub_end)
        self.summarizer = summarizer or summarizer_new.BART()
        self.key_to_model = key_to_model

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):
        
        print('date ranking...')
        ranked_dates = self.date_ranker.rank_dates(collection)

        start = collection.start.date()
        end = collection.end.date()
        ranked_dates = [d for d in ranked_dates if start <= d <= end]

        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        vectorizer.fit([s.raw for a in collection.articles() for s in a.sentences])

        print('candidates & summarization...')
        dates_with_sents = self.sent_collector.collect_sents(
            ranked_dates,
            collection,
            vectorizer,
            include_titles=False,
            )

        timeline = []
        l = 0

        for i, (d, d_sents) in enumerate(dates_with_sents):

            if l >= max_dates:
                break

            summary = self.summarizer.summarize(d_sents)
            if summary:
                time = datetime.datetime(d.year, d.month, d.day)
                timeline.append((time, summary))
                l += 1
            

        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)
    
    def load(self, ignored_topics):
        key = ' '.join(sorted(ignored_topics))
        if self.key_to_model:
            self.date_ranker.model = self.key_to_model[key]