import torch
from news_tls import data, utils



class PM_Mean_SentenceCollector:
    def __init__(self, clip_sents=5, pub_end=2):
        self.clip_sents = clip_sents
        self.pub_end = pub_end

    def collect_sents(self, ranked_dates, collection, vectorizer, include_titles):
        date_to_pub, date_to_ment = self._first_pass(
            collection, include_titles)
        for d, sents in self._second_pass(
            ranked_dates, date_to_pub, date_to_ment, vectorizer):
            yield d, sents

    def _first_pass(self, collection, include_titles):
        date_to_ment = collections.defaultdict(list)
        date_to_pub = collections.defaultdict(list)
        for a in collection.articles():
            pub_date = a.time.date()
            if include_titles:
                for k in range(self.pub_end):
                    pub_date2 = pub_date - datetime.timedelta(days=k)
                    if a.title_sentence:
                        date_to_pub[pub_date2].append(a.title_sentence)
            for j, s in enumerate(a.sentences):
                ment_date = s.get_date()
                if ment_date:
                    date_to_ment[ment_date].append(s)
                elif j <= self.clip_sents:
                    for k in range(self.pub_end):
                        pub_date2 = pub_date - datetime.timedelta(days=k)
                        date_to_pub[pub_date2].append(s)
        return date_to_pub, date_to_ment

    def _second_pass(self, ranked_dates, date_to_pub, date_to_ment, vectorizer):

        for d in ranked_dates:
            ment_sents = date_to_ment[d] # sentences that mentions date d
            pub_sents = date_to_pub[d]
            selected_sents = []

            if len(ment_sents) > 0 and len(pub_sents) > 0:
                X_ment = vectorizer.transform([s.raw for s in ment_sents])
                X_pub = vectorizer.transform([s.raw for s in pub_sents])

                C_ment = sparse.csr_matrix(X_ment.sum(0))
                C_pub = sparse.csr_matrix(X_pub.sum(0))
                ment_weight = 1 / len(ment_sents)
                pub_weight = 1 / len(pub_sents)
                C_mean = (ment_weight * C_ment + pub_weight * C_pub)
                _, indices = C_mean.nonzero()

                C_date = sparse.lil_matrix(C_ment.shape)
                for i in indices:
                    v_pub = C_pub[0, i]
                    v_ment = C_ment[0, i]
                    if v_pub == 0 or v_ment == 0:
                        C_date[0, i] = 0
                    else:
                        C_date[0, i] = pub_weight * v_pub + ment_weight * v_ment

                ment_sims = cosine_similarity(C_date, X_ment)[0]
                pub_sims = cosine_similarity(C_date, X_pub)[0]
                all_sims = np.concatenate([ment_sims, pub_sims])

                cut = detect_knee_point(sorted(all_sims, reverse=True))
                thresh = all_sims[cut]

                for s, sim in zip(ment_sents, ment_sims):
                    if sim > 0 and sim > thresh:
                        selected_sents.append(s)
                for s, sim in zip(pub_sents, pub_sims):
                    if sim > 0 and sim > thresh:
                        selected_sents.append(s)

                if len(selected_sents) == 0:
                    selected_sents = ment_sents + pub_sents
            elif len(ment_sents) > 0:
                selected_sents = ment_sents
            elif len(pub_sents) > 0:
                selected_sents = pub_sents
            yield d, selected_sents