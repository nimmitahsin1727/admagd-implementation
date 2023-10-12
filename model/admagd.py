import numpy as np
from gensim.models import AuthorTopicModel

class AuthorTopicModel:
    def __init__(self, corpus=None, num_topics=10, id2word=None, author2doc=None, *args, **kwargs):
        self.corpus = corpus
        self.num_topics = num_topics
        self.id2word = id2word
        self.author2doc = author2doc

class ADMAGD(AuthorTopicModel):
    def __init__(self, corpus=None, num_topics=10, id2word=None, authors=None, 
                 alpha_init=0.1, beta_init=0.1, a_init=0.1, b_init=0.1, *args, **kwargs):
        
        self.vocab_size = len(id2word)
        
        self.alpha = np.full(num_topics, alpha_init)
        self.beta = np.full(len(id2word), beta_init)
        self.a = a_init
        self.b = b_init
        
        self.authors = list(authors.keys())
        
        super(ADMAGD, self).__init__(corpus=corpus, num_topics=num_topics, id2word=id2word, author2doc=authors, *args, **kwargs)
        
        self.word_topic_matrix = np.zeros((num_topics, len(self.id2word)))
        self.author_topic_matrix = np.zeros((len(self.authors), num_topics))
        self.topic_counts = np.zeros(num_topics)
        self.topic_assignments = []

        for doc_id, document in enumerate(corpus):
            current_doc_assignments = []
            for word_pos, (word_id, _) in enumerate(document):
                initial_topic = np.random.choice(num_topics)
                current_doc_assignments.append(initial_topic)
                self.word_topic_matrix[initial_topic][word_id] += 1
                self.topic_counts[initial_topic] += 1
                author = self.get_author(doc_id)
                author_idx = self.authors.index(author)
                self.author_topic_matrix[author_idx][initial_topic] += 1
            self.topic_assignments.append(current_doc_assignments)

    def get_author(self, doc_id):
        return next(author for author, docs in self.author2doc.items() if doc_id in docs)

    def get_current_topic(self, doc_id, word_pos):
        return self.topic_assignments[doc_id][word_pos]

    def decrement_counts(self, doc_id, word_pos, current_topic, author):
        word_id = self.corpus[doc_id][word_pos][0]
        self.word_topic_matrix[current_topic][word_id] -= 1
        self.topic_counts[current_topic] -= 1
        author_idx = self.authors.index(author)
        self.author_topic_matrix[author_idx][current_topic] -= 1

    def calculate_phi_update(self):
        return (self.word_topic_matrix + self.beta) / (self.word_topic_matrix.sum(axis=1)[:, np.newaxis] + self.vocab_size * self.beta)

    def calculate_theta_update(self):
        return (self.author_topic_matrix + self.alpha) / (self.author_topic_matrix.sum(axis=1)[:, np.newaxis] + self.num_topics * self.alpha)

    def calculate_topic_probabilities(self, doc_id, word_id, author):
        author_idx = self.authors.index(author)
        author_topic_totals = self.author_topic_matrix[author_idx]
        author_probs = author_topic_totals / sum(author_topic_totals) if sum(author_topic_totals) != 0 else np.ones(self.num_topics) / self.num_topics
        word_probs = self.calculate_phi_update()[:, word_id]
        combined_probs = author_probs * word_probs
        normalized_probs = combined_probs / combined_probs.sum()
        return normalized_probs

    def gibbs_sampling(self, iterations=10):
        for iteration in range(iterations):
            print(f"iteration: {iteration}")
            for doc_id, document in enumerate(self.corpus):
                for word_pos, (word_id, _) in enumerate(document):
                    current_topic = self.get_current_topic(doc_id, word_pos)
                    author = self.get_author(doc_id)
                    self.decrement_counts(doc_id, word_pos, current_topic, author)
                    topic_probs = self.calculate_topic_probabilities(doc_id, word_id, author)
                    if any(val < 0 for val in topic_probs):
                        print("Negative probabilities detected:", topic_probs)
        #Handle or fix the negative probabilities here

                    topic_probs = topic_probs / sum(topic_probs)
                    new_topic = np.random.choice(self.num_topics, p=topic_probs)
                    self.topic_assignments[doc_id][word_pos] = new_topic
                    self.word_topic_matrix[new_topic][word_id] += 1
                    self.topic_counts[new_topic] += 1
                    author_idx = self.authors.index(author)
                    self.author_topic_matrix[author_idx][new_topic] += 1
