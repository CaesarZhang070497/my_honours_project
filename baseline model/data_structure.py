import numpy as np


class user(object):
    """docstring for User"""

    def __init__(self, userid):
        self.userid = userid
        self.lastActive = 0
        self.polls = {}  # Key is pollId value is [interaction, index]

    def get_engagement_array(self, length):
        arr = np.zeros(length * 5)
        for i in self.polls.keys():
            interaction = self.dec_to_binary_array(self.polls[i][0])
            idx = self.polls[i][1]
            arr[idx * 5:idx * 5 + 5] = interaction
        return arr

    def dec_to_binary_array(self, num):
        arr = np.zeros((5))
        # tmp = map(int, list(bin(num)[2:]))
        # arr[5-len(tmp):] = tmp
        tmp = list(map(int, list(bin(num)[2:])))
        arr[5 - len(tmp):] = tmp
        return arr

    def add_interaction(self, pollid, interaction, pollidx):
        if pollid in self.polls.keys():
            self.polls[pollid][0] += interaction
        else:
            self.polls[pollid] = [interaction, pollidx]

    def num_interactions(self):
        return len(self.polls)

    def num_votes(self):
        votes = 0
        for i in self.polls.keys():
            bi = self.dec_to_binary_array(self.polls[i][0])
            votes += bi[-1]
        return votes


class poll(object):
    """docstring for Poll"""

    def __init__(self, userid, pollid, timestamp, title, this, that, boost=0):
        self.pollid = pollid
        self.userid = userid
        self.timestamp = timestamp
        self.title = title
        self.this = this
        self.that = that
        self.boost = boost

    def get_text(self):
        return self.title + ' ' + self.this + '' + self.that
