import numpy as np


class user(object):
    """docstring for User"""

    def __init__(self, userid):
        self.userid = userid

        self.polls = {}  # Key is pollId value is [interaction, index]

    def get_engagement_array(self, length):
        arr = np.zeros(length)
        for i in self.polls.keys():

            interaction = self.polls[i][0]

            idx = self.polls[i][1]
            arr[idx] = interaction
        return arr


    def add_interaction(self, pollid, interaction, pollidx):
        if pollid in self.polls.keys():

            if self.polls[pollid][0]>=0 and interaction>=0:
                self.polls[pollid][0] += interaction

            if self.polls[pollid][0]>=0 and interaction<0:
                self.polls[pollid][0] = self.polls[pollid][0]

            if self.polls[pollid][0] < 0 and interaction >= 0:
                self.polls[pollid][0] = interaction

            if self.polls[pollid][0] < 0 and interaction < 0:
                self.polls[pollid][0] = interaction
        else:
            self.polls[pollid] = [interaction, pollidx]

    def num_interactions(self):
        return len(self.polls)


class poll(object):
    """docstring for Poll"""

    def __init__(self, userid, pollid, title, this, that, boost=0):
        self.pollid = pollid
        self.userid = userid
        self.title = title
        self.this = this
        self.that = that
        self.boost = boost

    def get_text(self):
        return self.title + ' ' + self.this + '' + self.that
