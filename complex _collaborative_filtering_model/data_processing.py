import json
from src import data_structure
import collections
import time
import numpy as np


class data_provider(object):

    def __init__(self, json_location):
        self.json_location = json_location
        self.polls = []
        self.users = collections.OrderedDict()
        self.data = []
        self.train = []
        self.test = []
        self.validation = []
        self.train_proportion = 0.70
        self.validation_proportion = 0.15
        self.test_users = []
    def read_json(self):
        with open(self.json_location, 'r') as file:
            content = file.read()
        self.data = json.loads(content)


    def put_users_into_user_list(self):

        keys = self.data['users'].keys()
        # print(len(keys))
        for i in set(keys):
            self.users[i] = data_structure.user(i)
            self.users[i].lastActive = self.data['users'][i]["dateActivity"]

    def put_polls_into_poll_list_and_add_user_interaction(self):
        for i in self.data['polls']:
            # print(len(self.data['polls']))
            for j in self.data['polls'][i]:
                timestamp = np.random.uniform(low=(time.time() - 24 * 3600),high=time.time())
                title = self.data['polls'][i][j]['question']
                this = self.data['polls'][i][j]['this']
                that = self.data['polls'][i][j]['that']

                p = data_structure.poll(self.users[i].userid, j, timestamp, title, this, that)
                self.polls.append(p)
                # this is the case where user owns the poll
                self.users[i].add_interaction(j, 4, self.polls.index(p))

    def check_which_user_skip_or_vote_on_the_poll(self):
        for i in self.data['votes'].keys():  # I = pollid
            if next((x for x in self.polls if x.pollid == i), None) == None:
                continue
            pollidx = self.polls.index(next((x for x in self.polls if x.pollid == i), None))
            for j in self.data['votes'][i].keys():  # J is userid
                if j not in self.users.keys():
                    self.users[j] = data_structure.user(j)
                # this is where the user skips thwe poll
                if j in self.users.keys() and u'skip' in self.data['votes'][i][j].keys():
                    self.users[j].add_interaction(i, -1, pollidx)
                # this is where the user votes on the poll
                if j in self.users.keys() and u'this0that1removed2' in self.data['votes'][i][j].keys() and \
                        self.data['votes'][i][j][u'this0that1removed2'] in [0, 1]:
                    self.users[j].add_interaction(i, 1, pollidx)

    def check_which_user_comment_on_the_poll(self):
        for i in self.data['commentsPath'].keys():  # i = userID
            for j in self.data['commentsPath'][i].keys():  # j is pollId
                if next((x for x in self.polls if x.pollid == j), None) == None:
                    continue
                pollidx = self.polls.index(next((x for x in self.polls if x.pollid == j), None))
                # this is where the user comment on the poll
                self.users[i].add_interaction(j, 2, pollidx)

    def check_which_user_track_on_the_poll(self):
        for i in self.data['tracking'].keys():  # i = userID
            for j in self.data['tracking'][i].keys():  # j is unknown
                for k in self.data['tracking'][i][j].keys():
                    if next((x for x in self.polls if x.pollid == k), None) == None:
                        continue
                    pollidx = self.polls.index(next((x for x in self.polls if x.pollid == k), None))
                    if i not in self.users.keys():
                        self.users[i] = data_structure.user(i)
                        # this is where the user track on the poll
                    self.users[i].add_interaction(k, 3, pollidx)

    def set_initial_training_and_test_data(self):
        # indices = np.random.choice(range(len(self.users)), int(len(self.users)*self.train_proportion), replace=False)

        training_amount = int(len(self.users)*self.train_proportion)
        validation_amount = int(len(self.users)*self.validation_proportion)

        training_indices = np.arange(0, training_amount)
        validation_indices = np.arange(  training_amount, training_amount + validation_amount    )

        total_polls = len(self.polls)
        for idx,i in enumerate(self.users.keys()):
            if self.users[i].num_interactions() > 40 :
                poll_array = self.users[i].get_engagement_array(total_polls)

                if idx in training_indices:
                    self.train.append(poll_array)
                if idx in validation_indices:
                    self.validation.append(poll_array)
                else:
                    if self.users[i].num_interactions() > 40 and self.users[i].num_interactions() < 100:
                        self.test.append(poll_array)
                        self.test_users.append(self.users[i])

        self.train = np.asarray(self.train)
        self.test = np.asarray(self.test)
        self.validation = np.asarray(self.validation)


    def parse(self):

        self.read_json()

        self.put_users_into_user_list()

        self.put_polls_into_poll_list_and_add_user_interaction()

        self.check_which_user_skip_or_vote_on_the_poll()

        self.check_which_user_comment_on_the_poll()

        self.check_which_user_track_on_the_poll()

        self.set_initial_training_and_test_data()

# data_provider = data_provider(("/Users/caesar/PycharmProjects/my_honours_project/src/this_that.json"))
# data_provider.parse()
# print(data_provider.train)




