
class Filtering(object):
    def __init__(self, data):
        self.data = data

    def filter_users_not_likes(self):
        return self.data[self.data['not_interested'] != 1]
