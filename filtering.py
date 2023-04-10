
class Filtering(object):
    def __init__(self, data):
        self.data = data

    def filter_users_not_likes(self):
        return self.data[self.data['do_not_show_anymore'] != 1]
