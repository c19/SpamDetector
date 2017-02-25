class Vocabulary(dict):
    def __init__(self, *arg, **kwarg):
        super(Vocabulary, self).__init__(*arg, **kwarg)
        self.sealed = False  # after seal, all new word would return 0

    def __getitem__(self, word):
        if self.sealed:
            return self.get(word, 0)
        if word in self:
            return super(Vocabulary, self).__getitem__(word)
        else:
            super(Vocabulary, self).__setitem__(word, len(self) + 1)
            return super(Vocabulary, self).__getitem__(word)

    def seal(self):
        self.sealed = True