import string

def test(str0):
    return "".join(c for c in str0 if c not in string.punctuation)
