def test(lst0):
    return sorted(lst0, key=lambda x: (x < 0, abs(x)))
