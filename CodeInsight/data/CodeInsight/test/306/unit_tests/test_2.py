arr0 = np.array([1, 2, 3])
arr1 = np.array([4, 5, 6])
expected_result =  np.array([1, 4, 2, 5, 3, 6])
result = test(arr0, arr1)
assert (result  ==  expected_result).all(), 'Test failed'