arr0 = np.array([[1, 2, 3], [4, 5, 6]])
expected_result =  [np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]
result = test(arr0)
assert all([np.array_equal(r, e) for r, e in zip(result, expected_result)]), 'Test failed'