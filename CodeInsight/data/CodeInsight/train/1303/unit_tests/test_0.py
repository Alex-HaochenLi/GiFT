arr0 = np.array([1, 2, 3, np.nan])
expected_result =  True
result = test(arr0)
assert result == expected_result, 'Test failed'