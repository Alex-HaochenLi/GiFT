arr0 = np.array([np.nan, np.nan])
expected_result =  np.array([])
result = test(arr0)
assert np.array_equal(result, expected_result), 'Test failed'