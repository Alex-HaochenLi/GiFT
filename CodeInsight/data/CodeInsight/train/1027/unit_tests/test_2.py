matrix0 = np.array([[1,2],[3,4],[5,6],[7,8]])
var0 = 1
expected_result =  np.array([[1],[3],[5],[7]])
result = test(matrix0, var0)
assert np.array_equal(result, expected_result), 'Test failed'