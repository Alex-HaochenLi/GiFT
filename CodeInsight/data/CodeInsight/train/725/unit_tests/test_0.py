lst0 = np.array([1,2,3])
var0 = 4
expected_result =  np.array([1,2,3,4])
result = test(lst0, var0)
assert np.array_equal(result, expected_result), 'Test failed'