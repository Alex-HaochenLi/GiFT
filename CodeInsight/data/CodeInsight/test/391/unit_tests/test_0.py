arr0 = np.array([1, 2, 3, 4, 5])
var0 = 3
expected_result =  (np.array([2]),)
result = test(arr0, var0)
assert result == expected_result, 'Test failed'