df0 = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})
var0 = None
expected_result =  pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})
result = test(df0, var0)
assert result.equals(expected_result), 'Test failed'