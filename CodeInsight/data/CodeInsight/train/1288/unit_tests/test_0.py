var0 = pd.DataFrame({'A': [1, -2, 3], 'B': [-4, 5, 6]})
expected_result =  pd.DataFrame({'A': [1, 0, 3], 'B': [0, 5, 6]})
result = test(var0)
assert result.equals(expected_result), 'Test failed'